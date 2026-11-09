"""Fail-closed server launch gates. No environment installation or automatic retry."""
from __future__ import annotations
import base64
import csv
import json
import ntpath
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_AUTHORIZATION = None
_SEARCH_RELATIVE_PATH = r'SystemApps\Microsoft.Windows.Search_cw5n1h2txyewy\SearchApp.exe'

# Fixed, read-only query. Only a validated decimal PID is substituted, never a
# process name/path. No elevation, execution-policy change, process stop or CUDA.
_WINDOWS_SEARCH_QUERY = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding -ArgumentList $false
$processId = __PROCESS_ID__
$system = @(Get-CimInstance -ClassName Win32_OperatingSystem)
if ($system.Count -ne 1) { throw 'Cannot identify Windows directory' }
$windowsDirectory = [string]$system[0].WindowsDirectory
$expectedPath = Join-Path $windowsDirectory 'SystemApps\Microsoft.Windows.Search_cw5n1h2txyewy\SearchApp.exe'
$before = @(Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $processId))
if ($before.Count -ne 1 -or -not $before[0].ExecutablePath -or -not $before[0].CreationDate) {
    throw 'Process identity unavailable'
}
if (-not [string]::Equals([string]$before[0].ExecutablePath, $expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Process is not the expected Windows Search binary'
}
$binary = Get-Item -LiteralPath $expectedPath -Force
if ($binary.PSIsContainer -or ($binary.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Search executable must be a regular system file'
}
$signature = Get-AuthenticodeSignature -LiteralPath $expectedPath
if (-not $signature.SignerCertificate) { throw 'Signer unavailable' }
$after = @(Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $processId))
if ($after.Count -ne 1 -or -not $after[0].ExecutablePath -or -not $after[0].CreationDate) {
    throw 'Process identity disappeared'
}
[ordered]@{
    windows_directory = $windowsDirectory
    before_pid = [int64]$before[0].ProcessId
    before_path = [string]$before[0].ExecutablePath
    before_creation_utc = $before[0].CreationDate.ToUniversalTime().ToString('o')
    after_pid = [int64]$after[0].ProcessId
    after_path = [string]$after[0].ExecutablePath
    after_creation_utc = $after[0].CreationDate.ToUniversalTime().ToString('o')
    signature_status = [string]$signature.Status
    signer_name = $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false)
    signer_subject = [string]$signature.SignerCertificate.Subject
    signer_thumbprint = [string]$signature.SignerCertificate.Thumbprint
} | ConvertTo-Json -Compress
'''


def _parse_compute_processes(output):
    """Accept only complete NVIDIA CSV rows; ambiguous output stays blocked."""
    rows, seen = [], set()
    try:
        for fields in csv.reader(output.splitlines(), skipinitialspace=True, strict=True):
            if not fields or all(not value.strip() for value in fields):
                continue
            if len(fields) != 2 or not re.fullmatch(r'[1-9][0-9]*', fields[0].strip()):
                raise ValueError('expected positive PID and executable path')
            process_id, name = int(fields[0].strip()), fields[1].strip()
            if process_id > 4294967295 or process_id in seen or not name or any(ord(char) < 32 for char in name):
                raise ValueError('invalid/duplicate PID or executable path')
            seen.add(process_id)
            rows.append((process_id, name))
    except (csv.Error, ValueError) as exc:
        raise RuntimeError('Unable to verify NVIDIA compute-process output; refusing launch') from exc
    return rows


def _query_windows_search_identity(process_id):
    if sys.platform != 'win32' or type(process_id) is not int or not 0 < process_id <= 4294967295:
        raise RuntimeError('Windows Search identity query requires Windows and a positive PID')
    script = _WINDOWS_SEARCH_QUERY.replace('__PROCESS_ID__', str(process_id))
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    try:
        result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive',
                                 '-EncodedCommand', encoded], check=True, capture_output=True,
                                text=True, encoding='utf-8', timeout=20)
        identity = json.loads(result.stdout.lstrip('\ufeff'))
        if not isinstance(identity, dict):
            raise ValueError('Expected a single identity object')
        return identity
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        raise RuntimeError('Cannot verify Windows Search identity/signature; refusing launch (no process was stopped)') from exc


def _verified_windows_search_process(process_id, reported_path):
    """Narrow exception for the signed, live Windows 10 SearchApp system image."""
    if sys.platform != 'win32':
        return None
    # Do not accept basenames, SearchHost, traversal, arbitrary packages or paths.
    candidate_pattern = r'[A-Za-z]:\\Windows\\' + re.escape(_SEARCH_RELATIVE_PATH)
    if not re.fullmatch(candidate_pattern, reported_path, flags=re.I):
        return None
    identity = _query_windows_search_identity(process_id)
    windows_directory = identity.get('windows_directory')
    if not isinstance(windows_directory, str) or not re.fullmatch(r'[A-Za-z]:\\Windows', windows_directory, flags=re.I):
        raise RuntimeError('Windows directory identity is invalid; refusing launch')
    expected = ntpath.normcase(ntpath.join(windows_directory, _SEARCH_RELATIVE_PATH))
    creation = identity.get('before_creation_utc')
    paths = (reported_path, identity.get('before_path'), identity.get('after_path'))
    signer_name = identity.get('signer_name')
    signer_subject = identity.get('signer_subject')
    if not (all(isinstance(path, str) and ntpath.normcase(path) == expected for path in paths)
            and type(identity.get('before_pid')) is int and identity['before_pid'] == process_id
            and type(identity.get('after_pid')) is int and identity['after_pid'] == process_id
            and isinstance(creation, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z', creation)
            and creation == identity.get('after_creation_utc')
            and identity.get('signature_status') == 'Valid'
            and signer_name in ('Microsoft Windows', 'Microsoft Corporation')
            and isinstance(signer_subject, str) and re.search(r'(?:^|,\s*)O=Microsoft Corporation(?:,|$)', signer_subject)
            and isinstance(identity.get('signer_thumbprint'), str)
            and re.fullmatch(r'[A-Fa-f0-9]{40}', identity['signer_thumbprint'])):
        raise RuntimeError('Windows Search process/signature identity did not verify; refusing launch')
    return {'pid': process_id, 'executable': identity['before_path'], 'creation_utc': creation,
            'signature_status': 'Valid', 'signer_name': signer_name,
            'signer_thumbprint': identity['signer_thumbprint']}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_environment():
    # Check the actual interpreter, not an unrelated CONDA_DEFAULT_ENV label.
    if Path(sys.prefix).name.lower() != 'radiodiffzcy':
        raise RuntimeError('Activate the real radiodiffzcy environment first. Current prefix: ' + sys.prefix)
    import torch
    match = re.match(r'(\d+)\.(\d+)', torch.__version__)
    if not match or tuple(map(int, match.groups())) < (1, 12):
        raise RuntimeError('PyTorch >= 1.12 is required; no environment will be modified automatically')


def gpu_snapshot(index):
    command = ['nvidia-smi', '-i', str(index),
               '--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError('Unable to identify exactly one selected GPU')
    fields = [part.strip() for part in lines[0].split(',')]
    if len(fields) != 6 or not re.search(r'\bRTX\s+3090\b', fields[1], re.I):
        raise RuntimeError('This package is targeted to RTX 3090; refusing another GPU: ' + lines[0])
    snapshot = dict(index=int(fields[0]), name=fields[1], uuid=fields[2],
                    total_mib=int(fields[3]), free_mib=int(fields[4]), utilization=int(fields[5]))
    processes = subprocess.run(['nvidia-smi', '-i', str(index),
                               '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                              check=True, capture_output=True, text=True, timeout=15)
    active, verified = [], []
    for process_id, name in _parse_compute_processes(processes.stdout):
        desktop = _verified_windows_search_process(process_id, name)
        if desktop is None:
            active.append('%d, %s' % (process_id, name))
        else:
            verified.append(desktop)
    if active:
        raise RuntimeError('Selected GPU already has compute processes; no concurrent training started: ' + '; '.join(active))
    if snapshot['free_mib'] < 20000 or snapshot['utilization'] > 10:
        raise RuntimeError('GPU must be idle with >=20000 MiB free; waiting is manual, no automatic retry')
    snapshot['verified_windows_desktop_processes'] = verified
    return snapshot


def require_idle_gpu(index):
    first = gpu_snapshot(index)
    time.sleep(1)
    second = gpu_snapshot(index)
    if first['uuid'] != second['uuid']:
        raise RuntimeError('GPU identity changed during readiness check')
    if second.get('verified_windows_desktop_processes'):
        print('GPU check: verified signed Windows Search desktop process only; '
              'training-process, free-memory and utilization checks remain enabled.', flush=True)
    return second


def bind_gpu_uuid(gpu):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before GPU UUID binding; refusing an ambiguous device')
    if not gpu['uuid'].startswith('GPU-'):
        raise ValueError('Expected a physical NVIDIA GPU UUID')
    # nvidia-smi index and CUDA index need not have the same enumeration order.
    # Mask by the identity that was actually inspected, before CUDA initialization.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']


def validate_preflight(proof, config, engine, gpu=None):
    if not (proof.get('state') == 'passed' and proof.get('weights_discarded') is True
            and proof.get('train_only') is True and proof.get('formal_optimizer_steps') == 0):
        raise ValueError('Full-size preflight is not a valid disposable train-only check')
    if proof.get('protocol_sha256') != engine.content_hash(config) or proof.get('source_sha256') != config['source_sha256']:
        raise ValueError('Preflight does not match this exact runtime/source/data protocol')
    if proof.get('condition_shape') != [2, 9, 4, 256, 256]:
        raise ValueError('Preflight must retain batch2 / four heights / full 256x256')
    if gpu is not None and (proof.get('cuda_visible_devices') != gpu['uuid']
                            or proof.get('cuda_device_index') != 0
                            or not re.search(r'\bRTX\s+3090\b', proof.get('gpu_name', ''), re.I)):
        raise ValueError('Actual CUDA preflight device is not bound to the inspected 3090 UUID')


def require_training_authorization(config):
    from scripts.run_prediction_10plus10_v2 import content_hash
    if not _AUTHORIZATION or _AUTHORIZATION.get('pid') != os.getpid() or _AUTHORIZATION.get('protocol_sha256') != content_hash(config):
        raise RuntimeError('Use python -B run.py; a verified 3090 full-size preflight is required before execution')


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file
    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.file.flush()


def launch(options):
    global _AUTHORIZATION
    from scripts import run_prediction_transformer_v3 as runner
    from package_integrity import verify_package
    verify_package()
    # Pure metadata inspection works outside radiodiffzcy and never probes CUDA.
    cli = ['--action', 'check' if options.action == 'preflight' else options.action,
           '--dataset-root', options.dataset_root, '--log-every', str(options.log_every)]
    if options.resume:
        cli.append('--resume')
    if options.action == 'check':
        runner.main(cli)
        print(json.dumps({'python': sys.version.split()[0], 'environment_prefix': sys.prefix,
                          'target_environment': 'radiodiffzcy', 'target_gpu': 'RTX 3090',
                          'gpu_preflight_performed': False}, indent=2))
        return
    require_environment()
    args = runner.parse_args(cli)
    if options.action == 'preflight':
        import torch
        args.device = torch.device('cuda')
        args.action = 'train'
    runner.validate_output_path(args.output)
    engine = runner.make_engine()
    config = engine.build_protocol(args, engine.audit(args.dataset_root))
    runner.validate_output_state(engine, args, config)
    control = ROOT / 'server_run'
    runner.reject_reparse(control)
    with runner.exclusive_run_lock(control):
        # Revalidate under lock: no duplicate run, old results are never reset.
        runner.validate_output_state(engine, args, config)
        gpu = require_idle_gpu(options.gpu)
        bind_gpu_uuid(gpu)
        for name in ('status.json', 'training.stdout.log', 'training.stderr.log', 'preflight'):
            runner.reject_reparse(control / name)
        status = {'state': 'preflight', 'time_utc': utc_now(), 'pid': os.getpid(),
                  'executable': sys.executable, 'arguments': sys.argv, 'gpu': gpu,
                  'output': str(args.output), 'action': options.action}
        engine.dump(control / 'status.json', status)
        with (control / 'training.stdout.log').open('a', encoding='utf-8') as stdout, \
             (control / 'training.stderr.log').open('a', encoding='utf-8') as stderr, \
             redirect_stdout(Tee(sys.stdout, stdout)), redirect_stderr(Tee(sys.stderr, stderr)):
            try:
                from server_preflight import run_preflight
                proof_dir = control / 'preflight'
                proof_dir.mkdir(exist_ok=True)
                proof_path = proof_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8] + '.json')
                run_preflight(runner, args, proof_path)
                proof = engine.read_json(proof_path)
                validate_preflight(proof, config, engine, gpu)
                status.update(state='preflight_passed', preflight=str(proof_path), time_utc=utc_now())
                engine.dump(control / 'status.json', status)
                if options.action == 'preflight':
                    print('PREFLIGHT_PASSED: weights discarded; no formal training started', flush=True)
                    return
                _AUTHORIZATION = {'pid': os.getpid(), 'protocol_sha256': engine.content_hash(config)}
                status.update(state='running', time_utc=utc_now())
                engine.dump(control / 'status.json', status)
                runner.main(cli)
                status.update(state='completed', time_utc=utc_now())
                engine.dump(control / 'status.json', status)
            except BaseException as exc:
                import traceback
                traceback.print_exc()
                status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              time_utc=utc_now(), error=repr(exc))
                engine.dump(control / 'status.json', status)
                raise
            finally:
                _AUTHORIZATION = None
