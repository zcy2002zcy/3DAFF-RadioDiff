"""CPU-only launch guard tests. No real subprocess, CUDA or dataset access."""
from __future__ import annotations
import json
import base64
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server_launch as launch
import torch


def completed(stdout):
    return SimpleNamespace(stdout=stdout)


SEARCH_PATH = r'C:\Windows\SystemApps\Microsoft.Windows.Search_cw5n1h2txyewy\SearchApp.exe'


def search_identity():
    return dict(windows_directory=r'C:\Windows', before_pid=16896, after_pid=16896,
                before_path=SEARCH_PATH, after_path=SEARCH_PATH,
                before_creation_utc='2026-10-28T10:01:02.1234567Z',
                after_creation_utc='2026-10-28T10:01:02.1234567Z',
                signature_status='Valid', signer_name='Microsoft Windows',
                signer_subject='CN=Microsoft Windows, O=Microsoft Corporation, C=US',
                signer_thumbprint='A' * 40)


class ServerLaunchTests(unittest.TestCase):
    def test_windows_signed_search_identity_allowed(self):
        with patch.object(sys, 'platform', 'win32'), patch.object(launch, '_query_windows_search_identity', return_value=search_identity()) as query:
            record = launch._verified_windows_search_process(16896, SEARCH_PATH)
        query.assert_called_once_with(16896)
        self.assertEqual(record['pid'], 16896)
        self.assertEqual(record['signature_status'], 'Valid')

    def test_search_basename_other_directory_and_other_program_blocked(self):
        names = ['SearchApp.exe', r'D:\apps\SearchApp.exe', SEARCH_PATH.replace('SearchApp.exe', 'SearchHost.exe'),
                 SEARCH_PATH.replace('SystemApps', 'SystemApps2'), SEARCH_PATH.replace('SearchApp.exe', '..\\SearchApp.exe'),
                 r'C:\Users\User\SearchApp.exe', 'python.exe']
        with patch.object(sys, 'platform', 'win32'), patch.object(launch, '_query_windows_search_identity') as query:
            for name in names:
                with self.subTest(name=name):
                    self.assertIsNone(launch._verified_windows_search_process(16896, name))
        query.assert_not_called()

    def test_linux_does_not_apply_windows_exception(self):
        with patch.object(sys, 'platform', 'linux'), patch.object(launch, '_query_windows_search_identity') as query:
            self.assertIsNone(launch._verified_windows_search_process(16896, SEARCH_PATH))
        query.assert_not_called()

    def test_search_identity_mismatch_fails_closed(self):
        bad_values = [('before_pid', 99), ('after_pid', 99), ('after_pid', True),
                      ('before_path', r'D:\SearchApp.exe'), ('after_path', r'C:\Windows\python.exe'),
                      ('windows_directory', r'D:\Windows'), ('before_creation_utc', None),
                      ('after_creation_utc', '2026-10-28T11:01:02.1234567Z'),
                      ('signature_status', 'NotSigned'), ('signature_status', 'UnknownError'),
                      ('signer_name', 'Untrusted'), ('signer_subject', 'CN=Microsoft Windows, O=Fake'),
                      ('signer_thumbprint', '')]
        for field, value in bad_values:
            with self.subTest(field=field, value=value), patch.object(sys, 'platform', 'win32'), \
                 patch.object(launch, '_query_windows_search_identity', return_value=dict(search_identity(), **{field: value})):
                with self.assertRaisesRegex(RuntimeError, 'refusing'):
                    launch._verified_windows_search_process(16896, SEARCH_PATH)

    def test_missing_identity_fields_fail_closed(self):
        for field in search_identity():
            identity = search_identity()
            del identity[field]
            with self.subTest(field=field), patch.object(sys, 'platform', 'win32'), \
                 patch.object(launch, '_query_windows_search_identity', return_value=identity):
                with self.assertRaises(RuntimeError):
                    launch._verified_windows_search_process(16896, SEARCH_PATH)

    def test_windows_query_is_readonly_encoded_and_pid_only(self):
        with patch.object(sys, 'platform', 'win32'), patch.object(launch.subprocess, 'run', return_value=completed(json.dumps(search_identity()))) as run:
            self.assertEqual(launch._query_windows_search_identity(16896), search_identity())
        args, kwargs = run.call_args
        command = args[0]
        script = base64.b64decode(command[-1]).decode('utf-16-le')
        self.assertEqual(command[:5], ['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand'])
        self.assertIn('$processId = 16896', script)
        self.assertIn('Get-AuthenticodeSignature -LiteralPath', script)
        self.assertNotIn('__PROCESS_ID__', script)
        self.assertNotIn('Stop-Process', script)
        self.assertNotIn('ExecutionPolicy', script)
        self.assertEqual(kwargs['timeout'], 20)
        self.assertTrue(kwargs['check'])
        self.assertEqual(kwargs['encoding'], 'utf-8')

    def test_invalid_query_pid_or_platform_never_runs(self):
        for platform, pid in [('linux', 16896), ('win32', 0), ('win32', True), ('win32', '1;stop-process'), ('win32', -1)]:
            with self.subTest(platform=platform, pid=pid), patch.object(sys, 'platform', platform), patch.object(launch.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    launch._query_windows_search_identity(pid)
                run.assert_not_called()

    def test_query_errors_and_malformed_json_fail_closed(self):
        errors = [subprocess.TimeoutExpired('powershell', 20), subprocess.CalledProcessError(1, 'powershell'), PermissionError('denied')]
        for error in errors:
            with self.subTest(error=error), patch.object(sys, 'platform', 'win32'), patch.object(launch.subprocess, 'run', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                    launch._query_windows_search_identity(16896)
        for output in ('', 'garbage', 'null', '[]', '{} {}'):
            with self.subTest(output=output), patch.object(sys, 'platform', 'win32'), patch.object(launch.subprocess, 'run', return_value=completed(output)):
                with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                    launch._query_windows_search_identity(16896)

    def test_strict_process_csv(self):
        self.assertEqual(launch._parse_compute_processes(''), [])
        self.assertEqual(launch._parse_compute_processes('12, "C:\\a,b\\python.exe"\n'), [(12, r'C:\a,b\python.exe')])
        for row in ('0,python', '-1,python', '1,', 'a,python', '1,p,x', '1,p\n1,p', 'No running mystery', '1,"broken', '4294967296,p'):
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                launch._parse_compute_processes(row)

    def test_gpu_snapshot_records_verified_search_only(self):
        results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, 23000, 0\n'), completed('16896, ' + SEARCH_PATH + '\n')]
        with patch.object(sys, 'platform', 'win32'), patch.object(launch.subprocess, 'run', side_effect=results), \
             patch.object(launch, '_query_windows_search_identity', return_value=search_identity()):
            snapshot = launch.gpu_snapshot(0)
        self.assertEqual(snapshot['verified_windows_desktop_processes'][0]['pid'], 16896)

    def test_verified_search_plus_python_still_blocked(self):
        results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, 23000, 0\n'), completed('16896, ' + SEARCH_PATH + '\n12, python.exe\n')]
        with patch.object(sys, 'platform', 'win32'), patch.object(launch.subprocess, 'run', side_effect=results), \
             patch.object(launch, '_query_windows_search_identity', return_value=search_identity()):
            with self.assertRaisesRegex(RuntimeError, '12, python.exe'):
                launch.gpu_snapshot(0)

    def test_verified_search_does_not_bypass_gpu_load_checks(self):
        for free, util in ((12000, 0), (23000, 50)):
            results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, %d, %d\n' % (free, util)), completed('16896, ' + SEARCH_PATH + '\n')]
            with self.subTest(free=free, util=util), patch.object(sys, 'platform', 'win32'), \
                 patch.object(launch.subprocess, 'run', side_effect=results), patch.object(launch, '_query_windows_search_identity', return_value=search_identity()):
                with self.assertRaisesRegex(RuntimeError, 'idle'):
                    launch.gpu_snapshot(0)

    def test_second_snapshot_compute_process_still_blocked(self):
        with patch.object(launch, 'gpu_snapshot', side_effect=[{'uuid': 'a'}, RuntimeError('compute processes')]), patch.object(launch.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'compute processes'):
                launch.require_idle_gpu(0)

    def test_environment_uses_real_prefix_not_env_label(self):
        with patch.object(sys, 'prefix', '/opt/conda/envs/not_the_requested_env'):
            with self.assertRaisesRegex(RuntimeError, 'radiodiffzcy'):
                launch.require_environment()

    def test_reject_wrong_gpu(self):
        with patch.object(launch.subprocess, 'run', return_value=completed('0, NVIDIA GeForce RTX 5070 Ti, GPU-a, 16303, 16000, 0\n')):
            with self.assertRaisesRegex(RuntimeError, '3090'):
                launch.gpu_snapshot(0)

    def test_reject_compute_process_even_if_gpu_low_utilization(self):
        results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, 23000, 0\n'), completed('12, python\n')]
        with patch.object(launch.subprocess, 'run', side_effect=results):
            with self.assertRaisesRegex(RuntimeError, 'compute processes'):
                launch.gpu_snapshot(0)

    def test_accept_idle_3090(self):
        results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, 23000, 0\n'), completed('')]
        with patch.object(launch.subprocess, 'run', side_effect=results):
            snapshot = launch.gpu_snapshot(0)
        self.assertEqual(snapshot['uuid'], 'GPU-a')

    def test_reject_low_memory_or_busy_gpu(self):
        for free, util in ((12000, 0), (23000, 50)):
            results = [completed('0, NVIDIA GeForce RTX 3090, GPU-a, 24576, %d, %d\n' % (free, util)), completed('')]
            with patch.object(launch.subprocess, 'run', side_effect=results):
                with self.assertRaisesRegex(RuntimeError, 'idle'):
                    launch.gpu_snapshot(0)

    def test_reject_identity_change(self):
        with patch.object(launch, 'gpu_snapshot', side_effect=[{'uuid': 'a'}, {'uuid': 'b'}]), patch.object(launch.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                launch.require_idle_gpu(0)

    def test_bind_uses_checked_uuid_before_cuda_initialization(self):
        with patch.object(torch.cuda, 'is_initialized', return_value=False), patch.dict(launch.os.environ, {}, clear=False):
            launch.bind_gpu_uuid({'uuid': 'GPU-checked-card'})
            self.assertEqual(launch.os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-checked-card')
        with patch.object(torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'initialized'):
                launch.bind_gpu_uuid({'uuid': 'GPU-checked-card'})

    def test_preflight_rejects_wrong_actual_cuda_device(self):
        config = {'source_sha256': {'a': 'abc'}}
        engine = SimpleNamespace(content_hash=lambda _: 'hash')
        proof = dict(state='passed', weights_discarded=True, train_only=True, formal_optimizer_steps=0,
                     protocol_sha256='hash', source_sha256={'a': 'abc'}, condition_shape=[2, 9, 4, 256, 256],
                     cuda_visible_devices='GPU-other', cuda_device_index=0, gpu_name='NVIDIA GeForce RTX 3090')
        with self.assertRaisesRegex(ValueError, 'UUID'):
            launch.validate_preflight(proof, config, engine, {'uuid': 'GPU-checked'})
        proof['cuda_visible_devices'] = 'GPU-checked'
        launch.validate_preflight(proof, config, engine, {'uuid': 'GPU-checked'})

    def test_preflight_binding_and_full_size(self):
        config = {'source_sha256': {'a': 'abc'}}
        engine = SimpleNamespace(content_hash=lambda _: 'hash')
        proof = dict(state='passed', weights_discarded=True, train_only=True, formal_optimizer_steps=0,
                     protocol_sha256='hash', source_sha256={'a': 'abc'}, condition_shape=[2, 9, 4, 256, 256])
        launch.validate_preflight(proof, config, engine)
        for field, bad in (('protocol_sha256', 'wrong'), ('condition_shape', [1, 9, 4, 16, 16]),
                           ('weights_discarded', False), ('train_only', False), ('state', 'failed')):
            with self.assertRaises(ValueError):
                launch.validate_preflight(dict(proof, **{field: bad}), config, engine)


if __name__ == '__main__':
    unittest.main()
