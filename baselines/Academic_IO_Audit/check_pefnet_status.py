"""Record why the bundled PEFNet-inspired adaptation is not table-ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=r"D:\zdata\3DAFF-RadioDiff_Baselines\PEFNet\configs\pefnet_inspired_adapted.yaml",
    )
    parser.add_argument("--output", default="pefnet_io_report.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample_rate = float(config["experiment"]["sample_rate"])
    channels = list(config["model"]["input_channel_names"])
    report = {
        "status": "BLOCKED_FROM_COMMON_TABLE",
        "config": str(config_path),
        "method_label": config["experiment"]["method_label"],
        "implementation_status": config["experiment"]["implementation_status"],
        "sample_rate": sample_rate,
        "input_channels": channels,
        "output": "two-channel field plus one-channel residual path-loss estimate",
        "reasons": [
            "Configured sample_rate is 0.0, not the common random 10% protocol.",
            "The adaptation uses a separate manifest/dataset and evaluation support.",
            "It is paper-inspired, not an official or faithful PEFNet reproduction.",
            "No value from this configuration may substantiate the common comparison table without alignment and retraining.",
        ],
    }
    if sample_rate != 0.0:
        raise AssertionError("Unexpected PEFNet configuration; audit assumptions require review")
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"PEFNET STATUS RECORDED: {output.resolve()}")


if __name__ == "__main__":
    main()
