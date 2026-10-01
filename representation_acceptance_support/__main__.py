from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run

parser = argparse.ArgumentParser(description="Frozen-plan H13 label-free support proxy; never predictive/native parity evidence")
parser.add_argument("--profile", required=True, type=Path)
parser.add_argument("--frozen-method", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
arguments = parser.parse_args()
result = run(arguments.profile, arguments.frozen_method, arguments.output)
print(f"support_proxy_passed={result['support_gate']['passed']}; native_support_proved=False; model_fits=0")
