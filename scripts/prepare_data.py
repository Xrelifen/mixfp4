#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_quant.data import prepare_c4, prepare_calibration, prepare_wikitext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--c4-count", type=int, default=256)
    parser.add_argument("--calibration-seq-len", type=int, default=128)
    parser.add_argument("--calibration-count", type=int, default=256)
    args = parser.parse_args()
    records = []
    for model in args.model:
        records.append(prepare_wikitext(model, args.seq_len))
        records.append(prepare_c4(model, args.seq_len, args.c4_count, 0))
        records.append(prepare_calibration(model, seq_len=args.calibration_seq_len, count=args.calibration_count))
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
