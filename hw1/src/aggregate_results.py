# -*- coding: utf-8 -*-
"""汇总六组实验，拒绝混用不同划分或旧格式结果。"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.experiment import EXPERIMENTS, load_results
from src.utils import RESULTS_DIR, load_nyt_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true",
                        help="缺少任何一组实验结果时返回非零退出码，供提交前检查。")
    args = parser.parse_args()
    protocol = load_nyt_splits(verbose=False)["metadata"]["protocol_id"]
    records = load_results(expected_protocol_id=protocol)
    rows = []
    for name, display_name, _ in EXPERIMENTS:
        data = records.get(name)
        rows.append({
            "experiment": name, "method": display_name,
            "accuracy": data["accuracy"] if data else None,
            "macro_f1": data["macro_f1"] if data else None,
            "val_accuracy": data["val_accuracy"] if data else None,
            "val_macro_f1": data["val_macro_f1"] if data else None,
            "total_seconds": data["total_seconds"] if data else None,
            "status": "complete" if data else "not_run",
        })
        if data and data.get("diagnostics", {}).get("convergence", {}).get("converged") is False:
            print(f"WARNING: {name} 未确认收敛，请查看该实验的 diagnostics。")
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, na_rep="-", float_format=lambda value: f"{value:.4f}"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"已完成 {len(records)}/{len(EXPERIMENTS)} 组实验。")
    if args.require_complete and len(records) != len(EXPERIMENTS):
        raise SystemExit("实验结果不完整，请先补齐上述 not_run 项。")


if __name__ == "__main__":
    main()
