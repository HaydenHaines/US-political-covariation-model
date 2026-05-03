"""CLI entry point for the cross-tab impact report.

Usage (from wethervane repo root):
    python scripts/run_xt_impact_report.py

Prints a JSON summary with keys: mean_delta, max_delta, races_with_xt, report_date.
Exits 0 on success, 1 on error.
"""

import json
import sys
import traceback


def main() -> int:
    try:
        from src.prediction.forecast_engine import make_xt_impact_report

        result = make_xt_impact_report()

        summary = {
            "mean_delta": result["mean_delta"],
            "max_delta": result["max_delta"],
            "races_with_xt": result["races_with_xt"],
            "report_date": result["report_date"],
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
