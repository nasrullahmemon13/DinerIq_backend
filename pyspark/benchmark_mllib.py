"""Actual Spark MLlib and independent Python benchmark entry point."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.verified_models import run


def run_benchmark():
    result = run()
    spark = result['engines']['spark']
    if spark['status'] != 'completed':
        raise RuntimeError(spark.get('error', 'Spark training failed'))
    return spark


if __name__ == '__main__':
    print(run_benchmark())
