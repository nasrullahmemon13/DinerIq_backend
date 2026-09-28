import json
from pathlib import Path
import pandas as pd
from src.verified_models import python_features, ROOT


def test_lag_features_do_not_use_future_or_same_day_target(tmp_path):
    records = []
    for day in range(80):
        for order in range(day % 8 + 1):
            records.append({'order_id':f'{day}-{order}', 'location_id':'A',
                            'order_date':pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)})
    source = tmp_path / 'orders.parquet'
    pd.DataFrame(records).to_parquet(source)
    before = python_features(source)
    changed = pd.DataFrame(records)
    changed = pd.concat([changed, pd.DataFrame([{'order_id':'future', 'location_id':'A', 'order_date':pd.Timestamp('2025-03-15')}])])
    changed.to_parquet(source)
    after = python_features(source)
    pd.testing.assert_frame_equal(before[before.order_date < '2025-03-15'], after[after.order_date < '2025-03-15'])
    row = before.iloc[0]
    assert row.lag_1 == 7
    assert row.lag_7 == 1
    assert row.mean_7 == 4


def test_real_independent_run_has_three_native_spark_models_and_matching_holdout():
    runs = [json.loads(p.read_text()) for p in (ROOT/'models/verified_runs').glob('*/manifest.json')]
    completed = [run for run in runs if run['status']=='completed']
    assert completed, 'No independently completed run'
    run = max(completed, key=lambda r:r['created_at'])
    assert run['engines']['spark']['engine']=='pyspark.ml'
    assert len(run['engines']['spark']['models']) >= 3
    assert len(run['engines']['python']['models']) >= 3
    result = pd.read_parquet(ROOT/'models/verified_runs'/run['run_id']/'comparison.parquet')
    assert len(result) >= 100
    assert result.record_id.is_unique
    assert (result.label_spark == result.label_python).all()
    assert (pd.to_datetime(result.order_date_spark) >= run['split']['test_start']).all()
    assert (result.difference.abs() > 0).any()
