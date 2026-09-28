"""Measured file inventory, profiling and join diagnostics; never synthetic telemetry."""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CLEANED = ROOT / 'processed_data/cleaned'


def inventory():
    tables = []
    for folder in sorted(CLEANED.iterdir()):
        files = list(folder.glob('*.parquet')) if folder.is_dir() else []
        if not files:
            continue
        tables.append({'name': folder.name, 'format': 'Parquet', 'storage_type': 'Local files',
                       'records': sum(pq.read_metadata(p).num_rows for p in files),
                       'partitions': len(files), 'schema_mode': 'Stored Parquet schema',
                       'size_mb': round(sum(p.stat().st_size for p in files)/1048576, 3),
                       'files': [str(p.relative_to(ROOT)) for p in files]})
    return {'status': 'success', 'tables': tables, 'summary': {
        'total_tables': len(tables), 'total_records': sum(t['records'] for t in tables),
        'total_size_mb': sum(t['size_mb'] for t in tables),
        'ingestion_status': 'Measured files on disk; file count is not a live Spark partition count'}}


def profile():
    checks, columns = [], []
    for folder in sorted(CLEANED.iterdir()):
        for path in folder.glob('*.parquet') if folder.is_dir() else []:
            parquet = pq.ParquetFile(path)
            for col in parquet.schema_arrow.names:
                if col.startswith('__index_level_'):
                    continue
                values = parquet.read(columns=[col]).to_pandas(ignore_metadata=True)[col]
                missing = int(values.isna().sum())
                invalid = pd.Series(False, index=values.index)
                rule = 'Non-null value'
                if col.endswith('_date'):
                    invalid = values.notna() & pd.to_datetime(values, errors='coerce').isna()
                    rule = 'Valid calendar date'
                elif col in ['quantity', 'unit_price', 'base_price', 'cost_price', 'quantity_wasted', 'discount_amount']:
                    numeric = pd.to_numeric(values, errors='coerce')
                    invalid = values.notna() & (numeric.isna() | (numeric < 0))
                    rule = 'Numeric and non-negative'
                elif col in ['overall_rating', 'food_rating', 'service_rating']:
                    numeric = pd.to_numeric(values, errors='coerce')
                    invalid = values.notna() & (~numeric.between(1, 5))
                    rule = 'Numeric rating from 1 to 5'
                count = int(invalid.sum())
                columns.append({'dataset': folder.name, 'column': col, 'dtype': str(values.dtype),
                                'rows': len(values), 'null_count': missing, 'invalid_count': count,
                                'completeness_pct': 100*(len(values)-missing)/len(values) if len(values) else None,
                                'invalid_samples': values[invalid].astype(str).head(5).tolist()})
                checks.append({'check_id': f'{folder.name}.{col}', 'name': f'{folder.name}: {col}',
                               'rule': rule, 'status': 'Issues found' if count or missing else 'Passed',
                               'anomalies_detected': count + missing,
                               'pass_rate_pct': round(100*(len(values)-count-missing)/len(values), 2) if len(values) else None})
            key = {'orders':'order_id', 'order_items':'order_item_id', 'customers':'customer_id',
                   'menu_items':'item_id', 'ratings':'rating_id', 'wastage':'wastage_id'}.get(folder.name)
            if key and key in parquet.schema_arrow.names:
                values = parquet.read(columns=[key]).to_pandas()[key]
                duplicate = int(values.duplicated().sum())
                checks.append({'check_id': f'{folder.name}.duplicates', 'name': f'{folder.name}: duplicate identifiers',
                               'rule': f'Unique {key}', 'status': 'Issues found' if duplicate else 'Passed',
                               'anomalies_detected': duplicate, 'pass_rate_pct': 100*(len(values)-duplicate)/len(values) if len(values) else None})
    return {'status':'success', 'checks':checks, 'columns':columns, 'summary': {
        'total_rules_evaluated':len(checks), 'passed_rules':sum(c['status']=='Passed' for c in checks),
        'overall_health_score_pct': None,
        'score_formula':'No combined score: nullability and business rules need dataset-specific interpretation.',
        'audit_timestamp':datetime.now(timezone.utc).isoformat()}}


RELATIONSHIPS = [('orders','customers','customer_id'), ('orders','order_items','order_id'),
                 ('order_items','menu_items','item_id'), ('menu_items','menu_categories','category_id'),
                 ('orders','restaurants','location_id'), ('orders','promotions','promotion_id'),
                 ('menu_items','pricing_history','item_id'), ('menu_items','ratings','item_id'),
                 ('menu_items','inventory','item_id'), ('menu_items','wastage','item_id')]


def joins():
    rows = []
    for i, (left, right, key) in enumerate(RELATIONSHIPS):
        record = {'join_id':f'J-{i+1:02}', 'primary_table':left, 'joined_table':right,
                  'join_key':key, 'join_type':'LEFT JOIN diagnostic', 'purpose':'Source-key cardinality and unmatched-record audit'}
        try:
            a = pd.read_parquet(CLEANED / left / f'{left}.parquet', columns=[key])[key]
            b = pd.read_parquet(CLEANED / right / f'{right}.parquet', columns=[key])[key]
            counts = b.dropna().value_counts()
            matches = a.map(counts).fillna(0)
            output = int(matches.clip(lower=1).sum())
            record.update(input_rows=len(a), right_rows=len(b), records_joined=output,
                          unmatched_rows=int((matches==0).sum()), duplicate_right_keys=int((counts>1).sum()),
                          row_multiplier=output/len(a) if len(a) else None,
                          integrity_pct=100*float((matches>0).mean()) if len(a) else None,
                          status='measured')
        except (OSError, KeyError, ValueError) as error:
            record.update(status='unavailable', error=str(error), records_joined=None, integrity_pct=None)
        rows.append(record)
    return {'status':'success', 'joins':rows, 'summary':{'total_required_joins':len(rows),
            'successful_joins':sum(r['status']=='measured' for r in rows),
            'note':'Independent relationship diagnostics; not proof of an executed integrated Spark job.'}}


def quarantine():
    path = ROOT / 'processed_data/quarantine/quarantine_manifest.json'
    if not path.exists():
        return {'status':'unavailable', 'batches':[], 'total_quarantined_records':None}
    return json.loads(path.read_text(encoding='utf-8'))


def spark_status():
    try:
        from pyspark.sql import SparkSession
        session = SparkSession.getActiveSession()
    except (ImportError, RuntimeError) as error:
        return {'status': 'unavailable', 'cluster_info': {}, 'recent_spark_jobs': [],
                'message': f'Spark runtime is unavailable in this API process: {error}'}
    info = {}
    if session:
        info = {'spark_version':session.version, 'master':session.sparkContext.master,
                'app_name':session.sparkContext.appName,
                'default_parallelism':session.sparkContext.defaultParallelism,
                'sql_shuffle_partitions':session.conf.get('spark.sql.shuffle.partitions'),
                'parquet_compression':session.conf.get('spark.sql.parquet.compression.codec')}
    return {'status':'active' if session else 'unavailable', 'cluster_info':info,
            'recent_spark_jobs':[], 'message':'No live Spark context in this API process. Run python -m src.verified_models; measured run history is in Models.' if not session else 'Measured current process Spark context'}
