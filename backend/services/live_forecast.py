from pathlib import Path
import hashlib
import json
import pandas as pd
import joblib
from src.verified_models import SOURCE, FEATURES, python_features

ROOT = Path(__file__).resolve().parents[2]


def forecast(horizon=14, location=None, selection=None):
    candidates = []
    for path in (ROOT / 'models/verified_runs').glob('*/manifest.json'):
        data = json.loads(path.read_text())
        if data.get('engines', {}).get('python', {}).get('status') == 'completed' and (path.parent / 'python_model.joblib').exists():
            candidates.append((data['created_at'], path, data))
    if not candidates:
        raise ValueError('No trained Python demand model available. Run python -m src.verified_models.')
    if selection:
        candidates = [item for item in candidates if item[2]['run_id'] == selection['run_id']]
        if not candidates:
            raise ValueError('Active model run is unavailable')
    _, path, run = max(candidates, key=lambda item:item[0])
    engine_name = selection['engine'] if selection else 'python'
    engine = run['engines'][engine_name]
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != run['dataset_sha256']:
        raise ValueError('Dataset changed since training. Train a new version before forecasting.')
    frame = python_features(SOURCE)
    locations = sorted(frame.location_id.unique().tolist())
    if location and location not in locations:
        raise ValueError('Unknown location')
    if location:
        frame = frame[frame.location_id == location]
    if engine_name == 'spark':
        from src.spark_artifact import PortableSparkModel
        model = PortableSparkModel(json.loads((path.parent / 'spark_model.json').read_text()))
    else:
        model = joblib.load(path.parent / 'python_model.joblib')
    future = []
    histories = {loc: group.label.tolist() for loc, group in frame.groupby('location_id')}
    last_dates = {loc: group.order_date.max() for loc, group in frame.groupby('location_id')}
    for step in range(1, horizon + 1):
        batch = []
        keys = list(histories)
        for loc in keys:
            values = histories[loc]
            day = last_dates[loc] + pd.Timedelta(days=step)
            batch.append([values[-1], values[-7], sum(values[-7:])/7, day.dayofweek, day.month])
        predictions_batch = model.predict(pd.DataFrame(batch, columns=FEATURES))
        for loc, predicted in zip(keys, predictions_batch):
            prediction = max(0.0, float(predicted))
            histories[loc].append(prediction)
            day = last_dates[loc] + pd.Timedelta(days=step)
            future.append({'location_id':loc, 'date':day.strftime('%Y-%m-%d'), 'predicted_orders':prediction})
    orders = pd.read_parquet(SOURCE, columns=['order_time', 'location_id'])
    if location:
        orders = orders[orders.location_id == location]
    hours = pd.to_numeric(orders.order_time.astype(str).str[:2], errors='coerce')
    dayparts = [{'name':name, 'boundary':f'{lo:02}:00–{hi:02}:00 (start inclusive, end exclusive)',
                 'orders':int(((hours>=lo)&(hours<hi)).sum())}
                for name,lo,hi in [('Overnight',0,6),('Breakfast',6,11),('Lunch',11,16),('Dinner',16,24)]]
    history = frame.groupby('order_date').label.sum().tail(90).reset_index()
    predictions = pd.DataFrame(future).groupby('date').predicted_orders.sum().reset_index()
    return {'version':run['run_id'], 'model':engine['champion'], 'engine':engine_name, 'dataset_sha256':run['dataset_sha256'],
            'locations':locations, 'horizon_days':horizon, 'split':run['split'],
            'evaluation':engine['test'], 'baseline_evaluation':engine['baseline_test'],
            'evaluation_scope':'All locations; later holdout dates, one-day-ahead evaluation. Multi-day recursive forecast error is not separately evaluated.',
            'history':json.loads(history.to_json(orient='records',date_format='iso')),
            'forecast':predictions.to_dict('records'), 'dayparts':dayparts,
            'assumptions':run['assumptions'] + ['Future predictions recursively use previous predicted demand; negative predictions clipped to zero.', 'Demand means order count, not menu units. No prediction intervals are produced.']}
