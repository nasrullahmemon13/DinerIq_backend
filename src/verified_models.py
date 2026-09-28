"""Independent Spark/Python daily location demand models (SRS 12-14, 20-22).

Both engines read operational orders and independently derive lag features.
Validation selects models; later test dates are evaluated once. No predictions
or fitted preprocessing are passed between engines. Missing dates are zero
orders within the observed dataset interval, an explicit modelling assumption.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'processed_data/cleaned/orders/orders.parquet'
FEATURES = ['lag_1', 'lag_7', 'mean_7', 'day_of_week', 'month']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def python_features(source):
    import pandas as pd
    orders = pd.read_parquet(source, columns=['order_id', 'location_id', 'order_date'])
    orders['order_date'] = pd.to_datetime(orders.order_date, errors='coerce').dt.normalize()
    orders = orders.dropna().drop_duplicates('order_id')
    dates = pd.date_range(orders.order_date.min(), orders.order_date.max())
    counts = orders.groupby(['location_id', 'order_date']).size()
    index = pd.MultiIndex.from_product([sorted(orders.location_id.unique()), dates], names=['location_id', 'order_date'])
    frame = counts.reindex(index, fill_value=0).rename('label').reset_index()
    group = frame.groupby('location_id')['label']
    frame['lag_1'] = group.shift(1)
    frame['lag_7'] = group.shift(7)
    frame['mean_7'] = group.transform(lambda s: s.shift(1).rolling(7).mean())
    frame['day_of_week'] = frame.order_date.dt.dayofweek
    frame['month'] = frame.order_date.dt.month
    frame['record_id'] = frame.location_id.astype(str) + ':' + frame.order_date.dt.strftime('%Y-%m-%d')
    return frame.dropna().reset_index(drop=True)


def metrics(actual, prediction):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    nonzero = actual != 0
    return {'mae': float(mean_absolute_error(actual, prediction)),
            'rmse': float(mean_squared_error(actual, prediction) ** .5),
            'r2': float(r2_score(actual, prediction)),
            'mape_pct': float(np.mean(np.abs((actual[nonzero] - prediction[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None,
            'sample_count': len(actual)}


def train_python(source, directory, boundaries):
    import joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    frame = python_features(source)
    train = frame[frame.order_date < boundaries['validation_start']]
    val = frame[(frame.order_date >= boundaries['validation_start']) & (frame.order_date < boundaries['test_start'])]
    test = frame[frame.order_date >= boundaries['test_start']]
    models = {'Linear Regression': LinearRegression(),
              'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
              'Random Forest': RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=2)}
    evaluations = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train[FEATURES], train.label)
        evaluations.append({'model_name': name, 'parameters': model.get_params(),
                            'runtime_seconds': time.perf_counter()-start,
                            'validation': metrics(val.label, model.predict(val[FEATURES]))})
    best = min(evaluations, key=lambda v: v['validation']['rmse'])
    model = models[best['model_name']]
    result = test[['record_id', 'location_id', 'order_date', 'label']].copy()
    result['prediction'] = model.predict(test[FEATURES])
    result.to_parquet(directory / 'python_predictions.parquet', index=False)
    joblib.dump(model, directory / 'python_model.joblib')
    return {'engine': 'scikit-learn', 'status': 'completed', 'models': evaluations,
            'champion': best['model_name'], 'test': metrics(test.label, result.prediction),
            'baseline_test': metrics(test.label, test.lag_7), 'artifact': 'python_model.joblib'}


def train_spark(source, directory, boundaries):
    from spark_jobs.spark_session import get_spark_session
    from pyspark.sql import functions as F, Window
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    spark = get_spark_session('DineIQ-Independent-Demand', master='local[2]', shuffle_partitions=4)
    try:
        orders = spark.read.parquet(str(source)).select('order_id', 'location_id', F.expr('try_cast(order_date as date)').alias('order_date')).dropna().dropDuplicates(['order_id'])
        bounds = orders.agg(F.min('order_date').alias('lo'), F.max('order_date').alias('hi'))
        dates = bounds.select(F.explode(F.sequence('lo', 'hi')).alias('order_date'))
        grid = orders.select('location_id').distinct().crossJoin(dates)
        counts = orders.groupBy('location_id', 'order_date').count().withColumnRenamed('count', 'label')
        frame = grid.join(counts, ['location_id', 'order_date'], 'left').fillna(0, subset=['label'])
        window = Window.partitionBy('location_id').orderBy('order_date')
        frame = (frame.withColumn('lag_1', F.lag('label', 1).over(window))
                 .withColumn('lag_7', F.lag('label', 7).over(window))
                 .withColumn('mean_7', F.avg('label').over(window.rowsBetween(-7, -1)))
                 .withColumn('day_of_week', F.pmod(F.dayofweek('order_date') + 5, F.lit(7)))
                 .withColumn('month', F.month('order_date'))
                 .withColumn('record_id', F.concat_ws(':', 'location_id', F.date_format('order_date', 'yyyy-MM-dd'))).dropna())
        frame = VectorAssembler(inputCols=FEATURES, outputCol='features').transform(frame).cache()
        train = frame.where(F.col('order_date') < boundaries['validation_start'])
        val = frame.where((F.col('order_date') >= boundaries['validation_start']) & (F.col('order_date') < boundaries['test_start']))
        test = frame.where(F.col('order_date') >= boundaries['test_start'])
        models = {'Linear Regression': LinearRegression(maxIter=30, regParam=.01),
                  'Decision Tree': DecisionTreeRegressor(maxDepth=5, seed=42),
                  'Random Forest': RandomForestRegressor(numTrees=30, maxDepth=5, seed=42)}
        fitted, evaluations = {}, []
        for name, estimator in models.items():
            start = time.perf_counter()
            fitted[name] = estimator.fit(train)
            output = fitted[name].transform(val)
            scores = {metric: RegressionEvaluator(metricName=metric).evaluate(output) for metric in ['rmse', 'mae', 'r2']}
            evaluations.append({'model_name': name, 'validation': scores,
                                'parameters': {p.name: v for p, v in estimator.extractParamMap().items()},
                                'runtime_seconds': time.perf_counter()-start})
        best = min(evaluations, key=lambda v: v['validation']['rmse'])
        model = fitted[best['model_name']]
        result = model.transform(test).select('record_id', 'location_id', 'order_date', 'label', 'prediction', *FEATURES).toPandas()
        result.to_parquet(directory / 'spark_predictions.parquet', index=False)
        from src.spark_artifact import export_model, PortableSparkModel
        import numpy as np
        portable = export_model(model, FEATURES)
        np.testing.assert_allclose(PortableSparkModel(portable).predict(result), result.prediction, rtol=1e-9, atol=1e-9)
        artifact, artifact_error = 'spark_model.json', None
        write_json(directory / artifact, portable)
        return {'engine': 'pyspark.ml', 'spark_version': spark.version, 'status': 'completed',
                'models': evaluations, 'champion': best['model_name'], 'test': metrics(result.label, result.prediction),
                'baseline_test': metrics(result.label, result.lag_7), 'artifact': artifact,
                'artifact_error': artifact_error, 'artifact_format': 'dineiq.spark.regression.v1',
                'artifact_parity_records': len(result)}
    finally:
        spark.stop()


def run(source=SOURCE, output=None):
    directory = Path(output) if output else ROOT / 'models/verified_runs' / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    frame = python_features(source)
    dates = sorted(frame.order_date.unique())
    if len(dates) < 60:
        raise ValueError('At least 67 observed calendar days required for lag features and time splits')
    boundaries = {'validation_start': str(dates[int(len(dates)*.7)])[:10],
                  'test_start': str(dates[int(len(dates)*.85)])[:10]}
    manifest = {'run_id': directory.name, 'created_at': datetime.now(timezone.utc).isoformat(),
                'task': 'next-day location order-count prediction', 'dataset': str(source),
                'dataset_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'split': boundaries, 'features': FEATURES, 'selection': 'minimum validation RMSE',
                'assumptions': ['Missing calendar dates within source range count as zero orders.',
                                'Rolling one-day-ahead evaluation uses earlier observed test demand as lags.',
                                'No prediction intervals; linear regression regularization differs between engines.'],
                'status': 'running', 'engines': {}}
    write_json(directory / 'manifest.json', manifest)
    for engine, trainer in [('python', train_python), ('spark', train_spark)]:
        try:
            manifest['engines'][engine] = trainer(source, directory, boundaries)
        except Exception as error:
            manifest['engines'][engine] = {'status': 'failed', 'error': str(error)}
        write_json(directory / 'manifest.json', manifest)
    if all(e['status'] == 'completed' for e in manifest['engines'].values()):
        import pandas as pd
        py = pd.read_parquet(directory / 'python_predictions.parquet')
        sp = pd.read_parquet(directory / 'spark_predictions.parquet')
        if set(py.record_id) != set(sp.record_id):
            raise ValueError('Independent pipelines produced different holdout records')
        compared = sp.merge(py, on='record_id', suffixes=('_spark', '_python'), validate='one_to_one')
        if not (compared.label_spark == compared.label_python).all():
            raise ValueError('Independent aggregation labels disagree')
        compared['difference'] = compared.prediction_spark - compared.prediction_python
        compared['match'] = compared.difference.abs() <= 1
        compared['explanation'] = 'Independent estimators and preprocessing; match tolerance is one order.'
        compared.to_parquet(directory / 'comparison.parquet', index=False)
        manifest['comparison'] = {'records': len(compared), 'agreement_pct': float(compared.match.mean()*100),
                                  'mismatches': int((~compared.match).sum()), 'tolerance_orders': 1}
        manifest['status'] = 'completed'
    else:
        manifest['status'] = 'partial'
    write_json(directory / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
