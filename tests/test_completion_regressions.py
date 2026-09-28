"""Real dataset integration checks; all authentication/database state is isolated."""
import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_security_regressions import client, login
from database.connection import get_db
from src.data_export_engine import DataExportEngine, EXPORTABLE_DATASETS


def test_dashboard_and_module_payloads(client):
    from backend.main import app
    token = login(client, 'admin')
    app.dependency_overrides[get_db] = client.app.dependency_overrides[get_db]
    paths = ['dashboard/executive', 'menu-intelligence', 'customer-intelligence', 'wastage',
             'orders/paginated', 'inventory/stock', 'ratings/feed', 'restaurants',
             'promotions/analytics', 'analytics/basket', 'analytics/peak-periods',
             'analytics/pricing', 'analytics/anomalies-churn', 'analytics/recommendations',
             'analytics/what-if/items', 'analytics/locations-channels/performance',
             'analytics/locations-channels/channels', 'reports', 'export/datasets',
             'data-governance/ingestion-overview', 'data-governance/engineered-features',
             'data-governance/spark-cluster', 'verified-models']
    try:
        with TestClient(app) as full:
            for path in paths:
                response = full.get('/api/v1/'+path, headers={'Authorization': f'Bearer {token}'})
                assert response.status_code == 200, (path, response.text[:500])
                payload = response.json()
                assert 'error' not in payload, (path, payload)
                if path == 'dashboard/executive':
                    assert payload['total_orders'] > 0
                    assert payload['monthly_trends'] and payload['top_menu_items']
                if path == 'data-governance/engineered-features':
                    assert payload['available_features'] == 22
                if path == 'ratings/feed':
                    assert payload['summary']['average_rating'] is None
                    assert payload['summary']['csat_pct'] is None
                    assert sum(r['count'] for r in payload['summary']['star_distribution']) == 0
            assert full.get('/api/v1/dashboard/executive').status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_exports_filter_exact_scope_and_missing_source(monkeypatch, tmp_path):
    data = pd.DataFrame({'location_id':['A','B','A'], 'amount':[1,2,3]})
    content, _, _ = DataExportEngine.export_dataset('fixture', filter_df=data, filters={'location_id':'A'})
    result = pd.read_csv(io.BytesIO(content))
    assert result.amount.sum() == 4 and len(result) == 2
    with pytest.raises(ValueError, match='does not support'):
        DataExportEngine.export_dataset('fixture', filter_df=data, filters={'item_id':'x'})
    monkeypatch.setitem(EXPORTABLE_DATASETS, 'missing', str(tmp_path/'missing.parquet'))
    with pytest.raises(FileNotFoundError):
        DataExportEngine.export_dataset('missing')


def test_pipeline_paths_cannot_escape_staging(tmp_path):
    from backend.services.pipeline_runner import PipelineRunner
    runner = PipelineRunner(str(tmp_path))
    for run in ['..', '../outside', 'C:\\outside', 'RUN-20250101-ABCDEF/../../outside']:
        with pytest.raises(FileNotFoundError):
            runner._get_run_paths(run)
    assert runner._get_run_paths('RUN-20250101-ABCDEF')['base'].startswith(str(tmp_path))


def test_raw_upload_is_immutable_and_preview_is_read_only(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from backend.services.pipeline_runner import PipelineRunner
    from backend.routers import data_pipeline
    runner = PipelineRunner(str(tmp_path))
    monkeypatch.setattr(data_pipeline, 'runner', runner)
    run = runner.initialize_run(uploaded_by='test-admin')
    original = b'order_id,total_amount,location_id\nO1,42,L1\n'
    first = runner.stage_file(run, 'orders.csv', original)
    second = runner.stage_file(run, 'orders.csv', b'order_id,total_amount,location_id\nO2,90,L1\n')
    assert first['file_name'] != second['file_name']
    assert Path(first['file_path']).read_bytes() == original

    assert first['sha256'] == hashlib.sha256(original).hexdigest()
    preview = data_pipeline.preview_dataset(run, 'orders', 'raw', 0, 25)
    assert preview['rows'][0]['order_id'] == 'O1'
    assert preview['total_records'] == 1
    assert Path(first['file_path']).read_bytes() == original


def test_chart_distributions_use_all_filtered_rows_before_pagination(client):
    from datetime import date
    from database.models import Rating, Order
    from backend.routers.ratings import router as ratings
    from backend.routers.orders import router as orders
    from fastapi import FastAPI
    chart_app = FastAPI()
    chart_app.include_router(ratings)
    chart_app.include_router(orders)
    chart_app.dependency_overrides[get_db] = client.app.dependency_overrides[get_db]
    chart_client = TestClient(chart_app)
    session = client.app.dependency_overrides[get_db]()
    db = next(session)
    try:
        for i in range(7):
            db.add(Rating(rating_id=f'R{i}', overall_rating=5 if i < 6 else 1,
                          location_id='A' if i < 5 else 'B', review_date=date(2025,1,1)))
            db.add(Order(order_id=f'O{i}', location_id='A' if i < 5 else 'B',
                         order_date=date(2025,1,1), order_type='Delivery' if i < 6 else 'Takeout',
                         order_status='COMPLETED', total_amount=10))
        db.commit()
    finally:
        session.close()
    headers = {'Authorization':f'Bearer {login(client,"admin")}'}
    summary = chart_client.get('/api/v1/ratings/feed?page_size=5',headers=headers).json()['summary']
    assert summary['rating_count'] == 7
    assert {r['stars']:r['count'] for r in summary['star_distribution']} == {1:1,2:0,3:0,4:0,5:6}
    assert summary['csat_pct'] == 85.7
    scoped = chart_client.get('/api/v1/ratings/feed?location_id=A',headers=headers).json()['summary']
    assert scoped['rating_count'] == 5 and scoped['csat_pct'] == 100
    summary = chart_client.get('/api/v1/orders/paginated?page_size=5',headers=headers).json()['summary']
    assert {r['channel']:r['count'] for r in summary['channel_distribution']} == {'Delivery':6,'Takeout':1}


def test_peak_heatmap_preserves_valid_rows_and_discloses_invalid_dates(monkeypatch, tmp_path):
    from backend.routers import peak_periods as module
    frames = {
        'HOURLY_PATH': pd.DataFrame({'hour':[12,18], 'order_count':[2,1], 'total_revenue':[20,10],
                                    'revenue_share_pct':[66.7,33.3], 'avg_order_value':[10,10]}),
        'DAILY_PATH': pd.DataFrame({'day_name':['Monday','Tuesday'], 'total_orders':[2,1],
                                   'total_revenue':[20,10], 'revenue_share_pct':[90,10]}),
        'ORDERS_PATH': pd.DataFrame({'order_id':['1','2','3'],
                                    'order_timestamp':['2025-01-06 12:00:00','invalid','2025-01-07 18:00:00']}),
    }
    for key, frame in frames.items():
        path = tmp_path/(key+'.parquet')
        frame.to_parquet(path)
        monkeypatch.setattr(module,key,str(path))
    monkeypatch.setattr(module,'SEASONAL_PATH',str(tmp_path/'none'))
    monkeypatch.setattr(module,'CHANNELS_PATH',str(tmp_path/'none'))
    result = module.get_peak_periods_analytics()
    assert result['heatmap_error'] is None
    assert result['excluded_timestamp_rows'] == 1
    assert sum(sum(r['hourly_counts']) for r in result['heatmap']) == 2
    assert result['summary']['peak_day_share_pct'] == 66.7
    assert result['hourly_distribution'][0]['order_count'] == 2


def test_filtered_analytical_summaries_and_literal_search():
    from backend.routers.market_basket import get_market_basket_analysis
    from backend.routers.pricing_intelligence import get_pricing_intelligence
    basket = get_market_basket_analysis(min_lift=None,min_confidence=None,category=None,search='[')
    assert basket['summary']['total_rules'] == 0
    assert basket['summary']['average_lift'] is None
    assert basket['association_rules'] == []
    assert basket['available_categories']
    pricing = get_pricing_intelligence(tier=None,category=None,search='no-such-dish-928415')
    assert pricing['summary']['total_analyzed_dishes'] == len(pricing['items']) == 0
    assert pricing['summary']['avg_system_margin'] is None

