import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.services import location_scope
from backend.routers.locations_channels import router
from backend.routers.data_governance import router as governance_router


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)
    app.include_router(governance_router)
    return TestClient(app)


def test_scoped_locations_match_completed_source_orders(api):
    raw = location_scope.read('orders')
    raw = raw[raw.order_status.eq('COMPLETED')].copy()
    date = location_scope.dates(raw, 'order_date', 'order_timestamp')
    expected = raw[date.between(date.max()-pd.Timedelta(days=29),date.max()) & raw.location_id.eq('LOC-001')]
    response = api.get('/api/v1/analytics/locations-channels/performance', params={'date_range':'LAST_30_DAYS','location_ids':'LOC-001'})
    assert response.status_code == 200
    data = response.json()
    assert len(data['locations']) == 1
    assert data['kpis']['total_orders'] == len(expected)
    assert data['kpis']['total_revenue'] == pytest.approx((expected.subtotal_amount-expected.discount_amount).sum(), abs=.01)
    assert data['kpis']['average_order_value'] == pytest.approx(data['kpis']['total_revenue']/len(expected),abs=.01)


def test_channels_reconcile_to_same_location_and_quarter(api):
    base='/api/v1/analytics/locations-channels/'
    perf=api.get(base+'performance',params={'date_range':'Q1','location_ids':'LOC-002'}).json()
    response=api.get(base+'channels',params={'date_range':'Q1','location_id':'LOC-002'})
    assert response.status_code==200
    data=response.json()
    assert sum(r['orders'] for r in data['channel_performance']) == perf['kpis']['total_orders']
    assert sum(r['revenue'] for r in data['channel_performance']) == pytest.approx(perf['kpis']['total_revenue'],abs=.01)
    assert sum(sum(r['distribution'].values()) for r in data['hourly_patterns']) == perf['kpis']['total_orders']
    assert all(r['location_id']=='LOC-002' for r in data['location_channel_matrix'])


def test_empty_scope_and_invalid_dates(api):
    base='/api/v1/analytics/locations-channels/'
    data=api.get(base+'channels?location_id=DOES-NOT-EXIST').json()
    assert data['channel_performance']==[]
    assert data['kpis']['total_channel_orders']==0
    assert api.get(base+'performance?date_range=made-up').status_code==422


def test_divergence_has_real_location_names(api):
    data=api.get('/api/v1/analytics/locations-channels/menu-intelligence').json()
    assert data['divergent_dishes']
    for row in data['divergent_dishes']:
        assert row['category_name'] and row['top_location_name'] and row['worst_location_name']
        assert row['max_quantity'] >= row['min_quantity']


@pytest.mark.parametrize('path',['ingestion-overview','quality-profile','quarantine-manifest','integration-joins','engineered-features','spark-cluster'])
def test_governance_feeds_serialize(api,path):
    response=api.get('/api/v1/data-governance/'+path)
    assert response.status_code==200
    assert isinstance(response.json(),dict)
