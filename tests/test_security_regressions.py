import io
import time
import pytest
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from database.models import Base, Role, User
from database.connection import get_db, hash_password
from src.routes import router, ACTIVE_TOKENS
from backend.routers.admin_management import router as admin_router
from backend.routers.exports import router as exports_router
from backend.routers.verified_models import router as models_router
from backend.routers.forecasting import router as forecasting_router
from backend.routers.recommendations import router as recommendations_router
from src.data_export_engine import DataExportEngine


@pytest.fixture
def client():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        for role in ['admin', 'analyst']:
            db.add(Role(role_id=role, role_name=role))
            db.add(User(user_id=role, username=role, email=f'{role}@test.local',
                        full_name=role, hashed_password=hash_password('Correct-Pass-123'),
                        role_id=role, is_active=True))
        db.commit()
    app = FastAPI()
    for item in [router, admin_router, exports_router, models_router, forecasting_router, recommendations_router]:
        app.include_router(item)
    def isolated_db():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = isolated_db
    with TestClient(app) as test_client:
        yield test_client
    ACTIVE_TOKENS.clear()
    engine.dispose()


def login(client, name):
    response = client.post('/api/v1/auth/login', json={'username': name, 'password': 'Correct-Pass-123'})
    assert response.status_code == 200
    return response.json()['access_token']


def test_forged_roles_mock_tokens_and_password_aliases_rejected(client):
    for headers in [{}, {'X-User-Role': 'admin'}, {'Authorization': 'Bearer mock_test_admin'}]:
        assert client.get('/api/v1/admin-management/users', headers=headers).status_code == 401
        assert client.get('/api/v1/export/menu_performance', headers=headers).status_code == 401
    assert client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'password123'}).status_code == 401


def test_admin_roles_expiry_logout_and_registration(client):
    analyst = login(client, 'analyst')
    assert client.get('/api/v1/admin-management/users', headers={'Authorization': f'Bearer {analyst}'}).status_code == 403
    admin = login(client, 'admin')
    headers = {'Authorization': f'Bearer {admin}'}
    assert client.get('/api/v1/admin-management/users', headers=headers).status_code == 200
    assert client.post('/api/v1/auth/register', json={'username':'attacker','email':'x@test.local', 'full_name':'x','password':'secret','role_id':'admin'}).status_code == 401
    ACTIVE_TOKENS[admin]['expires_at'] = time.time() - 1
    assert client.get('/api/v1/auth/me', headers=headers).status_code == 401
    headers = {'Authorization': f'Bearer {analyst}'}
    assert client.post('/api/v1/auth/logout', headers=headers).status_code == 200
    assert client.get('/api/v1/auth/me', headers=headers).status_code == 401


def test_csv_and_excel_neutralize_formulas_and_preserve_numbers():
    frame = pd.DataFrame({'name':['=1+1', '  @SUM(A1)', 'normal'], 'amount':[-2, 0, 3]})
    for fmt in ['csv', 'xlsx']:
        data, _, _ = DataExportEngine.export_dataset('fixture', fmt, 'analyst', frame)
        result = pd.read_csv(io.BytesIO(data)) if fmt == 'csv' else pd.read_excel(io.BytesIO(data))
        assert result['name'].iloc[0] == "'=1+1"
        assert result['name'].iloc[1].startswith("'")
        assert result['amount'].tolist() == [-2, 0, 3]


def test_activation_rollback_changes_real_inference_and_artifacts_download(client):
    admin = {'Authorization': f'Bearer {login(client, "admin")}'}
    analyst = {'Authorization': f'Bearer {login(client, "analyst")}'}
    runs = client.get('/api/v1/verified-models', headers=admin).json()['runs']
    run = next(r for r in runs if r.get('engines', {}).get('spark', {}).get('artifact') == 'spark_model.json')
    path = f'/api/v1/verified-models/{run["run_id"]}'
    assert client.post(path+'/activate', headers=analyst, json={'engine':'spark'}).status_code == 403
    outputs = {}
    for engine in ['spark', 'python', 'spark']:
        assert client.post(path+'/activate', headers=admin, json={'engine':engine}).status_code == 200
        response = client.get('/api/v1/analytics/forecasting/live?horizon=2', headers=admin)
        assert response.status_code == 200
        result = response.json()
        assert result['engine'] == engine
        assert result['version'] == run['run_id']
        outputs[engine] = result['forecast']
        artifact = client.get(path+f'/artifact/{engine}', headers=admin)
        assert artifact.status_code == 200 and len(artifact.content) > 50
    assert outputs['spark'] != outputs['python']
    logs = client.get('/api/v1/admin-management/audit-logs', headers=admin).json()['logs']
    assert sum(log['action']=='ACTIVATE_FORECAST_MODEL' for log in logs) == 3


def test_recommendation_review_persists_with_actor_and_audit(client):
    headers = {'Authorization': f'Bearer {login(client, "analyst")}'}
    root = '/api/v1/analytics/recommendations'
    records = client.get(root, headers=headers).json()['recommendations']
    identifier = records[0]['recommendation_id']
    assert client.post(root+'/'+identifier+'/review', json={'status':'accepted'}).status_code == 401
    result = client.post(root+'/'+identifier+'/review', headers=headers, json={'status':'accepted', 'note':'Reviewed supporting figures'})
    assert result.status_code == 200
    refreshed = client.get(root, headers=headers).json()['recommendations']
    selected = next(r for r in refreshed if r['recommendation_id'] == identifier)
    assert selected['review']['status'] == 'accepted'
    assert selected['review']['actor'] == 'analyst'
    assert client.get(root+'?search=%5B', headers=headers).status_code == 200
    admin = {'Authorization': f'Bearer {login(client, "admin")}'}
    logs = client.get('/api/v1/admin-management/audit-logs', headers=admin).json()['logs']
    assert any(log['action']=='REVIEW_RECOMMENDATION' for log in logs)
