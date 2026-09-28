"""Read-only registry of measured independent training runs."""
import json
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from src.routes import require_roles
from fastapi.responses import FileResponse
from database.connection import get_db
from database.models import SystemConfig, AuditLog
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Literal
import uuid

ROOT = Path(__file__).resolve().parents[2] / 'models/verified_runs'
router = APIRouter(prefix='/api/v1/verified-models', dependencies=[Depends(require_roles(['admin', 'analyst']))])

def load_run(run_id):
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise HTTPException(404, 'Unknown run')
    path = ROOT / run_id / 'manifest.json'
    if not path.exists():
        raise HTTPException(404, 'Unknown run')
    return path.parent, json.loads(path.read_text(encoding='utf-8'))

class Activation(BaseModel):
    engine: Literal['spark', 'python']


@router.post('/{run_id}/activate')
def activate(run_id: str, request: Activation, actor=Depends(require_roles(['admin'])), db=Depends(get_db)):
    directory, run = load_run(run_id)
    engine = run.get('engines', {}).get(request.engine, {})
    artifact = engine.get('artifact')
    if engine.get('status') != 'completed' or not artifact or not (directory / artifact).is_file():
        raise HTTPException(409, 'A completed run and available inference artifact are required')
    from src.verified_models import SOURCE
    import hashlib
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != run['dataset_sha256']:
        raise HTTPException(409, 'Dataset differs from training data; retrain before activation')
    config = db.query(SystemConfig).filter(SystemConfig.config_key == 'forecast.active_model').first()
    previous = config.config_value if config else None
    value = json.dumps({'run_id':run_id, 'engine':request.engine})
    if config is None:
        config = SystemConfig(config_key='forecast.active_model', config_value=value, category='models')
        db.add(config)
    config.config_value = value
    config.updated_by = actor['username']
    config.updated_at = datetime.now(timezone.utc)
    db.add(AuditLog(audit_id='AUD-'+uuid.uuid4().hex, event_type='ADMIN_ACTION', action='ACTIVATE_FORECAST_MODEL',
                    actor=actor['username'], resource_id=run_id, status='SUCCESS', timestamp=datetime.now(timezone.utc),
                    details=json.dumps({'previous':previous, 'selected':json.loads(value)})))
    db.commit()
    return {'active':json.loads(value), 'previous':json.loads(previous) if previous else None}


@router.get('/{run_id}/artifact/{engine}')
def download_artifact(run_id: str, engine: Literal['spark', 'python']):
    directory, run = load_run(run_id)
    artifact = run.get('engines', {}).get(engine, {}).get('artifact')
    allowed = {'spark':'spark_model.json', 'python':'python_model.joblib'}
    if artifact != allowed[engine] or not (directory / artifact).is_file():
        raise HTTPException(404, 'Model artifact unavailable')
    return FileResponse(directory / artifact, filename=f'{run_id}_{artifact}')


@router.get('')
def list_runs():
    runs = []
    for path in ROOT.glob('*/manifest.json'):
        try:
            run = json.loads(path.read_text(encoding='utf-8'))
            for engine in run.get('engines', {}).values():
                artifact = engine.get('artifact')
                engine['artifact_available'] = bool(artifact and (path.parent / artifact).exists())
            runs.append(run)
        except (OSError, ValueError):
            runs.append({'run_id': path.parent.name, 'status': 'unreadable', 'engines': {}})
    return {'runs': sorted(runs, key=lambda r: r.get('created_at', ''), reverse=True)}


@router.get('/{run_id}/comparison')
def comparison(run_id: str):
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise HTTPException(404, 'Unknown run')
    path = ROOT / run_id / 'comparison.parquet'
    if not path.exists():
        raise HTTPException(404, 'Both independent pipelines must complete before comparison is available')
    import pandas as pd
    return json.loads(pd.read_parquet(path).head(100).to_json(orient='records', date_format='iso'))
