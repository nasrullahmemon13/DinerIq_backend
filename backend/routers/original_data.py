"""Admin-only durable import of existing project Parquet outputs."""
import hashlib
import json
import re
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from database.connection import get_db
from database.models import DatasetPart, SystemConfig
from src.routes import require_roles

router = APIRouter(prefix='/api/v1/admin/original-data', dependencies=[Depends(require_roles(['admin']))])


def validate_key(key, version):
    if len(key) > 90 or not re.fullmatch(r'[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*\.parquet', key):
        raise HTTPException(400, 'Invalid dataset key')
    if not re.fullmatch(r'[a-f0-9]{64}', version):
        raise HTTPException(400, 'Invalid SHA-256')


@router.put('/part')
async def upload_part(request: Request, key: str, version: str,
                      part: int = Query(ge=0, le=255), db=Depends(get_db)):
    validate_key(key, version)
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > 2 * 1024 * 1024:
            raise HTTPException(413, 'Each part must be at most 2 MiB')
    stmt = insert(DatasetPart).values(dataset_key=key, version=version, part_number=part, payload=bytes(payload))
    db.execute(stmt.on_conflict_do_update(index_elements=['dataset_key','version','part_number'], set_={'payload':stmt.excluded.payload}))
    db.commit()
    return {'part':part, 'bytes':len(payload)}


class CommitDataset(BaseModel):
    key: str
    sha256: str
    parts: int = Field(ge=1, le=256)


@router.post('/commit')
def commit_dataset(body: CommitDataset, db=Depends(get_db)):
    import pyarrow.parquet as pq
    validate_key(body.key, body.sha256)
    parts = db.execute(select(DatasetPart).where(DatasetPart.dataset_key == body.key,
        DatasetPart.version == body.sha256).order_by(DatasetPart.part_number)).scalars().all()
    if [p.part_number for p in parts] != list(range(body.parts)):
        raise HTTPException(400, 'Dataset parts incomplete')
    payload = b''.join(p.payload for p in parts)
    if hashlib.sha256(payload).hexdigest() != body.sha256:
        raise HTTPException(400, 'Checksum mismatch')
    try:
        rows = pq.ParquetFile(BytesIO(payload)).metadata.num_rows
    except Exception:
        raise HTTPException(400, 'Invalid Parquet file')
    info = {'sha256':body.sha256,'parts':body.parts,'bytes':len(payload),'rows':rows}
    stmt = insert(SystemConfig).values(config_key='asset/'+body.key,config_value=json.dumps(info),category='original_data',is_secret=False)
    db.execute(stmt.on_conflict_do_update(index_elements=['config_key'],set_={'config_value':stmt.excluded.config_value}))
    db.commit()
    return {'key':body.key, **info}
