"""Read original, checksum-verified Parquet artifacts from durable database storage."""
from pathlib import Path
from io import BytesIO
from functools import lru_cache
import hashlib
import json
import os
import pandas as pd
from sqlalchemy import select, inspect
from database.models import DatasetPart, SystemConfig

ROOT = Path(__file__).resolve().parents[2]


def dataset_key(path):
    try:
        relative = Path(path).resolve().relative_to(ROOT / 'processed_data')
    except ValueError:
        return None
    return relative.as_posix() if relative.suffix == '.parquet' else None


def manifest(key):
    from database.connection import engine
    with engine.connect() as connection:
        if not inspect(connection).has_table('system_configs'):
            return None
        value = connection.execute(select(SystemConfig.config_value).where(
            SystemConfig.config_key == 'asset/' + key)).scalar()
    return json.loads(value) if value else None


@lru_cache(maxsize=8)
def artifact_bytes(key, version):
    from database.connection import engine
    with engine.connect() as connection:
        parts = connection.execute(select(DatasetPart.payload).where(
            DatasetPart.dataset_key == key, DatasetPart.version == version
        ).order_by(DatasetPart.part_number)).scalars().all()
    payload = b''.join(parts)
    if hashlib.sha256(payload).hexdigest() != version:
        raise ValueError('Stored dataset checksum mismatch: ' + key)
    return payload


def data_exists(path):
    if os.path.exists(path):
        return True
    key = dataset_key(path)
    return bool(key and manifest(key))


def read_frame(path, columns=None, **kwargs):
    key = dataset_key(path)
    if key:
        info = manifest(key)
        if info:
            return pd.read_parquet(BytesIO(artifact_bytes(key, info['sha256'])), columns=columns, **kwargs)
    if os.path.exists(path):
        return pd.read_parquet(path, columns=columns, **kwargs)
    return pd.DataFrame(columns=columns or [])
