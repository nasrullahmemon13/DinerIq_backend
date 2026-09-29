"""
DineIQ Analytics - Prescriptive AI Recommendations Engine Router
Implements SRS Step 48 / Functional Requirement (xlvii):
- 57 evidence-backed strategic recommendations
- Categorized by Menu, Promotions, Wastage, Customer Retention, Operations
- Priority, Implementation Effort, Business Impact Rationale, and Formatted Evidence
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from database.connection import get_db
from database.models import SystemConfig, AuditLog
from src.routes import require_roles
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from backend.services.data_store import read_frame, data_exists

router = APIRouter(prefix="/api/v1/analytics/recommendations", tags=["Prescriptive Recommendations"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_recommendations() -> pd.DataFrame:
    p = os.path.join(PROJECT_ROOT, "processed_data", "recommendations", "recommendations.parquet")
    if data_exists(p):
        try:
            return read_frame(p)
        except Exception as e:
            print(f"Error loading recommendations: {e}")
    return pd.DataFrame()

@router.get("")
def get_recommendations(
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    effort: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db=Depends(get_db)
) -> Dict[str, Any]:
    df = load_recommendations()
    if df.empty:
        return {
            "status": "success",
            "summary": {"total_recommendations": 0, "high_priority_count": 0},
            "recommendations": []
        }

    filtered = df.copy()
    if category and category != "All":
        filtered = filtered[filtered["category"] == category]
    if priority and priority != "All":
        filtered = filtered[filtered["priority"] == priority]
    if effort and effort != "All":
        filtered = filtered[filtered["implementation_effort"] == effort]
    if search:
        m1 = filtered["recommended_action"].str.contains(search, case=False, na=False, regex=False)
        m2 = filtered["target_entity_name"].str.contains(search, case=False, na=False, regex=False)
        m3 = filtered["business_impact_rationale"].str.contains(search, case=False, na=False, regex=False)
        filtered = filtered[m1 | m2 | m3]

    recs = []
    for r in filtered.to_dict(orient="records"):
        cleaned_r = {}
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                cleaned_r[k] = list(v)
            elif hasattr(v, "tolist"):
                cleaned_r[k] = v.tolist()
            elif pd.isna(v):
                cleaned_r[k] = None
            else:
                cleaned_r[k] = v
        recs.append(cleaned_r)

    summary = {
        "total_recommendations": len(df),
        "high_priority_count": int((df["priority"].str.contains("High|Critical", case=False, na=False)).sum()),
        "categories": [str(c) for c in df["category"].dropna().unique().tolist()],
        "priorities": [str(p) for p in df["priority"].dropna().unique().tolist()],
        "efforts": [str(e) for e in df["implementation_effort"].dropna().unique().tolist()]
    }

    reviews = {c.config_key.removeprefix('recommendation.review.'): json.loads(c.config_value)
               for c in db.query(SystemConfig).filter(SystemConfig.category == 'recommendation_review').all()}
    for rec in recs:
        rec['review'] = reviews.get(rec['recommendation_id'], {'status':'pending'})
    return {
        "status": "success",
        "summary": summary,
        "recommendations": recs
    }


class ReviewRequest(BaseModel):
    status: Literal['pending', 'accepted', 'dismissed', 'implemented']
    note: str = Field('', max_length=2000)


@router.post('/{recommendation_id}/review')
def review_recommendation(recommendation_id: str, request: ReviewRequest,
                          actor=Depends(require_roles(['admin', 'analyst', 'regional_manager'])),
                          db=Depends(get_db)):
    records = load_recommendations()
    if records.empty or recommendation_id not in set(records.recommendation_id):
        raise HTTPException(404, 'Recommendation not found')
    key = 'recommendation.review.' + recommendation_id
    config = db.query(SystemConfig).filter(SystemConfig.config_key == key).first()
    previous = json.loads(config.config_value) if config else None
    value = {'status':request.status, 'note':request.note, 'actor':actor['username'],
             'updated_at':datetime.now(timezone.utc).isoformat()}
    if config is None:
        config = SystemConfig(config_key=key, config_value=json.dumps(value), category='recommendation_review')
        db.add(config)
    config.config_value = json.dumps(value)
    config.updated_by = actor['username']
    config.updated_at = datetime.now(timezone.utc)
    db.add(AuditLog(audit_id='AUD-'+uuid.uuid4().hex, event_type='ADMIN_ACTION',
                    action='REVIEW_RECOMMENDATION', actor=actor['username'], resource_id=recommendation_id,
                    status='SUCCESS', timestamp=datetime.now(timezone.utc),
                    details=json.dumps({'previous':previous, 'review':value})))
    db.commit()
    return value
