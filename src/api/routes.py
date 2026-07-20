# src/api/routes.py
"""
Complete API Routes with All Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import hashlib
import json

from src.api.schemas import (
    PatientData,
    RiskPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    ModelMetricsResponse,
    FeatureImportanceResponse,
    DriftReportResponse,
    ModelPromotionRequest,
    AuditLogResponse,
    UserLoginRequest,
    UserLoginResponse,
    PaginatedResponse,
)
from src.api.dependencies import (
    get_model,
    get_redis_client,
    get_drift_detector,
    get_circuit_breaker,
    get_feature_store,
    get_current_user,
    get_audit_logger,
    require_permission,
)
from src.pipelines.inference_pipeline import InferencePipeline
from src.async_tasks.tasks import generate_explanation_report
from src.caching.redis_client import FastRedisClient
from src.monitoring.drift_detection import DriftDetector
from src.feature_store.online_store import FeatureStore
from src.logger import get_logger
from src.utils.circuit_breaker import CircuitBreaker
from src.security.auth import AuthManager, get_current_user
from src.security.audit import AuditLogger

logger = get_logger(__name__)

router = APIRouter()

# ============================================================================
# 🎯 Prediction Endpoints
# ============================================================================

@router.post(
    "/predict",
    response_model=RiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"},
    },
)
async def predict_risk(
    request: Request,
    patient_data: PatientData,
    background_tasks: BackgroundTasks,
    model: Any = Depends(get_model),
    redis_client: FastRedisClient = Depends(get_redis_client),
    drift_detector: DriftDetector = Depends(get_drift_detector),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
    feature_store: FeatureStore = Depends(get_feature_store),
    current_user: Dict = Depends(get_current_user),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """
    Predict disease risk from patient clinical data
    
    **Performance:**
    - <100ms response time (cached)
    - <500ms response time (uncached)
    - 1000+ requests/second capacity
    
    **Features:**
    - Real-time validation
    - Data drift detection
    - Automatic caching
    - Async explanation generation
    - Audit logging
    """
    
    correlation_id = request.state.correlation_id
    start_time = time.time()
    
    try:
        # Step 1: Input validation (already done by Pydantic)
        input_df = pd.DataFrame([patient_data.dict()])
        
        # Step 2: Check data drift
        drift_report = drift_detector.detect_drift(input_df)
        drift_detected = drift_report.get("drift_detected", False)
        
        if drift_detected:
            logger.warning(
                f"Data drift detected for request {correlation_id}",
                extra={"drift_share": drift_report.get("drift_share")}
            )
        
        # Step 3: Feature engineering
        features = await feature_store.get_features(input_df)
        
        # Step 4: Check cache (for identical requests)
        cache_key = _generate_cache_key(patient_data)
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            logger.info(f"✅ Cache hit for {correlation_id}")
            response = RiskPredictionResponse(**cached_result)
            response.drift_detected = drift_detected
            return response
        
        # Step 5: Run inference with circuit breaker
        with circuit_breaker:
            inference_pipeline = InferencePipeline(model)
            result = await inference_pipeline.predict(
                features,
                correlation_id,
                explain=False,  # Generate explanation async
            )
        
        # Step 6: Prepare response
        response = RiskPredictionResponse(
            patient_id=result["patient_id"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            probability=result["probability"],
            top_factors=result["top_factors"][:5],
            contributing_factors=result["contributing_factors"][:10],
            clinical_explanation=result["clinical_explanation"],
            confidence=result.get("confidence", 0.85),
            drift_detected=drift_detected,
            model_version=result.get("model_version", "v3.0.0"),
            correlation_id=correlation_id,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Step 7: Cache response
        await redis_client.set(
            cache_key,
            response.dict(),
            ttl=300,  # 5 minutes cache
        )
        
        # Step 8: Async tasks
        if response.risk_level in ["High Risk", "Moderate Risk"]:
            background_tasks.add_task(
                generate_explanation_report,
                patient_data.dict(),
                response.dict(),
                correlation_id,
            )
        
        # Step 9: Audit log
        await audit_logger.log_prediction(
            user_id=current_user.get("id"),
            patient_id=response.patient_id,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            correlation_id=correlation_id,
        )
        
        # Step 10: Record metrics
        request.app.state.prometheus.record_prediction(
            response.risk_score,
            response.risk_level,
            response.processing_time_ms,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model: Any = Depends(get_model),
    current_user: Dict = Depends(get_current_user),
):
    """
    Batch prediction for multiple patients
    
    **Performance:**
    - Up to 1000 patients per batch
    - Parallel processing
    - <5s for 1000 patients
    """
    
    correlation_id = request.state.correlation_id
    results = []
    
    # Process in parallel
    async def process_patient(patient, idx):
        try:
            input_df = pd.DataFrame([patient.dict()])
            inference_pipeline = InferencePipeline(model)
            result = await inference_pipeline.predict(input_df, f"{correlation_id}_{idx}")
            return {"index": idx, "success": True, "result": result}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}
    
    # Create tasks
    tasks = [
        process_patient(patient, idx)
        for idx, patient in enumerate(request.patients)
    ]
    
    # Execute in parallel with concurrency limit
    semaphore = asyncio.Semaphore(50)  # Max 50 concurrent
    
    async def limited_process(task):
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[
        limited_process(task) for task in tasks
    ])
    
    # Format response
    predictions = []
    for result in results:
        if result["success"]:
            predictions.append(result["result"])
        else:
            predictions.append({"error": result["error"]})
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.patients),
        successful=sum(1 for r in results if r["success"]),
        failed=sum(1 for r in results if not r["success"]),
        processing_time_ms=time.time() - start_time,
    )


@router.post(
    "/predict/stream",
    tags=["Prediction"],
)
async def stream_predict(
    request: Request,
    patients: List[PatientData],
    model: Any = Depends(get_model),
):
    """
    Stream predictions for real-time applications
    
    Uses Server-Sent Events (SSE) for streaming
    """
    
    async def event_generator():
        for idx, patient in enumerate(patients):
            try:
                input_df = pd.DataFrame([patient.dict()])
                inference_pipeline = InferencePipeline(model)
                result = await inference_pipeline.predict(input_df)
                
                yield f"data: {json.dumps({'index': idx, 'result': result})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'index': idx, 'error': str(e)})}\n\n"
            
            await asyncio.sleep(0.01)  # Prevent overwhelming client
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============================================================================
# 🧠 Explainability Endpoints
# ============================================================================

@router.get(
    "/explanations/{patient_id}",
    response_model=ExplanationResponse,
    tags=["Explainability"],
)
async def get_explanations(
    patient_id: str,
    model: Any = Depends(get_model),
    current_user: Dict = Depends(get_current_user),
):
    """
    Get detailed SHAP explanations for a patient
    
    **Features:**
    - Global and local explanations
    - Feature contribution breakdown
    - Natural language explanation
    - Visualization URLs
    """
    
    try:
        # Load from cache or database
        from src.components.explainability import ClinicalSHAPExplainer
        
        explainer = ClinicalSHAPExplainer(model)
        explanation = await explainer.get_explanation(patient_id)
        
        return ExplanationResponse(
            patient_id=patient_id,
            risk_score=explanation["risk_score"],
            risk_level=explanation["risk_level"],
            shap_values=explanation["shap_values"],
            base_value=explanation["base_value"],
            feature_names=explanation["feature_names"],
            feature_values=explanation["feature_values"],
            contribution_breakdown=explanation["contribution_breakdown"],
            natural_language_explanation=explanation["natural_language_explanation"],
            waterfall_plot_url=f"/plots/waterfall/{patient_id}",
            force_plot_url=f"/plots/force/{patient_id}",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Explanations not found for patient {patient_id}",
        )


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    tags=["Explainability"],
)
async def get_feature_importance(
    model: Any = Depends(get_model),
    top_n: int = 20,
):
    """
    Get global feature importance ranking
    """
    
    from src.components.explainability import ClinicalSHAPExplainer
    
    explainer = ClinicalSHAPExplainer(model)
    importance = await explainer.get_feature_importance(top_n)
    
    return FeatureImportanceResponse(
        features=importance["features"],
        scores=importance["scores"],
        description=importance["description"],
        timestamp=datetime.now(),
    )


# ============================================================================
# 📊 Monitoring Endpoints
# ============================================================================

@router.get(
    "/model-metrics",
    response_model=Dict[str, ModelMetricsResponse],
    tags=["Monitoring"],
)
async def get_model_metrics(
    model_registry: Any = Depends(get_model_registry),
):
    """
    Get model performance metrics
    
    **Metrics:**
    - Accuracy, Precision, Recall
    - Specificity, F1 Score
    - ROC-AUC, PR-AUC
    - Confusion Matrix
    """
    
    metrics = await model_registry.get_latest_metrics()
    return {
        model_name: ModelMetricsResponse(
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            specificity=m["specificity"],
            f1_score=m["f1_score"],
            roc_auc=m["roc_auc"],
            confusion_matrix=m["confusion_matrix"],
            model_name=model_name,
            version=m["version"],
            last_updated=m["timestamp"],
        )
        for model_name, m in metrics.items()
    }


@router.get(
    "/drift-report",
    response_model=DriftReportResponse,
    tags=["Monitoring"],
)
async def get_drift_report(
    drift_detector: DriftDetector = Depends(get_drift_detector),
):
    """
    Get data drift detection report
    
    **Features:**
    - Per-feature drift scores
    - Statistical significance
    - Temporal trends
    """
    
    report = drift_detector.get_report()
    return DriftReportResponse(
        drift_detected=report["drift_detected"],
        drift_share=report["drift_share"],
        column_drifts=report["column_drifts"],
        recommendation=report.get("recommendation"),
        timestamp=datetime.now(),
    )


# ============================================================================
# 🔧 Admin Endpoints
# ============================================================================

@router.post(
    "/promote-model",
    tags=["Admin"],
    dependencies=[Depends(require_permission("admin"))],
)
async def promote_model(
    request: ModelPromotionRequest,
    model_registry: Any = Depends(get_model_registry),
    current_user: Dict = Depends(get_current_user),
):
    """
    Promote model version to production
    
    **Process:**
    1. Validate model in staging
    2. Run golden tests
    3. Deploy to canary
    4. Monitor metrics
    5. Promote to production
    """
    
    try:
        result = await model_registry.promote_model(
            version=request.version,
            user=current_user["id"],
            canary_percentage=request.canary_percentage,
            run_tests=request.run_tests,
        )
        
        return {
            "status": "success",
            "version": request.version,
            "canary_percentage": request.canary_percentage,
            "result": result,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/rollback",
    tags=["Admin"],
    dependencies=[Depends(require_permission("admin"))],
)
async def rollback_model(
    model_registry: Any = Depends(get_model_registry),
    current_user: Dict = Depends(get_current_user),
):
    """
    Rollback to previous stable model
    
    **Safety:**
    - Automatic rollback on metrics degradation
    - Manual rollback available
    - Audit trail
    """
    
    try:
        result = await model_registry.rollback(user=current_user["id"])
        return {
            "status": "success",
            "previous_version": result["previous_version"],
            "current_version": result["current_version"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/audit-logs",
    response_model=PaginatedResponse[AuditLogResponse],
    tags=["Admin"],
    dependencies=[Depends(require_permission("audit"))],
)
async def get_audit_logs(
    page: int = 1,
    page_size: int = 50,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """
    Get audit logs with filters
    
    **Security:**
    - Requires audit permission
    - Paginated results
    - Date range filtering
    - User filtering
    """
    
    logs = await audit_logger.get_logs(
        page=page,
        page_size=page_size,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
    )
    
    return PaginatedResponse(
        items=logs["items"],
        total=logs["total"],
        page=page,
        page_size=page_size,
        pages=logs["pages"],
    )


# ============================================================================
# 🔥 Performance Optimization Helpers
# ============================================================================

def _generate_cache_key(patient_data: PatientData) -> str:
    """Generate deterministic cache key"""
    data_str = json.dumps(patient_data.dict(), sort_keys=True)
    return f"prediction:{hashlib.md5(data_str.encode()).hexdigest()}"


# ============================================================================
# 📊 Schema Classes (moved from schemas for completeness)
# ============================================================================

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class FeatureImportanceResponse(BaseModel):
    features: List[str]
    scores: List[float]
    description: str
    timestamp: datetime


class DriftReportResponse(BaseModel):
    drift_detected: bool
    drift_share: float
    column_drifts: Dict[str, Dict]
    recommendation: Optional[str]
    timestamp: datetime


class ModelPromotionRequest(BaseModel):
    version: str
    canary_percentage: float = Field(0.05, ge=0, le=1)
    run_tests: bool = True


class AuditLogResponse(BaseModel):
    id: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    timestamp: datetime


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int