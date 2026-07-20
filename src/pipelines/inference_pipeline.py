# src/pipelines/inference_pipeline.py
"""
Production Inference Pipeline with Enterprise Features

Features:
- Sub-100ms latency
- Batch processing
- Model versioning
- Fallback strategies
- Circuit breaker
- Data validation
- Performance monitoring
- Cache integration
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.components.model_registry import ModelRegistry
from src.components.preprocessing import ClinicalPreprocessor
from src.components.explainability import ClinicalSHAPExplainer
from src.components.fallback_system import FallbackSystem
from src.monitoring.drift_detection import DriftDetector
from src.validation.schema_validation import DataValidator
from src.caching.redis_client import FastRedisClient
from src.utils.circuit_breaker import CircuitBreaker
from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Prediction result with metadata"""
    patient_id: str
    risk_score: float
    risk_level: str
    probability: float
    top_factors: List[str]
    contributing_factors: List[Dict]
    clinical_explanation: str
    confidence: float
    drift_detected: bool
    model_version: str
    processing_time_ms: float
    correlation_id: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "probability": self.probability,
            "top_factors": self.top_factors,
            "contributing_factors": self.contributing_factors,
            "clinical_explanation": self.clinical_explanation,
            "confidence": self.confidence,
            "drift_detected": self.drift_detected,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


class InferencePipeline:
    """
    Production inference pipeline with enterprise features
    
    Features:
    - <100ms latency for single predictions
    - Batch processing with parallelization
    - Automatic fallback on failure
    - Data drift detection
    - Model version tracking
    - Response caching
    - Performance metrics
    """
    
    def __init__(
        self,
        model_version: Optional[str] = None,
        use_cache: bool = True,
        fallback_enabled: bool = True,
        drift_detection_enabled: bool = True,
    ):
        self.model_version = model_version or "production"
        self.use_cache = use_cache
        self.fallback_enabled = fallback_enabled
        self.drift_detection_enabled = drift_detection_enabled
        
        # Components
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.registry = ModelRegistry()
        self.fallback = FallbackSystem() if fallback_enabled else None
        self.drift_detector = DriftDetector() if drift_detection_enabled else None
        self.validator = DataValidator()
        self.redis_client = FastRedisClient()
        self.circuit_breaker = CircuitBreaker(name="inference")
        
        # Thread pool for batch processing
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Metrics
        self.metrics = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_used": 0,
            "drift_detected": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
        }
        
        # Load model
        self._load_model()
        
        logger.info("⚡ Inference Pipeline initialized")
        logger.info(f"   Model: {self.model_version}")
        logger.info(f"   Cache: {'Enabled' if use_cache else 'Disabled'}")
        logger.info(f"   Fallback: {'Enabled' if fallback_enabled else 'Disabled'}")
        logger.info(f"   Drift Detection: {'Enabled' if drift_detection_enabled else 'Disabled'}")
    
    def _load_model(self):
        """Load model from registry"""
        try:
            if self.model_version == "production":
                self.model = self.registry.get_production_model()
            elif self.model_version == "staging":
                self.model = self.registry.get_staging_model()
            else:
                self.model = self.registry.get_model(self.model_version)
            
            # Load associated components
            self.preprocessor = ClinicalPreprocessor()
            self.preprocessor.load_from_model(self.model)
            
            self.explainer = ClinicalSHAPExplainer(self.model)
            
            logger.info(f"✅ Model loaded: {self.model_version}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            if self.fallback_enabled:
                logger.warning("⚠️ Using fallback system")
                self.model = None
            else:
                raise
    
    # ============================================================================
    # 🚀 Prediction Methods
    # ============================================================================
    
    async def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        correlation_id: str,
        explain: bool = True,
        validate: bool = True,
    ) -> PredictionResult:
        """
        Predict risk for a single patient
        
        Args:
            data: Patient data as dict or DataFrame
            correlation_id: Request correlation ID
            explain: Generate SHAP explanations
            validate: Validate input data
            
        Returns:
            PredictionResult with prediction and metadata
        """
        
        start_time = time.perf_counter()
        self.metrics["total_predictions"] += 1
        
        try:
            # Step 1: Validate input
            if validate:
                data = await self._validate_input(data, correlation_id)
            
            # Step 2: Check cache
            if self.use_cache:
                cache_key = self._generate_cache_key(data)
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    logger.info(f"✅ Cache hit: {correlation_id}")
                    return self._parse_cached_result(cached_result)
                self.metrics["cache_misses"] += 1
            
            # Step 3: Check drift
            drift_detected = False
            if self.drift_detection_enabled and self.drift_detector:
                drift_report = self.drift_detector.detect_drift(data)
                drift_detected = drift_report.get('drift_detected', False)
                if drift_detected:
                    self.metrics["drift_detected"] += 1
                    logger.warning(f"⚠️ Data drift detected: {correlation_id}")
            
            # Step 4: Run inference with circuit breaker
            with self.circuit_breaker:
                if self.model is None and self.fallback_enabled:
                    # Use fallback
                    self.metrics["fallback_used"] += 1
                    logger.warning(f"⚠️ Using fallback system: {correlation_id}")
                    result = await self._fallback_predict(data, correlation_id)
                else:
                    # Use ML model
                    result = await self._model_predict(
                        data, correlation_id, drift_detected, explain
                    )
            
            # Step 5: Cache result
            if self.use_cache and not result.drift_detected:
                await self._cache_result(cache_key, result)
            
            # Step 6: Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency_ms)
            result.processing_time_ms = latency_ms
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}", exc_info=True)
            
            # Try fallback if enabled
            if self.fallback_enabled:
                logger.warning(f"⚠️ Using fallback after failure: {correlation_id}")
                self.metrics["fallback_used"] += 1
                return await self._fallback_predict(data, correlation_id)
            
            # Re-raise if no fallback
            raise
    
    async def batch_predict(
        self,
        data_list: List[Union[pd.DataFrame, Dict]],
        correlation_id: str,
        explain: bool = False,
        max_concurrent: int = 50,
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple patients
        
        Args:
            data_list: List of patient data
            correlation_id: Base correlation ID
            explain: Generate explanations
            max_concurrent: Max concurrent predictions
            
        Returns:
            List of PredictionResult objects
        """
        
        start_time = time.perf_counter()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_predict(data, idx):
            async with semaphore:
                return await self.predict(
                    data,
                    f"{correlation_id}_{idx}",
                    explain=explain,
                    validate=True
                )
        
        # Execute all predictions
        tasks = [
            limited_predict(data, idx)
            for idx, data in enumerate(data_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle failures
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch prediction {idx} failed: {str(result)}")
                # Use fallback for failed predictions
                if self.fallback_enabled:
                    fallback_result = await self._fallback_predict(
                        data_list[idx],
                        f"{correlation_id}_{idx}_fallback"
                    )
                    final_results.append(fallback_result)
            else:
                final_results.append(result)
        
        logger.info(f"✅ Batch prediction complete: {len(final_results)} results")
        logger.info(f"   Duration: {(time.perf_counter() - start_time)*1000:.2f}ms")
        
        return final_results
    
    # ============================================================================
    # 🔧 Core Prediction Methods
    # ============================================================================
    
    async def _model_predict(
        self,
        data: pd.DataFrame,
        correlation_id: str,
        drift_detected: bool,
        explain: bool,
    ) -> PredictionResult:
        """Run ML model prediction"""
        
        # Preprocess data
        features = self.preprocessor.transform(data)
        
        # Get prediction
        probability = self.model.predict_proba(features)[0][1]
        risk_score = probability * 100
        
        # Get feature importance
        if explain and self.explainer:
            explanation = self.explainer.generate_local_explanations(features, 0)
            top_factors = explanation['top_factors']
            contributing_factors = explanation['contributing_factors']
            clinical_explanation = explanation['clinical_explanation']
        else:
            top_factors = []
            contributing_factors = []
            clinical_explanation = "Explanation not available"
        
        # Determine risk level
        risk_level = self._classify_risk(risk_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(probability)
        
        # Generate patient ID
        patient_id = f"P{int(time.time()*1000):x}{hash(str(data)):x}"[:16]
        
        return PredictionResult(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            probability=probability,
            top_factors=top_factors[:5],
            contributing_factors=contributing_factors[:10],
            clinical_explanation=clinical_explanation,
            confidence=confidence,
            drift_detected=drift_detected,
            model_version=self.model_version,
            processing_time_ms=0,
            correlation_id=correlation_id,
        )
    
    async def _fallback_predict(
        self,
        data: pd.DataFrame,
        correlation_id: str,
    ) -> PredictionResult:
        """Fallback prediction using rule-based system"""
        
        result = self.fallback.predict(data)
        
        return PredictionResult(
            patient_id=f"FALLBACK_{hash(str(data)):x}"[:16],
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            probability=result.get('probability', result['risk_score'] / 100),
            top_factors=['fallback', 'rule_based'],
            contributing_factors=[],
            clinical_explanation=result.get('explanation', "Using fallback system"),
            confidence=0.6,
            drift_detected=False,
            model_version=f"fallback_v{self.fallback.version}",
            processing_time_ms=0,
            correlation_id=correlation_id,
        )
    
    # ============================================================================
    # 🔧 Validation Methods
    # ============================================================================
    
    async def _validate_input(
        self,
        data: Union[pd.DataFrame, Dict],
        correlation_id: str,
    ) -> pd.DataFrame:
        """Validate input data against schema"""
        
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Validate
        is_valid, errors = self.validator.validate(data)
        
        if not is_valid:
            logger.error(f"❌ Input validation failed: {errors}")
            raise ValueError(f"Invalid input data: {errors}")
        
        return data
    
    # ============================================================================
    # 🔧 Cache Methods
    # ============================================================================
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key from data"""
        import hashlib
        data_str = data.to_json(sort_keys=True)
        return f"prediction:{hashlib.md5(data_str.encode()).hexdigest()}"
    
    async def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if result is cached"""
        try:
            if self.redis_client._connected:
                return await self.redis_client.get(cache_key)
        except Exception:
            pass
        return None
    
    async def _cache_result(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        try:
            if self.redis_client._connected:
                await self.redis_client.set(
                    cache_key,
                    result.to_dict(),
                    ttl=300  # 5 minutes
                )
        except Exception:
            pass
    
    def _parse_cached_result(self, cached: Dict) -> PredictionResult:
        """Parse cached result"""
        return PredictionResult(**cached)
    
    # ============================================================================
    # 🔧 Helper Methods
    # ============================================================================
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level"""
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 60:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate prediction confidence"""
        # Confidence based on probability distance from 0.5
        confidence = abs(probability - 0.5) * 2
        return min(confidence, 0.99)
    
    def _update_metrics(self, latency_ms: float):
        """Update performance metrics"""
        # Simple moving average
        alpha = 0.1
        self.metrics["avg_latency_ms"] = (
            (1 - alpha) * self.metrics["avg_latency_ms"] +
            alpha * latency_ms
        )
        
        # Store for percentile calculation
        if not hasattr(self, '_latency_history'):
            self._latency_history = []
        
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-1000:]
        
        # Update percentiles
        if len(self._latency_history) >= 100:
            sorted_latencies = sorted(self._latency_history)
            self.metrics["p95_latency_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.metrics["p99_latency_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    # ============================================================================
    # 🔧 Model Management
    # ============================================================================
    
    def reload_model(self, version: Optional[str] = None):
        """Reload model from registry"""
        if version:
            self.model_version = version
        self._load_model()
        logger.info(f"🔄 Model reloaded: {self.model_version}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return {
            "model_version": self.model_version,
            **self.metrics,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0
            ),
            "fallback_rate": (
                self.metrics["fallback_used"] / self.metrics["total_predictions"]
                if self.metrics["total_predictions"] > 0
                else 0
            ),
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check pipeline health"""
        return {
            "model_loaded": self.model is not None,
            "preprocessor_ready": self.preprocessor is not None,
            "explainer_ready": self.explainer is not None,
            "fallback_ready": self.fallback_enabled,
            "redis_connected": self.redis_client._connected,
        }