# tests/load/locustfile.py
from locust import HttpUser, task, between, events
import json
import random
from typing import Dict, Any, List
import time
from datetime import datetime

class CDSSLoadTest(HttpUser):
    """
    Comprehensive load testing for CDSS system
    Tests system performance under production-like load
    """
    
    wait_time = between(0.5, 2)
    token = None
    
    def on_start(self):
        """Login to get JWT token"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        if response.status_code == 200:
            self.token = response.json().get('access_token')
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}",
                "X-Correlation-ID": f"load_test_{random.randint(1000, 9999)}"
            })
            print(f"✅ Login successful")
        else:
            print(f"❌ Login failed: {response.status_code}")
    
    @task(5)
    def predict_single(self):
        """Test single prediction endpoint"""
        patient = self._generate_patient()
        
        start_time = time.time()
        response = self.client.post("/api/v1/predict", json={
            "patient": patient,
            "include_explanation": False
        })
        
        if response.status_code == 200:
            prediction_time = time.time() - start_time
            data = response.json()
            
            # Record metrics
            self.environment.events.request.fire(
                request_type="PREDICT",
                name="/api/v1/predict",
                response_time=prediction_time * 1000,
                response_length=len(response.content),
                exception=None,
                context=None
            )
            
            # Track risk distribution
            risk_status = data.get('risk_status', 'UNKNOWN')
            self._record_risk_metric(risk_status)
        else:
            print(f"Prediction failed: {response.status_code}")
    
    @task(3)
    def predict_with_explanation(self):
        """Test prediction with explanation"""
        patient = self._generate_patient()
        
        response = self.client.post("/api/v1/predict", json={
            "patient": patient,
            "include_explanation": True
        })
        
        if response.status_code != 200:
            print(f"Prediction with explanation failed: {response.status_code}")
    
    @task(2)
    def batch_predict(self):
        """Test batch prediction endpoint"""
        patients = [self._generate_patient() for _ in range(random.randint(2, 10))]
        
        start_time = time.time()
        response = self.client.post("/api/v1/batch_predict", json={
            "patients": patients
        })
        
        if response.status_code == 200:
            batch_time = time.time() - start_time
            data = response.json()
            
            # Record metrics
            self.environment.events.request.fire(
                request_type="BATCH_PREDICT",
                name="/api/v1/batch_predict",
                response_time=batch_time * 1000,
                response_length=len(response.content),
                exception=None,
                context=None
            )
            
            # Track batch stats
            print(f"Batch prediction: {data.get('total_patients')} patients in {batch_time:.2f}s")
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        response = self.client.get("/api/v1/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
    
    @task(1)
    def explain_prediction(self):
        """Test explanation endpoint"""
        patient = self._generate_patient()
        
        response = self.client.post("/api/v1/explain", json=patient)
        if response.status_code != 200:
            print(f"Explanation failed: {response.status_code}")
    
    def _generate_patient(self) -> Dict[str, Any]:
        """Generate realistic patient data for testing"""
        return {
            "Pregnancies": random.randint(0, 10),
            "Glucose": random.uniform(60, 200),
            "BloodPressure": random.uniform(60, 140),
            "SkinThickness": random.uniform(10, 60),
            "Insulin": random.uniform(0, 400),
            "BMI": random.uniform(15, 45),
            "DiabetesPedigreeFunction": random.uniform(0.1, 2.0),
            "Age": random.randint(20, 80),
            "Sex": random.randint(0, 1)
        }
    
    def _record_risk_metric(self, risk_status: str):
        """Record risk status for metrics"""
        # Custom metric recording
        if hasattr(self.environment, 'stats'):
            stats = self.environment.stats
            if not hasattr(stats, 'risk_counts'):
                stats.risk_counts = {}
            stats.risk_counts[risk_status] = stats.risk_counts.get(risk_status, 0) + 1

# Custom response for load test results
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary when test stops"""
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    
    # Print request stats
    print("\nRequest Statistics:")
    for stat in environment.stats.entries.values():
        if stat.num_requests > 0:
            print(f"  {stat.name}:")
            print(f"    Requests: {stat.num_requests}")
            print(f"    Avg: {stat.avg_response_time:.2f}ms")
            print(f"    P95: {stat.get_response_time_percentile(0.95):.2f}ms")
            print(f"    Failures: {stat.num_failures}")
            print(f"    RPS: {stat.total_rps:.2f}")
    
    # Print risk distribution
    if hasattr(environment.stats, 'risk_counts'):
        print("\nRisk Distribution:")
        for risk, count in environment.stats.risk_counts.items():
            print(f"  {risk}: {count}")

# Run with:
# locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=5m --web-port=8089