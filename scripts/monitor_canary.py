# scripts/monitor_canary.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import requests
import json
import time
from datetime import datetime
import argparse

from src.logger import get_logger
from src.monitoring.drift_detection import DriftDetector

logger = get_logger(__name__)

class CanaryMonitor:
    """Advanced canary deployment monitoring with statistical analysis"""
    
    def __init__(self, api_url: str, canary_endpoint: str, production_endpoint: str):
        self.api_url = api_url
        self.canary_endpoint = canary_endpoint
        self.production_endpoint = production_endpoint
        self.drift_detector = DriftDetector()
        self.results = []
    
    def collect_samples(self, n_samples: int = 1000, duration: int = 300) -> Dict[str, List]:
        """Collect prediction samples from both endpoints"""
        logger.info(f"Collecting {n_samples} samples from canary and production")
        
        canary_predictions = []
        production_predictions = []
        start_time = time.time()
        
        # Generate test patients
        patients = self._generate_test_patients(n_samples)
        
        # Get auth token
        token = self._get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        for i, patient in enumerate(patients):
            # Check if duration exceeded
            if time.time() - start_time > duration:
                break
            
            # Canary prediction
            try:
                response = requests.post(
                    f"{self.api_url}{self.canary_endpoint}",
                    json={"patient": patient},
                    headers=headers,
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    canary_predictions.append({
                        'probability': data['probability'],
                        'risk_status': data['risk_status'],
                        'confidence': data['confidence']
                    })
            except Exception as e:
                logger.warning(f"Canary prediction failed: {e}")
            
            # Production prediction
            try:
                response = requests.post(
                    f"{self.api_url}{self.production_endpoint}",
                    json={"patient": patient},
                    headers=headers,
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    production_predictions.append({
                        'probability': data['probability'],
                        'risk_status': data['risk_status'],
                        'confidence': data['confidence']
                    })
            except Exception as e:
                logger.warning(f"Production prediction failed: {e}")
            
            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.1)
        
        return {
            'canary': canary_predictions,
            'production': production_predictions
        }
    
    def _generate_test_patients(self, n: int) -> List[Dict]:
        """Generate realistic test patients"""
        patients = []
        for _ in range(n):
            patients.append({
                "Pregnancies": np.random.randint(0, 10),
                "Glucose": np.random.uniform(60, 200),
                "BloodPressure": np.random.uniform(60, 140),
                "SkinThickness": np.random.uniform(10, 60),
                "Insulin": np.random.uniform(0, 400),
                "BMI": np.random.uniform(15, 45),
                "DiabetesPedigreeFunction": np.random.uniform(0.1, 2.0),
                "Age": np.random.randint(20, 80)
            })
        return patients
    
    def _get_auth_token(self) -> str:
        """Get authentication token"""
        response = requests.post(
            f"{self.api_url}/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        return response.json().get('access_token')
    
    def analyze(self, samples: Dict[str, List]) -> Dict[str, Any]:
        """Analyze differences between canary and production"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'canary_count': len(samples['canary']),
            'production_count': len(samples['production']),
            'differences': {},
            'recommendations': []
        }
        
        if not samples['canary'] or not samples['production']:
            results['error'] = 'Insufficient samples'
            return results
        
        # Convert to DataFrames
        canary_df = pd.DataFrame(samples['canary'])
        production_df = pd.DataFrame(samples['production'])
        
        # 1. Probability distribution comparison (KS Test)
        ks_stat, ks_pvalue = stats.ks_2samp(
            canary_df['probability'],
            production_df['probability']
        )
        
        results['differences']['probability_ks_stat'] = ks_stat
        results['differences']['probability_ks_pvalue'] = ks_pvalue
        
        if ks_stat > 0.1:
            results['recommendations'].append(
                f"⚠️ Probability distribution differs significantly (KS={ks_stat:.3f})"
            )
        
        # 2. Risk status distribution
        canary_risk = canary_df['risk_status'].value_counts(normalize=True)
        production_risk = production_df['risk_status'].value_counts(normalize=True)
        
        for risk in ['HIGH RISK', 'MODERATE RISK', 'LOW RISK']:
            diff = abs(canary_risk.get(risk, 0) - production_risk.get(risk, 0))
            results['differences'][f'risk_{risk}_diff'] = diff
            if diff > 0.1:
                results['recommendations'].append(
                    f"⚠️ {risk} distribution differs by {diff:.1%}"
                )
        
        # 3. Confidence comparison
        results['differences']['canary_avg_confidence'] = canary_df['confidence'].apply(
            lambda x: 1 if x == 'High Confidence' else 0.5 if x == 'Moderate Confidence' else 0
        ).mean()
        
        results['differences']['production_avg_confidence'] = production_df['confidence'].apply(
            lambda x: 1 if x == 'High Confidence' else 0.5 if x == 'Moderate Confidence' else 0
        ).mean()
        
        # 4. Statistical summary
        results['statistics'] = {
            'canary': {
                'mean_prob': canary_df['probability'].mean(),
                'std_prob': canary_df['probability'].std(),
                'median_prob': canary_df['probability'].median()
            },
            'production': {
                'mean_prob': production_df['probability'].mean(),
                'std_prob': production_df['probability'].std(),
                'median_prob': production_df['probability'].median()
            }
        }
        
        # 5. Final decision
        if ks_stat < 0.05 and all(diff < 0.05 for diff in results['differences'].values() 
                                 if isinstance(diff, float)):
            results['decision'] = 'PROMOTE'
            results['recommendations'].append('✅ Canary behavior matches production - promote')
        elif ks_stat < 0.10:
            results['decision'] = 'MONITOR'
            results['recommendations'].append('⚠️ Minor differences detected - continue monitoring')
        else:
            results['decision'] = 'ROLLBACK'
            results['recommendations'].append('❌ Significant differences detected - rollback recommended')
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-url', default='http://localhost:8000')
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--samples', type=int, default=1000)
    args = parser.parse_args()
    
    monitor = CanaryMonitor(
        api_url=args.api_url,
        canary_endpoint='/api/v1/predict',
        production_endpoint='/api/v1/predict'
    )
    
    print("🔄 Collecting samples...")
    samples = monitor.collect_samples(n_samples=args.samples, duration=args.duration)
    
    print("📊 Analyzing results...")
    results = monitor.analyze(samples)
    
    print("\n" + "=" * 60)
    print("CANARY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📈 Samples collected:")
    print(f"  Canary: {results['canary_count']}")
    print(f"  Production: {results['production_count']}")
    
    print(f"\n📊 Statistical comparison:")
    if 'statistics' in results:
        print(f"  Canary mean probability: {results['statistics']['canary']['mean_prob']:.3f}")
        print(f"  Production mean probability: {results['statistics']['production']['mean_prob']:.3f}")
    
    print(f"\n🔬 KS Test:")
    print(f"  Statistic: {results['differences'].get('probability_ks_stat', 0):.3f}")
    print(f"  P-value: {results['differences'].get('probability_ks_pvalue', 0):.3f}")
    
    print(f"\n📋 Recommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")
    
    print(f"\n🎯 Decision: {results.get('decision', 'INCONCLUSIVE')}")
    print("=" * 60)
    
    # Save results
    with open('canary_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    if results.get('decision') == 'ROLLBACK':
        sys.exit(1)
    elif results.get('decision') == 'PROMOTE':
        sys.exit(0)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()