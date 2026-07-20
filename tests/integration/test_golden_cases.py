# tests/integration/test_golden_cases.py
"""
Golden test cases for model validation
"""

import pytest
import pandas as pd
import json
from pathlib import Path

from src.pipelines.inference_pipeline import InferencePipeline
from src.components.model_registry import ModelRegistry


class TestGoldenCases:
    """Golden test cases for model validation"""
    
    @pytest.fixture
    def golden_tests(self):
        """Load golden test cases"""
        path = Path("data/golden_tests.json")
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return []
    
    @pytest.fixture
    def inference_pipeline(self):
        """Create inference pipeline"""
        return InferencePipeline()
    
    def test_golden_cases(self, golden_tests, inference_pipeline):
        """Run golden test cases"""
        if not golden_tests:
            pytest.skip("No golden tests found")
        
        failures = []
        
        for test_case in golden_tests:
            try:
                # Prepare input
                input_data = test_case['input']
                expected = test_case['expected']
                
                # Run prediction
                result = inference_pipeline.predict(input_data, "golden_test")
                
                # Check results
                if result.risk_level != expected['risk_level']:
                    failures.append({
                        "test": test_case['name'],
                        "expected": expected['risk_level'],
                        "actual": result.risk_level
                    })
                    
            except Exception as e:
                failures.append({
                    "test": test_case['name'],
                    "error": str(e)
                })
        
        assert len(failures) == 0, f"Golden test failures: {failures}"