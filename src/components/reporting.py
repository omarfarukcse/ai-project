# src/components/reporting.py
"""
Clinical Report Generation with Governance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
import markdown
import jinja2

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalReport:
    """Clinical report data structure"""
    patient_id: str
    report_date: str
    dataset_type: str
    risk_assessment: Dict[str, Any]
    contributing_factors: List[Dict]
    clinical_explanation: str
    recommendations: List[str]
    follow_up: str
    priority: str
    monitoring_plan: List[str]
    educational_materials: List[str]


class ClinicalReporter:
    """
    Advanced Clinical Report Generator with:
    - Automated report generation
    - Multiple formats (JSON, PDF, HTML)
    - Clinical recommendations
    - Follow-up planning
    - Patient education materials
    """
    
    CLINICAL_GUIDELINES = {
        'diabetes': {
            'high_risk': {
                'recommendations': [
                    'Immediate clinical evaluation recommended',
                    'Perform fasting glucose test and HbA1c',
                    'Consult with endocrinologist',
                    'Start diabetes education program',
                    'Lifestyle modification: diet and exercise',
                ],
                'follow_up': 'Follow up within 1 week',
                'priority': 'Urgent',
            },
            'moderate_risk': {
                'recommendations': [
                    'Schedule comprehensive metabolic panel',
                    'Review family history',
                    'Implement dietary changes',
                    'Increase physical activity',
                    'Monitor glucose levels',
                ],
                'follow_up': 'Follow up within 4 weeks',
                'priority': 'High',
            },
            'low_risk': {
                'recommendations': [
                    'Maintain healthy lifestyle',
                    'Regular exercise (30 mins/day)',
                    'Balanced diet with low glycemic index',
                    'Annual screening',
                    'Monitor for symptoms',
                ],
                'follow_up': 'Follow up in 6 months',
                'priority': 'Routine',
            }
        },
        'heart_disease': {
            'high_risk': {
                'recommendations': [
                    'Immediate cardiac evaluation',
                    'Stress test and ECG recommended',
                    'Consult with cardiologist',
                    'Start cardiac medications if indicated',
                    'Lifestyle: diet, exercise, smoking cessation',
                ],
                'follow_up': 'Follow up within 1 week',
                'priority': 'Urgent',
            },
            'moderate_risk': {
                'recommendations': [
                    'Comprehensive lipid panel',
                    'Review cardiovascular risk factors',
                    'Dietary consultation',
                    'Exercise stress test',
                    'Monitor blood pressure',
                ],
                'follow_up': 'Follow up within 4 weeks',
                'priority': 'High',
            },
            'low_risk': {
                'recommendations': [
                    'Maintain cardiovascular health',
                    'Regular exercise (30 mins/day)',
                    'Heart-healthy diet',
                    'Annual cardiovascular screening',
                    'Monitor for symptoms',
                ],
                'follow_up': 'Follow up in 6 months',
                'priority': 'Routine',
            }
        }
    }
    
    def __init__(
        self,
        dataset_type: str = 'diabetes',
        template_dir: Optional[str] = None,
        output_dir: str = 'outputs/reports/',
    ):
        self.dataset_type = dataset_type
        self.guidelines = self.CLINICAL_GUIDELINES.get(
            dataset_type,
            self.CLINICAL_GUIDELINES['diabetes']
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template engine
        self.template_dir = template_dir or str(Path(__file__).parent / 'templates')
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        logger.info(f"📄 ClinicalReporter initialized: {dataset_type}")
    
    # ============================================================================
    # 🚀 Report Generation
    # ============================================================================
    
    def generate_patient_report(
        self,
        patient_id: str,
        risk_score: float,
        risk_level: str,
        contributing_factors: List[Dict],
        explanation: str,
        feature_values: Dict,
    ) -> ClinicalReport:
        """Generate comprehensive patient report"""
        
        # Get guidelines based on risk level
        risk_key = risk_level.lower().replace(' ', '_')
        guidelines = self.guidelines.get(risk_key, self.guidelines['moderate_risk'])
        
        # Create report
        report = ClinicalReport(
            patient_id=patient_id,
            report_date=datetime.now().isoformat(),
            dataset_type=self.dataset_type,
            risk_assessment={
                'risk_score': risk_score,
                'risk_level': risk_level,
                'interpretation': self._interpret_risk(risk_score),
            },
            contributing_factors=contributing_factors[:5],
            clinical_explanation=explanation,
            recommendations=guidelines['recommendations'],
            follow_up=guidelines['follow_up'],
            priority=guidelines['priority'],
            monitoring_plan=self._generate_monitoring_plan(risk_level),
            educational_materials=self._get_educational_materials(risk_level),
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _interpret_risk(self, risk_score: float) -> str:
        """Provide clinical interpretation"""
        if risk_score < 30:
            return "Low risk. Continue preventive care and routine monitoring."
        elif risk_score < 60:
            return "Moderate risk. Consider additional screening and lifestyle modifications."
        else:
            return "High risk. Immediate clinical intervention recommended."
    
    def _generate_monitoring_plan(self, risk_level: str) -> List[str]:
        """Generate monitoring plan"""
        
        base_plan = [
            'Track vital signs monthly',
            'Monitor for symptoms',
            'Maintain medication compliance',
        ]
        
        if risk_level == 'High Risk':
            return [
                'Daily blood glucose monitoring',
                'Weekly blood pressure checks',
                'Monthly lab work',
                'Quarterly specialist review',
            ] + base_plan
        elif risk_level == 'Moderate Risk':
            return [
                'Blood glucose monitoring 3x per week',
                'Bi-weekly blood pressure checks',
                'Every 3 months lab work',
                'Semi-annual specialist review',
            ] + base_plan
        else:
            return [
                'Blood glucose monitoring weekly',
                'Monthly blood pressure checks',
                'Every 6 months lab work',
                'Annual specialist review',
            ] + base_plan
    
    def _get_educational_materials(self, risk_level: str) -> List[str]:
        """Get educational materials"""
        
        materials = {
            'High Risk': [
                'Diabetes: A Comprehensive Guide',
                'Nutrition for Managing Diabetes',
                'Exercise Guidelines for Diabetes',
                'Medication Management Guide',
                'Emergency Preparedness',
            ],
            'Moderate Risk': [
                'Understanding Diabetes Risk',
                'Healthy Eating Guidelines',
                'Physical Activity Recommendations',
                'Stress Management',
            ],
            'Low Risk': [
                'Diabetes Prevention Program',
                'Nutrition and Wellness Guide',
                'Exercise for Health',
                'Healthy Living Tips',
            ]
        }
        
        return materials.get(risk_level, materials['Low Risk'])
    
    # ============================================================================
    # 💾 Report Storage
    # ============================================================================
    
    def _save_report(self, report: ClinicalReport):
        """Save report to files"""
        
        # Convert to dict
        report_dict = report.__dict__
        
        # Save JSON
        json_path = self.output_dir / f"report_{report.patient_id}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Generate HTML
        html_path = self.output_dir / f"report_{report.patient_id}_{datetime.now().strftime('%Y%m%d')}.html"
        self._generate_html_report(report_dict, html_path)
        
        # Generate Markdown
        md_path = self.output_dir / f"report_{report.patient_id}_{datetime.now().strftime('%Y%m%d')}.md"
        self._generate_markdown_report(report_dict, md_path)
        
        logger.info(f"✅ Report saved: {json_path}")
        return json_path
    
    def _generate_html_report(self, report_dict: Dict, output_path: Path):
        """Generate HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Report - {report_dict['patient_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .risk-high {{ color: #e74c3c; font-weight: bold; }}
                .risk-moderate {{ color: #f39c12; font-weight: bold; }}
                .risk-low {{ color: #27ae60; font-weight: bold; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
                .factor {{ margin: 5px 0; padding: 5px; background: white; border-radius: 3px; }}
                .positive {{ color: #e74c3c; }}
                .negative {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical Decision Support Report</h1>
                <p>Patient: {report_dict['patient_id']} | Date: {report_dict['report_date']}</p>
            </div>
            
            <div class="section">
                <h2>Risk Assessment</h2>
                <p>Risk Score: <strong>{report_dict['risk_assessment']['risk_score']:.1f}%</strong></p>
                <p>Risk Level: <span class="risk-{report_dict['risk_assessment']['risk_level'].lower().replace(' ', '-')}">
                    {report_dict['risk_assessment']['risk_level']}</span></p>
                <p>{report_dict['risk_assessment']['interpretation']}</p>
            </div>
            
            <div class="section">
                <h2>Contributing Factors</h2>
                <ul>
                {''.join([
                    f'<li class="factor"><span class="{f["direction"]}">{f["feature"]}</span>: '
                    f'Contribution {f["contribution"]:.3f} ({f["direction"]})</li>'
                    for f in report_dict['contributing_factors'][:5]
                ])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Clinical Explanation</h2>
                <p>{report_dict['clinical_explanation']}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {''.join([f'<li>{r}</li>' for r in report_dict['recommendations']])}
                </ul>
                <p><strong>Follow-up:</strong> {report_dict['follow_up']}</p>
                <p><strong>Priority:</strong> {report_dict['priority']}</p>
            </div>
            
            <div class="section">
                <h2>Monitoring Plan</h2>
                <ul>
                {''.join([f'<li>{m}</li>' for m in report_dict['monitoring_plan']])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Educational Materials</h2>
                <ul>
                {''.join([f'<li>{m}</li>' for m in report_dict['educational_materials']])}
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 12px;">
                Generated by CDSS v3.0.0 | For Clinical Use Only
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, report_dict: Dict, output_path: Path):
        """Generate Markdown report"""
        
        md_content = f"""# Clinical Decision Support Report

**Patient:** {report_dict['patient_id']}
**Date:** {report_dict['report_date']}
**Dataset:** {report_dict['dataset_type']}

---

## Risk Assessment

- **Risk Score:** {report_dict['risk_assessment']['risk_score']:.1f}%
- **Risk Level:** {report_dict['risk_assessment']['risk_level']}
- **Interpretation:** {report_dict['risk_assessment']['interpretation']}

---

## Contributing Factors

{chr(10).join([
    f"- **{f['feature']}**: Contribution {f['contribution']:.3f} ({f['direction']})"
    for f in report_dict['contributing_factors'][:5]
])}

---

## Clinical Explanation

{report_dict['clinical_explanation']}

---

## Recommendations

{chr(10).join([f"- {r}" for r in report_dict['recommendations']])}

**Follow-up:** {report_dict['follow_up']}
**Priority:** {report_dict['priority']}

---

## Monitoring Plan

{chr(10).join([f"- {m}" for m in report_dict['monitoring_plan']])}

---

## Educational Materials

{chr(10).join([f"- {m}" for m in report_dict['educational_materials']])}

---

*Generated by CDSS v3.0.0 | For Clinical Use Only*
"""
        
        with open(output_path, 'w') as f:
            f.write(md_content)