# dashboard/explainability_panel.py
"""
Explainability Panel for Clinical AI Transparency
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from src.api.dependencies import get_explainability_engine


class ExplainabilityPanel:
    """Explainability panel for clinical AI predictions"""
    
    def __init__(self):
        self.explain_engine = get_explainability_engine()
        
        st.set_page_config(
            page_title="AI Explainability Panel",
            page_icon="🧠",
            layout="wide"
        )
    
    def render(self):
        """Render explainability panel"""
        
        st.title("🧠 AI Explainability Panel")
        st.markdown("Understanding why the AI made a prediction")
        st.markdown("---")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="Enter Patient ID")
        
        with col2:
            model_type = st.selectbox(
                "Model",
                ["Diabetes Risk", "Heart Disease", "Hypertension"]
            )
        
        if patient_id:
            self._render_explanation(patient_id, model_type)
        else:
            self._render_empty_state()
    
    def _render_empty_state(self):
        """Render empty state"""
        
        st.info("👆 Enter a Patient ID to see explanation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Explanations Generated", "1,247")
        with col2:
            st.metric("Average Confidence", "87%")
        with col3:
            st.metric("Top Factors", "Glucose, BMI, Age")
    
    def _render_explanation(self, patient_id: str, model_type: str):
        """Render explanation for a prediction"""
        
        # Tabs
        tabs = st.tabs([
            "📊 Overview",
            "📈 Feature Importance",
            "💡 Clinical Interpretation",
            "📉 SHAP Analysis"
        ])
        
        with tabs[0]:
            self._render_overview_explanation(patient_id)
        
        with tabs[1]:
            self._render_feature_importance(patient_id)
        
        with tabs[2]:
            self._render_clinical_interpretation(patient_id)
        
        with tabs[3]:
            self._render_shap_analysis(patient_id)
    
    def _render_overview_explanation(self, patient_id: str):
        """Render overview explanation"""
        
        st.subheader("Prediction Overview")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Prediction:** High Risk")
            st.markdown("**Confidence:** 92%")
            st.markdown("**Model:** XGBoost v3.0")
        
        with col2:
            st.markdown("**Patient ID:** " + patient_id)
            st.markdown("**Assessment Date:** Today")
            st.markdown("**Assessment Time:** 10:30 AM")
        
        with col3:
            st.metric("Risk Score", "82", "High")
        
        # Decision path
        st.subheader("Decision Path")
        
        path_data = {
            "Step": ["Input Data", "Feature Processing", "Model Prediction", "Risk Assessment"],
            "Description": [
                "Patient clinical data collected",
                "Features processed and scaled",
                "Model predicted 82% probability",
                "Risk classified as High",
            ],
            "Status": ["✅", "✅", "✅", "⚠️"],
        }
        st.table(pd.DataFrame(path_data))
    
    def _render_feature_importance(self, patient_id: str):
        """Render feature importance visualization"""
        
        st.subheader("Feature Importance")
        
        # Sample feature importance data
        features = {
            "Glucose": 0.35,
            "BMI": 0.25,
            "Age": 0.20,
            "Blood Pressure": 0.12,
            "Cholesterol": 0.08,
        }
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(features.values()),
                y=list(features.keys()),
                orientation='h',
                marker_color=['#e74c3c' if v > 0.2 else '#3498db' for v in features.values()],
                text=[f"{v*100:.1f}%" for v in features.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top Contributing Features",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Glucose** and **BMI** are the most important factors for this prediction")
    
    def _render_clinical_interpretation(self, patient_id: str):
        """Render clinical interpretation"""
        
        st.subheader("Clinical Interpretation")
        
        st.markdown("""
        ### Risk Factors
        1. **Glucose Level (148 mg/dL)** - Significantly elevated, major contributor to risk
        2. **BMI (33.6)** - Obese, increases risk by 21%
        3. **Age (50 years)** - Age-related risk factor
        
        ### Clinical Recommendations
        - Monitor blood glucose levels closely
        - Implement lifestyle modifications
        - Consider metformin therapy
        - Regular follow-up scheduling
        """)
        
        st.markdown("---")
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=87,
            title={'text': "Explanation Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 50], 'color': "#e74c3c"},
                    {'range': [50, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#2ecc71"},
                ],
            }
        ))
        fig.update_layout(height=250, width=300)
        st.plotly_chart(fig)
    
    def _render_shap_analysis(self, patient_id: str):
        """Render SHAP analysis visualization"""
        
        st.subheader("SHAP Analysis")
        
        st.info("SHAP values show the contribution of each feature to the prediction")
        
        # Sample SHAP data
        shap_data = {
            "Feature": ["Glucose", "BMI", "Age", "Blood Pressure", "Cholesterol"],
            "SHAP Value": [0.45, 0.32, 0.18, 0.08, -0.03],
            "Impact": ["↑ Risk", "↑ Risk", "↑ Risk", "↑ Risk", "↓ Risk"],
        }
        st.table(pd.DataFrame(shap_data))
        
        # Waterfall plot placeholder
        st.subheader("Waterfall Plot")
        st.image("https://via.placeholder.com/800x400?text=SHAP+Waterfall+Plot", use_column_width=True)
        
        st.markdown("""
        ### What This Means
        - **Positive SHAP values** increase risk
        - **Negative SHAP values** decrease risk
        - **Glucose** has the largest positive impact on risk
        - **Cholesterol** slightly reduces overall risk
        """)


if __name__ == "__main__":
    panel = ExplainabilityPanel()
    panel.render()