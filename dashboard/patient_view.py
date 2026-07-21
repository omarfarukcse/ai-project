# dashboard/patient_view.py
"""
Patient View Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.api.dependencies import get_prediction_engine


class PatientView:
    """Patient-facing dashboard"""
    
    def __init__(self):
        self.prediction_engine = get_prediction_engine()
        
        st.set_page_config(
            page_title="My Health Dashboard",
            page_icon="🩺",
            layout="wide"
        )
    
    def render(self):
        """Render patient dashboard"""
        
        st.title("🩺 My Health Dashboard")
        st.markdown("---")
        
        # Patient authentication (simplified)
        patient_id = st.sidebar.text_input("Patient ID", placeholder="Enter your Patient ID")
        
        if patient_id:
            self._render_patient_dashboard(patient_id)
        else:
            self._render_login_prompt()
    
    def _render_login_prompt(self):
        """Render login prompt"""
        
        st.info("👋 Please enter your Patient ID to view your health dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Assessments", "3")
        with col2:
            st.metric("Risk Level", "Low")
        with col3:
            st.metric("Next Check-up", "2 weeks")
    
    def _render_patient_dashboard(self, patient_id: str):
        """Render patient dashboard"""
        
        # Header
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", "22", "Low")
        with col2:
            st.metric("Last Check-up", "1 week ago")
        with col3:
            st.metric("Next Appointment", "2 weeks")
        
        # Risk trend
        st.subheader("📊 Your Health Trend")
        
        # Sample trend data
        dates = pd.date_range(end=datetime.now(), periods=6, freq='M')
        scores = [25, 24, 23, 22, 21, 22]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=10)
        ))
        
        fig.add_hline(y=30, line_dash="dash", line_color="yellow")
        fig.add_hline(y=60, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Risk Score Over Time",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Health summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Clinical Summary")
            summary = {
                "Diabetes Risk": "Low",
                "Heart Disease Risk": "Low",
                "Hypertension Risk": "Low",
                "Overall Health": "Good",
            }
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("💪 Health Tips")
            tips = [
                "Maintain regular exercise routine",
                "Follow balanced diet",
                "Monitor blood pressure monthly",
                "Keep stress levels in check",
            ]
            for tip in tips:
                st.write(f"• {tip}")
        
        # Recommendations
        st.subheader("📝 Recommendations")
        
        with st.expander("Dietary Recommendations", expanded=True):
            st.write("""
            - Include more fruits and vegetables
            - Reduce sodium intake
            - Limit processed foods
            - Stay hydrated
            """)
        
        with st.expander("Exercise Recommendations"):
            st.write("""
            - 30 minutes of moderate activity daily
            - Include both cardio and strength training
            - Start with walking and gradually increase
            """)
        
        with st.expander("Follow-up Schedule"):
            st.write("""
            - Next check-up: 2 weeks
            - Lab tests: 1 month
            - Specialist visit: 3 months
            """)
        
        # Risk factors
        st.subheader("🔬 Your Risk Factors")
        
        factors = {
            "Age": "45 years",
            "BMI": "24.5 (Normal)",
            "Blood Pressure": "120/80",
            "Glucose": "95 mg/dL",
            "Cholesterol": "190 mg/dL",
        }
        
        factor_cols = st.columns(3)
        for idx, (key, value) in enumerate(factors.items()):
            col = factor_cols[idx % 3]
            col.metric(key, value)


if __name__ == "__main__":
    view = PatientView()
    view.render()