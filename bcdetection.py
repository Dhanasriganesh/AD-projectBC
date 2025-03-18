import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # Changed from RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier  # Changed from GradientBoostingClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class MedicalAnalysisSystem:
    def __init__(self):
        self.setup_page_config()
        self.load_and_process_data()
        self.setup_navigation()

    def setup_page_config(self):
        st.set_page_config(
            page_title="Medical Analysis Assistant",
            page_icon="🩺",
            layout="wide"
        )
        self.apply_custom_styling()

    def apply_custom_styling(self):
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
                background-color: #f8f9fa;
            }
            .stButton>button {
                width: 100%;
                background-color: #007bff;
                color: white;
            }
            .metric-card {
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)

    @st.cache_data
    def load_and_process_data(self):
        # Load dataset
        raw_data = load_breast_cancer()
        
        # Create meaningful feature names
        self.feature_categories = {
            "Size Metrics": [name for name in raw_data.feature_names if 'mean' in name],
            "Shape Indicators": [name for name in raw_data.feature_names if 'texture' in name or 'symmetry' in name],
            "Composition Markers": [name for name in raw_data.feature_names if 'compactness' in name or 'fractal' in name]
        }
        
        # Create DataFrame with enhanced column names
        self.df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
        self.df['analysis_result'] = raw_data.target
        self.raw_data = raw_data

    def setup_navigation(self):
        st.title("Medical Tissue Analysis System")
        st.markdown("*Advanced Analysis Tool for Medical Professionals*")
        
        self.current_page = st.sidebar.radio(
            "System Navigation",
            ["📊 Analysis Dashboard", "🔍 Data Explorer", "🏥 Diagnostic Assistant"]
        )

    def run_analysis_dashboard(self):
        st.header("Analysis Dashboard")
        
        # Enhanced metrics display
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            self.display_metric_card("Total Samples", len(self.df))
        with metrics_cols[1]:
            self.display_metric_card("Healthy Samples", len(self.df[self.df['analysis_result'] == 1]))
        with metrics_cols[2]:
            self.display_metric_card("Concerning Samples", len(self.df[self.df['analysis_result'] == 0]))
        with metrics_cols[3]:
            self.display_metric_card("Analysis Date", datetime.now().strftime("%Y-%m-%d"))

        # Interactive data preview
        st.subheader("Sample Data Preview")
        sample_size = st.slider("Sample size", 5, 100, 10)
        st.dataframe(self.df.sample(sample_size))

    def display_metric_card(self, title, value):
        st.markdown(f"""
            <div class="metric-card">
                <h3>{title}</h3>
                <h2>{value}</h2>
            </div>
        """, unsafe_allow_html=True)

    def run_data_explorer(self):
        st.header("Data Exploration Tools")
        
        # Enhanced correlation analysis
        st.subheader("Feature Relationship Analysis")
        selected_features = st.multiselect(
            "Select features to analyze",
            self.df.columns[:-1],
            default=self.df.columns[:3]
        )
        
        if selected_features:
            correlation_data = self.df[selected_features].corr()
            fig = go.Figure(data=go.Heatmap(
                z=correlation_data,
                x=selected_features,
                y=selected_features,
                colorscale='Viridis'
            ))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    def run_diagnostic_assistant(self):
        st.header("Diagnostic Assistant")
        
        # Initialize and train model
        model = self.initialize_model()
        
        # Organized input collection
        st.subheader("Patient Data Entry")
        user_inputs = {}
        
        for category, features in self.feature_categories.items():
            st.markdown(f"#### {category}")
            cols = st.columns(2)
            for idx, feature in enumerate(features):
                user_inputs[feature] = cols[idx % 2].number_input(
                    f"{feature}",
                    value=float(self.df[feature].mean()),
                    format="%.2f"
                )
        
        if st.button("Generate Analysis"):
            self.perform_analysis(model, user_inputs)

    @st.cache_resource
    def initialize_model(self):
        X = self.df.drop('analysis_result', axis=1)
        y = self.df['analysis_result']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def perform_analysis(self, model, user_inputs):
        input_data = np.array(list(user_inputs.values())).reshape(1, -1)
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        st.subheader("Analysis Results")
        
        # Enhanced results display
        result_cols = st.columns(2)
        with result_cols[0]:
            if prediction[0] == 0:
                st.error("⚠️ Results indicate concerning patterns")
            else:
                st.success("✅ Results indicate normal patterns")
        
        with result_cols[1]:
            fig = px.pie(
                values=probabilities[0],
                names=['Concerning', 'Normal'],
                title='Analysis Confidence',
                color_discrete_sequence=['#ff6b6b', '#51cf66']
            )
            st.plotly_chart(fig)
        
        # Detailed probability breakdown
        st.markdown("### Detailed Analysis")
        st.info(f"Confidence in normal tissue patterns: {probabilities[0][1]:.1%}")
        st.warning(f"Confidence in concerning patterns: {probabilities[0][0]:.1%}")

    def run(self):
        if self.current_page == "📊 Analysis Dashboard":
            self.run_analysis_dashboard()
        elif self.current_page == "🔍 Data Explorer":
            self.run_data_explorer()
        elif self.current_page == "🏥 Diagnostic Assistant":
            self.run_diagnostic_assistant()

if __name__ == "__main__":
    system = MedicalAnalysisSystem()
    system.run()