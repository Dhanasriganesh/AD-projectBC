import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
import plotly.express as px
import plotly.graph_objects as go

# Custom configuration
st.set_page_config(
    page_title="Cancer Detection Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Application header
st.title("Medical Imaging Analysis System")
st.markdown("*An AI-powered tool for analyzing medical diagnostic data*")

# Data loading with custom preprocessing
@st.cache_data
def prepare_dataset():
    dataset = load_breast_cancer()
    # Create custom feature names that are more descriptive
    feature_mapping = {
        name: f"biomarker_{i+1}_{name.replace(' ', '_')}" 
        for i, name in enumerate(dataset.feature_names)
    }
    
    data_df = pd.DataFrame(dataset.data, columns=[feature_mapping[name] for name in dataset.feature_names])
    data_df['diagnosis'] = dataset.target
    return data_df, dataset, feature_mapping

data_df, dataset, feature_mapping = prepare_dataset()

# Custom navigation
navigation = st.sidebar.selectbox(
    "Navigation Menu",
    ["📊 Data Insights", "📈 Visual Analytics", "🔍 Diagnostic Tool"]
)

if navigation == "📊 Data Insights":
    st.header("Dataset Insights")
    
    # Added custom metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases", len(data_df))
    with col2:
        st.metric("Non-Cancerous Cases", len(data_df[data_df['diagnosis'] == 1]))
    with col3:
        st.metric("Cancerous Cases", len(data_df[data_df['diagnosis'] == 0]))
    
    # Interactive data viewer
    st.subheader("Interactive Data Explorer")
    rows_to_show = st.slider("Number of rows to display", 5, 50)
    st.dataframe(data_df.head(rows_to_show))
    
    # Custom statistics
    st.subheader("Statistical Analysis")
    selected_stats = st.multiselect(
        "Select statistics to view",
        ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        default=['mean', '50%']
    )
    if selected_stats:
        st.dataframe(data_df.describe().loc[selected_stats])

elif navigation == "📈 Visual Analytics":
    st.header("Visual Data Analysis")
    
    # Interactive correlation matrix using plotly
    st.subheader("Interactive Correlation Analysis")
    corr_matrix = data_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu'
    ))
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom feature distribution
    st.subheader("Feature Distribution Analysis")
    selected_feature = st.selectbox(
        "Select Biomarker",
        list(feature_mapping.values())
    )
    
    fig = px.histogram(
        data_df,
        x=selected_feature,
        color='diagnosis',
        barmode='overlay',
        labels={'diagnosis': 'Diagnosis Type'},
        color_discrete_map={0: 'red', 1: 'green'}
    )
    st.plotly_chart(fig)

elif navigation == "🔍 Diagnostic Tool":
    st.header("Diagnostic Analysis Tool")
    
    # Model training with custom parameters
    @st.cache_resource
    def initialize_model():
        X = data_df.drop('diagnosis', axis=1)
        y = data_df['diagnosis']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        return model, scaler

    model, scaler = initialize_model()
    
    # Custom input interface
    st.subheader("Patient Data Entry")
    
    # Group related features
    feature_groups = {
        "Mean Values": [f for f in data_df.columns if 'mean' in f],
        "Standard Error Values": [f for f in data_df.columns if 'error' in f],
        "Worst Values": [f for f in data_df.columns if 'worst' in f]
    }
    
    user_inputs = {}
    
    for group_name, features in feature_groups.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            user_inputs[feature] = cols[idx % 2].number_input(
                feature,
                value=float(data_df[feature].mean()),
                format="%.6f"
            )
    
    # Enhanced prediction display
    if st.button("Run Diagnostic Analysis"):
        input_vector = np.array(list(user_inputs.values())).reshape(1, -1)
        scaled_input = scaler.transform(input_vector)
        prediction = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)
        
        st.subheader("Diagnostic Results")
        
        # Modified probability display
        prob_df = pd.DataFrame({
            'Diagnosis': ['Cancerous', 'Non-Cancerous'],
            'Probability': probabilities[0]
        })
        
        fig = px.bar(
            prob_df,
            x='Diagnosis',
            y='Probability',
            color='Diagnosis',
            color_discrete_map={'Cancerous': 'red', 'Non-Cancerous': 'green'}
        )
        
        if prediction[0] == 0:
            st.error("📊 Analysis Result: Likely Cancerous")
        else:
            st.success("📊 Analysis Result: Likely Non-Cancerous")
            
        st.plotly_chart(fig)
        
        # Modified confidence levels
        st.markdown("### Analysis Confidence")
        st.info(f"Likelihood of being Non-Cancerous: {probabilities[0][1]:.1%}")
        st.warning(f"Likelihood of being Cancerous: {probabilities[0][0]:.1%}")