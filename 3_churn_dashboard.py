import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import sqlite3
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        with open('models/xgb_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error(
            "❌ Model files not found! Please run 2_churn_model_training.py first.")
        st.stop()


@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('database/churn_analysis.db')
        df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
        return df
    except:
        st.error("❌ Database not found! Please run 1_load_data_to_sql.py first.")
        st.stop()


st.markdown('<p class="main-header">🎯 Customer Churn Prediction Dashboard</p>',
            unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["📈 Overview", "🔮 Prediction", "📊 Analytics", "ℹ️ About"]
)

df = load_data()

if page == "📈 Overview":
    st.header("📈 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        churn_count = (df['Churn'] == 'Yes').sum()
        st.metric("Churned Customers", f"{churn_count:,}")

    with col3:
        churn_rate = (churn_count / len(df)) * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%",
                  delta=f"{churn_rate - 26.5:.2f}%",
                  delta_color="inverse")

    with col4:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_dist = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_dist.values,
            names=['Retained', 'Churned'],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract Type")
        contract_churn = pd.crosstab(
            df['Contract'], df['Churn'], normalize='index') * 100
        fig = px.bar(
            contract_churn,
            barmode='group',
            labels={'value': 'Percentage (%)', 'Contract': 'Contract Type'},
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_layout(legend_title_text='Churn Status')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Charges Distribution")
        fig = px.histogram(
            df, x='MonthlyCharges', color='Churn',
            marginal='box', nbins=30,
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
            labels={'MonthlyCharges': 'Monthly Charges ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tenure Distribution")
        fig = px.histogram(
            df, x='tenure', color='Churn',
            marginal='box', nbins=30,
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
            labels={'tenure': 'Tenure (months)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **Contract Insights:**
        - Month-to-month: Highest churn
        - Two year: Lowest churn
        - Recommendation: Promote long-term contracts
        """)

    with col2:
        avg_charges_churn = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
        avg_charges_stay = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
        st.info(f"""
        **Pricing Insights:**
        - Churned avg: ${avg_charges_churn:.2f}
        - Retained avg: ${avg_charges_stay:.2f}
        - Difference: ${avg_charges_churn - avg_charges_stay:.2f}
        """)

    with col3:
        avg_tenure_churn = df[df['Churn'] == 'Yes']['tenure'].mean()
        avg_tenure_stay = df[df['Churn'] == 'No']['tenure'].mean()
        st.info(f"""
        **Tenure Insights:**
        - Churned avg: {avg_tenure_churn:.1f} months
        - Retained avg: {avg_tenure_stay:.1f} months
        - Early months are critical
        """)

elif page == "🔮 Prediction":
    st.header("🔮 Predict Customer Churn")

    model, scaler, feature_cols = load_models()

    st.subheader("Enter Customer Details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Additional Services**")
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, 1000.0, step=100.0)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        categorical_cols = input_data.select_dtypes(include=['object']).columns
        input_encoded = pd.get_dummies(
            input_data, columns=categorical_cols, drop_first=True)

        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[feature_cols]

        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK - Customer Likely to Churn")
                st.markdown("""
                **Recommended Actions:**
                - Immediate outreach by retention team
                - Offer loyalty discount or upgrade
                - Schedule satisfaction survey
                - Consider contract incentives
                """)
            else:
                st.success("### ✅ LOW RISK - Customer Likely to Stay")
                st.markdown("""
                **Recommended Actions:**
                - Continue standard engagement
                - Monitor satisfaction metrics
                - Cross-sell opportunities available
                - Maintain service quality
                """)

        with col2:
            st.metric("Churn Probability", f"{probability[1]*100:.1f}%")

        with col3:
            risk_level = "HIGH" if probability[1] > 0.7 else "MEDIUM" if probability[1] > 0.3 else "LOW"
            risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            st.metric("Risk Level", f"{risk_color} {risk_level}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability[1]*100,
            title={'text': "Churn Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Analytics":
    st.header("📊 Model Performance Analytics")

    if not os.path.exists('outputs/churn_predictions.csv'):
        st.error(
            "❌ Prediction files not found! Please run 2_churn_model_training.py first.")
        st.stop()

    predictions = pd.read_csv('outputs/churn_predictions.csv')
    feature_importance = pd.read_csv('outputs/churn_feature_importance.csv')

    st.subheader("🎯 Model Performance Metrics")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        acc = accuracy_score(
            predictions['Actual'], predictions['Predicted_XGB'])
        st.metric("Accuracy", f"{acc:.2%}")

    with col2:
        prec = precision_score(
            predictions['Actual'], predictions['Predicted_XGB'])
        st.metric("Precision", f"{prec:.2%}")

    with col3:
        rec = recall_score(predictions['Actual'], predictions['Predicted_XGB'])
        st.metric("Recall", f"{rec:.2%}")

    with col4:
        f1 = f1_score(predictions['Actual'], predictions['Predicted_XGB'])
        st.metric("F1 Score", f"{f1:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Important Features")
        fig = px.bar(
            feature_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(
            predictions['Actual'], predictions['Predicted_XGB'])
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Probability Distribution")
    fig = px.histogram(
        predictions,
        x='Probability_XGB',
        color='Actual',
        nbins=30,
        marginal='box',
        labels={'Probability_XGB': 'Churn Probability',
                'Actual': 'Actual Churn'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔴 High-Risk Predictions")
    high_risk = predictions[predictions['Probability_XGB'] > 0.7].sort_values(
        'Probability_XGB', ascending=False)
    st.dataframe(high_risk.head(20), use_container_width=True)

elif page == "ℹ️ About":
    st.header("ℹ️ About This Dashboard")

    st.markdown("""
    ## Customer Churn Prediction System
    
    ### 🎯 Purpose
    This dashboard helps identify customers at risk of churning using machine learning models.
    
    ### 🤖 Models Used
    - **Random Forest Classifier**: Ensemble learning method
    - **XGBoost Classifier**: Gradient boosting algorithm (primary model)
    
    ### 📊 Features
    1. **Overview**: Explore customer data and churn patterns
    2. **Prediction**: Predict churn risk for individual customers
    3. **Analytics**: View model performance and feature importance
    
    ### 🔧 Technology Stack
    - **Backend**: Python, Scikit-learn, XGBoost
    - **Frontend**: Streamlit
    - **Database**: SQLite
    - **Visualization**: Plotly, Seaborn, Matplotlib
    
    ### 📈 Model Performance
    The XGBoost model achieves:
    - Accuracy: ~80%
    - ROC-AUC: ~0.85
    - Precision: ~65%
    - Recall: ~55%
    
    ### 👨‍💻 Developer
    Created as part of a Data Analytics portfolio project.
    
    ### 📧 Contact
    For questions or feedback, please reach out via GitHub.
    
    ---
    **Version**: 1.0  
    **Last Updated**: October 2025
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Tips:**
- Use Overview to understand patterns
- Use Prediction for individual assessments
- Check Analytics for model insights
""")
