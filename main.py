import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="FinVue | AI Loan Approval System", page_icon="💰", layout="wide")

# --- CUSTOM CSS FOR HIGH-END UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004aad; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('clientes.csv')
    # Preprocessing
    df['renda_conjuge'] = pd.to_numeric(df['renda_conjuge'], errors='coerce').fillna(0)
    df['sexo'] = df['sexo'].fillna(df['sexo'].mode()[0])
    df['estado_civil'] = df['estado_civil'].fillna(df['estado_civil'].mode()[0])
    df['dependentes'] = df['dependentes'].fillna(df['dependentes'].mode()[0])
    df['empregado'] = df['empregado'].fillna(df['empregado'].mode()[0])
    df['emprestimo'] = df['emprestimo'].fillna(df['emprestimo'].median())
    df['prestacao_mensal'] = df['prestacao_mensal'].fillna(df['prestacao_mensal'].mode()[0])
    df['historico_credito'] = df['historico_credito'].fillna(df['historico_credito'].mode()[0])
    return df

@st.cache_resource
def train_model(df):
    data = df.copy()
    le = LabelEncoder()
    cat_cols = ['sexo', 'estado_civil', 'educacao', 'empregado', 'imovel', 'aprovacao_emprestimo', 'dependentes']
    mappings = {}
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    X = data.drop(columns=['cod_cliente', 'aprovacao_emprestimo'])
    y = data['aprovacao_emprestimo']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, mappings

# --- LOAD DATA ---
df = load_and_clean_data()
model, mappings = train_model(df)

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("FinVue Intelligence")
page = st.sidebar.radio("Navigate System", ["Executive Dashboard", "Loan Predictor", "Portfolio Analytics", "Raw Data Explorer"])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "Executive Dashboard":
    st.title("📊 Executive Credit Dashboard")
    st.markdown("Real-time portfolio health and credit distribution overview.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applications", len(df))
    m2.metric("Approval Rate", f"{(df['aprovacao_emprestimo'] == 'Y').mean():.1%}")
    m3.metric("Avg Loan Amount", f"${df['emprestimo'].mean():.2f}K")
    m4.metric("Avg Applicant Income", f"${df['renda'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="renda", color="aprovacao_emprestimo", title="Income vs Approval Distribution", nbins=50, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(df, names='imovel', title='Property Area Distribution', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: LOAN PREDICTOR ---
elif page == "Loan Predictor":
    st.title("🤖 AI Risk Assessment Engine")
    st.info("Input client parameters below to generate a real-time credit risk score.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dep = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        with c2:
            edu = st.selectbox("Education", ["Graduate", "Not Graduate"])
            emp = st.selectbox("Self Employed", ["Yes", "No"])
            prop = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        with c3:
            cred = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x==1 else "Poor")
            term = st.slider("Loan Term (Months)", 12, 480, 360)
        
        c4, c5, c6 = st.columns(3)
        income = c4.number_input("Applicant Income ($)", min_value=0, value=5000)
        co_income = c5.number_input("Co-applicant Income ($)", min_value=0, value=0)
        loan_amt = c6.number_input("Loan Amount (Thousands)", min_value=0, value=150)
        
        submit = st.form_submit_button("GENERATE DECISION")

    if submit:
        # Prepare Input
        input_data = pd.DataFrame([[
            mappings['sexo'][gender], mappings['estado_civil'][married], mappings['dependentes'][dep],
            mappings['educacao'][edu], mappings['empregado'][emp], income, co_income, loan_amt,
            term, cred, mappings['imovel'][prop]
        ]], columns=df.drop(columns=['cod_cliente', 'aprovacao_emprestimo']).columns)
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.subheader("Decision Results")
        res_col1, res_col2 = st.columns([1, 2])
        
        if prediction == 1:
            res_col1.success("✅ LOAN APPROVED")
            res_col2.metric("Approval Confidence", f"{prob:.1%}")
        else:
            res_col1.error("❌ LOAN REJECTED")
            res_col2.metric("Approval Confidence", f"{prob:.1%}")
            
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Credit Score Probability"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "darkblue"},
                     'steps': [
                         {'range': [0, 50], 'color': "red"},
                         {'range': [50, 75], 'color': "yellow"},
                         {'range': [75, 100], 'color': "green"}]}
        ))
        st.plotly_chart(fig_gauge)

# --- PAGE 3: PORTFOLIO ANALYTICS ---
elif page == "Portfolio Analytics":
    st.title("📈 Advanced Analytics")
    st.markdown("Deeper insights into credit correlations and feature importance.")
    
    feat_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['cod_cliente', 'aprovacao_emprestimo']).columns)
    fig_feat = px.bar(feat_importances.sort_values(), orientation='h', title="Key Drivers for Loan Approval")
    st.plotly_chart(fig_feat, use_container_width=True)
    
    fig_box = px.box(df, x="educacao", y="renda", color="aprovacao_emprestimo", title="Income Distribution by Education and Status")
    st.plotly_chart(fig_box, use_container_width=True)

# --- PAGE 4: RAW DATA ---
elif page == "Raw Data Explorer":
    st.title("📁 Data Management")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Processed Data", df.to_csv(index=False), "processed_loans.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("Made by Muhammad Ali Kahoot")
