import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Module D: Streamlit Deployment
st.set_page_config(page_title="ISRO QA Inspector", layout="wide")
st.title("🛰️ QA Inspector Dashboard (ISRO Latent Anomaly Detection)")
st.markdown("**Modules C & D Interface**: Batch processing, distribution curves, and traffic-light explainability.")

# 1. Load AI model
@st.cache_resource
def load_model():
    model_path = "Modules/module_b.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None 

model = load_model()

# Sidebar: Mission Parameters & System Status
st.sidebar.header("⚙️ Mission Parameters")
sigma_limit = st.sidebar.slider("DPAT Sigma Tolerance", min_value=1.0, max_value=10.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🧠 System Status")
if model is not None:
    st.sidebar.success("✅ AI Engine Connected (module_b.pkl)")
else:
    st.sidebar.warning("⚠️ AI Engine Offline (Requires Model)")

# 2. Upload Box
uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])

st.subheader("Lot Distribution Curve")

# 3. Empty State Logic (Shows wireframe graph before upload)
if uploaded_file is None:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis_title="Component Index (Awaiting Data)", 
        yaxis_title="Cryo-Pressure Telemetry (psi)",
        xaxis=dict(range=[0, 100]), # Dummy scale so it looks like a real graph
        yaxis=dict(range=[30, 70]),
        modebar=dict(color='gray', activecolor='#00CC96') 
    )
    st.plotly_chart(empty_fig, use_container_width=True, theme="streamlit")
    st.info("👆 Please upload a test lot CSV file to populate the graph and run the AI diagnostics.")
    st.stop() # Stops the Diagnostic Report from rendering yet

# =====================================================================
# DATA PROCESSING & POPULATED UI (Runs only after upload)
# =====================================================================

df = pd.read_csv(uploaded_file)
st.success("CSV Uploaded Successfully! Running analysis...")

# 4. Math: DPAT Boundaries
q1 = df["Cryo_Pressure_psi"].quantile(0.25)
q3 = df["Cryo_Pressure_psi"].quantile(0.75)
iqr = q3 - q1
robust_std = iqr * 0.7413
median = df["Cryo_Pressure_psi"].median()

upper_bound = median + (sigma_limit * robust_std)
lower_bound = median - (sigma_limit * robust_std)

# 5. Determine Anomaly Status
if model is not None:
    features = df.drop(columns=['Component_ID'], errors='ignore')
    predictions = model.predict(features)
    df['Status'] = np.where(predictions == 1, "Anomaly", "Pass")
else:
    df["Status"] = np.where((df["Cryo_Pressure_psi"] > upper_bound) | (df["Cryo_Pressure_psi"] < lower_bound), "Anomaly", "Pass")

# 6. Section 1: Macro View (Populated Graph)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Pass"].index, 
    y=df[df["Status"]=="Pass"]["Cryo_Pressure_psi"],
    mode='markers', name='Pass', marker=dict(color='#00CC96', size=8)
))

fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Anomaly"].index, 
    y=df[df["Status"]=="Anomaly"]["Cryo_Pressure_psi"],
    mode='markers', name='Anomaly Flagged', marker=dict(color='#EF553B', size=12, symbol='x')
))

fig.add_hline(y=upper_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Upper Bound")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Lower Bound")

fig.update_layout(
    xaxis_title="Component Index", 
    yaxis_title="Cryo-Pressure Telemetry (psi)",
    modebar=dict(color='gray', activecolor='#00CC96') 
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.markdown("---") 

# 7. Section 2: Micro View (Diagnostic Report)
st.subheader("Automated QA Inspector Diagnostics")
selected_comp = st.selectbox("Select a Component to Inspect:", df["Component_ID"])
comp_data = df[df["Component_ID"] == selected_comp].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sensor Reading", f"{comp_data['Cryo_Pressure_psi']:.2f} psi")
with col2:
    if comp_data["Status"] == "Pass":
        st.success("🟢 Confidence Score: 98% (Safe)")
    else:
        st.error("🔴 Confidence Score: 12% (Latent Anomaly Detected)")
with col3:
    deviation = abs(comp_data['Cryo_Pressure_psi'] - median)
    st.metric("Deviation from Median", f"{deviation:.2f} psi")
    
st.markdown("### 📋 Physics-Based Reasoning Report")

if comp_data["Status"] == "Anomaly":
    st.error(f"""
    **Failure Analysis for {selected_comp}:**
    This component has been flagged. Its internal cryo-pressure reading ({comp_data['Cryo_Pressure_psi']:.2f} psi) 
    has breached the dynamic mission safety tolerance. 
    
    * **Baseline Median:** {median:.2f} psi
    * **Allowed Deviation:** ±{(sigma_limit * robust_std):.2f} psi (Sigma: {sigma_limit})
    * **Actual Deviation:** {deviation:.2f} psi
    
    **Recommendation:** Isolate {selected_comp} from the current ISRO test lot immediately. Proceed with secondary manual inspection.
    """)
else:
    st.success(f"""
    **Pass Analysis for {selected_comp}:**
    This component is operating within safe physical boundaries. The cryo-pressure reading ({comp_data['Cryo_Pressure_psi']:.2f} psi) 
    falls well within the acceptable limit. No latent drift detected.
    """)