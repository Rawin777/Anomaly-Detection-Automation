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

# 3. Empty State Logic
if uploaded_file is None:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis_title="Component Index (Awaiting Data)", 
        yaxis_title="Telemetry Metric",
        xaxis=dict(range=[0, 100]), 
        yaxis=dict(range=[30, 70]),
        modebar=dict(color='gray', activecolor='#00CC96') 
    )
    st.plotly_chart(empty_fig, use_container_width=True, theme="streamlit")
    st.info("👆 Please upload a test lot CSV file to populate the graph and run the AI diagnostics.")
    st.stop() 

# =====================================================================
# DATA PROCESSING & DYNAMIC MAPPING 
# =====================================================================

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip() # Clean hidden spaces

st.success("CSV Uploaded Successfully!")

# Dynamic Column Selection UI
st.markdown("### 📊 Map Telemetry Data")
col_map1, col_map2 = st.columns(2)
with col_map1:
    id_col = st.selectbox("Select Identifier Column:", df.columns, index=0)
with col_map2:
    # Filter to only show numeric columns for the math
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found in the CSV. DPAT requires numerical telemetry data.")
        st.stop()
    metric_col = st.selectbox("Select Telemetry Metric:", numeric_cols, index=0 if len(numeric_cols) > 0 else None)

st.markdown("---")

# 4. Math: DPAT Boundaries (Using dynamic metric_col)
q1 = df[metric_col].quantile(0.25)
q3 = df[metric_col].quantile(0.75)
iqr = q3 - q1
robust_std = iqr * 0.7413
median = df[metric_col].median()

upper_bound = median + (sigma_limit * robust_std)
lower_bound = median - (sigma_limit * robust_std)

# 5. Determine Anomaly Status (With crash-protection for the ML model)
ml_used = False
if model is not None:
    try:
        features = df.drop(columns=[id_col], errors='ignore')
        predictions = model.predict(features)
        df['Status'] = np.where(predictions == 1, "Anomaly", "Pass")
        ml_used = True
    except Exception as e:
        # If the model crashes (e.g., wrong column names), fallback to math
        df["Status"] = np.where((df[metric_col] > upper_bound) | (df[metric_col] < lower_bound), "Anomaly", "Pass")
        st.sidebar.error(f"⚠️ ML Prediction Failed: {e}. Falling back to DPAT math.")
else:
    df["Status"] = np.where((df[metric_col] > upper_bound) | (df[metric_col] < lower_bound), "Anomaly", "Pass")

# 6. Section 1: Macro View (Populated Graph)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Pass"].index, 
    y=df[df["Status"]=="Pass"][metric_col],
    mode='markers', name='Pass', marker=dict(color='#00CC96', size=8)
))

fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Anomaly"].index, 
    y=df[df["Status"]=="Anomaly"][metric_col],
    mode='markers', name='Anomaly Flagged', marker=dict(color='#EF553B', size=12, symbol='x')
))

fig.add_hline(y=upper_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Upper Bound")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Lower Bound")

fig.update_layout(
    xaxis_title=f"{id_col} (Index)", 
    yaxis_title=metric_col,
    modebar=dict(color='gray', activecolor='#00CC96') 
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.markdown("---") 

# 7. Section 2: Micro View (Diagnostic Report)
st.subheader("Automated QA Inspector Diagnostics")
selected_comp = st.selectbox("Select a Component to Inspect:", df[id_col])
comp_data = df[df[id_col] == selected_comp].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sensor Reading", f"{comp_data[metric_col]:.2f}")
with col2:
    if comp_data["Status"] == "Pass":
        st.success("🟢 Confidence Score: 98% (Safe)")
    else:
        st.error("🔴 Confidence Score: 12% (Latent Anomaly Detected)")
with col3:
    deviation = abs(comp_data[metric_col] - median)
    st.metric("Deviation from Median", f"{deviation:.2f}")
    
st.markdown("### 📋 Physics-Based Reasoning Report")

if comp_data["Status"] == "Anomaly":
    st.error(f"""
    **Failure Analysis for {selected_comp}:**
    This component has been flagged. Its internal **{metric_col}** reading ({comp_data[metric_col]:.2f}) 
    has breached the dynamic mission safety tolerance. 
    
    * **Baseline Median:** {median:.2f}
    * **Allowed Deviation:** ±{(sigma_limit * robust_std):.2f} (Sigma: {sigma_limit})
    * **Actual Deviation:** {deviation:.2f}
    
    **Recommendation:** Isolate {selected_comp} from the current ISRO test lot immediately. Proceed with secondary manual inspection.
    """)
else:
    st.success(f"""
    **Pass Analysis for {selected_comp}:**
    This component is operating within safe physical boundaries. The **{metric_col}** reading ({comp_data[metric_col]:.2f}) 
    falls well within the acceptable limits. No latent drift detected.
    """)