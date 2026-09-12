import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Module D: Streamlit Deployment
st.set_page_config(page_title="ISRO QA Inspector", layout="wide")
st.title("🛰️ QA Inspector Dashboard (ISRO Latent Anomaly Detection)")
st.markdown("**Modules C & D Interface**: Batch processing, distribution curves, and traffic-light explainability.")

# Sidebar: Mission Parameters & System Status
st.sidebar.header("⚙️ Mission Parameters")
# We map the teammate's hardcoded "4" threshold to a slider for the pitch
mad_threshold = st.sidebar.slider("MAD Anomaly Threshold", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🧠 System Status")
st.sidebar.success("✅ Module A Logic Integrated (MAD Engine Active)")

# 2. Upload Box
uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])

st.subheader("Lot Distribution Curve")

# 3. Empty State Logic
if uploaded_file is None:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis_title="Component Index (Awaiting Data)", 
        yaxis_title="Telemetry Metric",
        xaxis=dict(range=[0, 10]), 
        yaxis=dict(range=[5, 20]),
        modebar=dict(color='gray', activecolor='#00CC96') 
    )
    st.plotly_chart(empty_fig, use_container_width=True, theme="streamlit")
    st.info("👆 Please upload a test lot CSV file to populate the graph and run the diagnostics.")
    st.stop() 

# =====================================================================
# DATA PROCESSING (Module A Logic)
# =====================================================================

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip() 

st.success("CSV Uploaded Successfully!")

# Dynamic Column Selection UI
st.markdown("### 📊 Map Telemetry Data")
col_map1, col_map2 = st.columns(2)
with col_map1:
    id_col = st.selectbox("Select Identifier Column:", df.columns, index=0)
with col_map2:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found in the CSV. Analysis requires numerical telemetry data.")
        st.stop()
    # Default to 'Value_0h' if it exists, otherwise use the first numeric column
    default_metric = "Value_0h" if "Value_0h" in numeric_cols else numeric_cols[0]
    metric_col = st.selectbox("Select Telemetry Metric:", numeric_cols, index=numeric_cols.index(default_metric))

st.markdown("---")

# 4. Math: Module A MAD Logic
# The teammate grouped by Lot_ID. If Lot_ID is missing from the uploaded CSV, we treat the whole file as one lot.
if "Lot_ID" in df.columns:
    df["Median"] = df.groupby("Lot_ID")[metric_col].transform("median")
else:
    df["Median"] = df[metric_col].median()

df["Deviations"] = abs(df[metric_col] - df["Median"])

if "Lot_ID" in df.columns:
    df["MAD"] = df.groupby("Lot_ID")["Deviations"].transform("median")
else:
    df["MAD"] = df["Deviations"].median()

df["Anomaly_Score"] = df["Deviations"] / (df["MAD"] + 1e-9)

# 5. Determine Anomaly Status based on Module A logic
df["Status"] = np.where(df["Anomaly_Score"] > mad_threshold, "Anomaly", "Pass")

# Calculate upper and lower bounds for the graph visually based on the MAD score
upper_bound = df["Median"].iloc[0] + (mad_threshold * df["MAD"].iloc[0])
lower_bound = df["Median"].iloc[0] - (mad_threshold * df["MAD"].iloc[0])


# 6. Section 1: Macro View (MAD Graph)
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

fig.add_hline(y=upper_bound, line_dash="dash", line_color="#FFA15A", annotation_text=f"MAD Upper Bound ({mad_threshold})")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="#FFA15A", annotation_text=f"MAD Lower Bound ({mad_threshold})")

fig.update_layout(
    title=f"Statistical MAD Analysis: {metric_col}",
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
        st.success("🟢 Status: Safe")
    else:
        st.error("🔴 Status: Latent Anomaly Detected")
with col3:
    st.metric("Anomaly Score (MAD multiplier)", f"{comp_data['Anomaly_Score']:.2f}")
    
st.markdown("### 📋 Physics-Based Reasoning Report")

if comp_data["Status"] == "Anomaly":
    st.error(f"""
    **Failure Analysis for {selected_comp}:**
    This component has been flagged. Its internal **{metric_col}** reading ({comp_data[metric_col]:.2f}) 
    has breached the dynamic mission safety tolerance based on Median Absolute Deviation (MAD).
    
    * **Lot Baseline Median:** {comp_data['Median']:.2f}
    * **Lot MAD:** {comp_data['MAD']:.2f}
    * **Calculated Anomaly Score:** {comp_data['Anomaly_Score']:.2f} (Threshold: > {mad_threshold})
    
    **Recommendation:** Isolate {selected_comp} from the current ISRO test lot immediately. Proceed with secondary manual inspection.
    """)
else:
    st.success(f"""
    **Pass Analysis for {selected_comp}:**
    This component is operating within safe physical boundaries. The **{metric_col}** reading ({comp_data[metric_col]:.2f}) 
    falls well within the acceptable limits (Anomaly Score: {comp_data['Anomaly_Score']:.2f}). No latent drift detected.
    """)