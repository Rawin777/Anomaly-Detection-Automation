import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Module D: Dark-Mode Streamlit Deployment
st.set_page_config(page_title="ISRO QA Inspector", layout="wide")
st.title("🛰️ QA Inspector Dashboard (ISRO Latent Anomaly Detection)")
st.markdown("**Modules C & D Interface**: Batch processing, distribution curves, and traffic-light explainability.")

# 1. Generate Mock Test Lot Data (Simulating a CSV Upload)
np.random.seed(42)
components = [f"COMP-{i:03d}" for i in range(1, 101)]
cryo_pressure = np.random.normal(loc=50.0, scale=2.0, size=100)

# Introduce two latent anomalies (drift)
cryo_pressure[88] = 62.1 
cryo_pressure[42] = 37.5
df = pd.DataFrame({"Component_ID": components, "Cryo_Pressure_psi": cryo_pressure})

# 2. Module A/B Math: Dynamic Part Average Testing (DPAT)
q1 = df["Cryo_Pressure_psi"].quantile(0.25)
q3 = df["Cryo_Pressure_psi"].quantile(0.75)
iqr = q3 - q1
robust_std = iqr * 0.7413
median = df["Cryo_Pressure_psi"].median()

# Define boundaries (Median +/- 6 * Robust SD is a standard aerospace tolerance)
upper_bound = median + (6 * robust_std)
lower_bound = median - (6 * robust_std)
df["Status"] = np.where((df["Cryo_Pressure_psi"] > upper_bound) | (df["Cryo_Pressure_psi"] < lower_bound), "Anomaly", "Pass")

# 3. Module D: Lot Distribution Curve
st.subheader("Lot Distribution Curve")
fig = go.Figure()

# Plot Normal Components
fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Pass"].index, 
    y=df[df["Status"]=="Pass"]["Cryo_Pressure_psi"],
    mode='markers', name='Pass', marker=dict(color='#00CC96', size=8)
))

# Plot Anomalies
fig.add_trace(go.Scatter(
    x=df[df["Status"]=="Anomaly"].index, 
    y=df[df["Status"]=="Anomaly"]["Cryo_Pressure_psi"],
    mode='markers', name='Anomaly Flagged', marker=dict(color='#EF553B', size=12, symbol='x')
))

# Plot DPAT Boundaries
fig.add_hline(y=upper_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Upper Bound")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="#FFA15A", annotation_text="DPAT Lower Bound")
fig.update_layout(template="plotly_dark", xaxis_title="Component Index", yaxis_title="Cryo-Pressure Telemetry (psi)")
st.plotly_chart(fig, use_container_width=True)

# 4. Module C: Inspector Explainability (Traffic Light Score)
st.subheader("QA Inspector Diagnostics")
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
    st.write("**Physics-Based Parameter Deviation:**")
    st.write(f"Deviation from Median: {abs(comp_data['Cryo_Pressure_psi'] - median):.2f} psi")
    st.caption("Robust Standard Deviation limit exceeded." if comp_data["Status"] == "Anomaly" else "Within normal DPAT bounds.")

# pip install streamlit pandas numpy plotly
# streamlit run backend\app.py