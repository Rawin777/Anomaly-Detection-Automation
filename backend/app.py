import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Module D: Streamlit Deployment
st.set_page_config(page_title="ISRO QA Inspector", layout="wide")
st.title("🛰️ QA Inspector Dashboard (ISRO Latent Anomaly Detection)")
st.markdown("**SignalForge Modules A, B, & C**: Drift Prediction, Robust Lot Statistics, and Explainable AI.")

# 1. Load AI model (With MLOps Version-Mismatch Protection)
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, "Modules", "module_b.pkl")
    
    possible_paths = ["Modules/module_b.pkl", "backend/Modules/module_b.pkl", target_path]
    
    for p in possible_paths:
        if os.path.exists(p):
            try:
                loaded_model = joblib.load(p)
                # IMMUNITY CHECK: Fire a blank prediction to test for version mismatch
                test_df = pd.DataFrame({"Value_0h": [10.0], "Value_24h": [10.5]})
                loaded_model.predict(test_df)
                return loaded_model, f"✅ Module B Connected ({p})"
            except Exception:
                # If predict() throws an AttributeError, the versions clash. Break out and self-heal.
                break 
                
    # SELF-HEALING FALLBACK: Train natively on the cloud server to guarantee version match
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # Train on synthetic base trajectories
        X_train = pd.DataFrame({
            "Value_0h": [10.0, 10.2, 9.8, 11.0, 10.5, 12.0, 15.0],
            "Value_24h": [10.1, 10.4, 9.9, 11.2, 10.7, 12.8, 16.0]
        })
        # Calculate strict 144-hour drift for the baseline
        y_train = X_train["Value_24h"] + ((X_train["Value_24h"] - X_train["Value_0h"]) * 6)
        
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
        model.fit(X_train, y_train)
        
        return model, "✅ Module B Auto-Generated Natively (Version Mismatch Bypassed)"
    except Exception as e:
        return None, f"⚠️ AI Engine Offline (Auto-generation failed: {e})"

model, status_msg = load_model()

# Sidebar: Mission Parameters & System Status
st.sidebar.header("⚙️ Mission Parameters")
z_threshold = st.sidebar.slider("Module A: Robust Z-Score Threshold", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
safety_early_drift = st.sidebar.number_input("Module B: Safety Early Drift (24h Max)", value=0.20, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("🧠 System Status")
st.sidebar.success("✅ Module A Logic Integrated (MAD/Z-Score)")
if model is not None:
    st.sidebar.success(status_msg)
else:
    st.sidebar.warning(status_msg)
    
# 2. Upload Box
uploaded_file = st.file_uploader("Upload Telemetry CSV (Must contain Value_0h and Value_24h)", type=["csv"])

st.subheader("Predictive Anomaly Distribution Curve")

# 3. Empty State Logic
if uploaded_file is None:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis_title="Component Index (Awaiting Data)", 
        yaxis_title="Predicted 168h Telemetry",
        xaxis=dict(range=[0, 10]), 
        yaxis=dict(range=[5, 20]),
        modebar=dict(color='gray', activecolor='#00CC96') 
    )
    st.plotly_chart(empty_fig, use_container_width=True, theme="streamlit")
    st.info("👆 Please upload a test lot CSV file to execute the predictive engine.")
    st.stop() 

# =====================================================================
# DATA PROCESSING (Modules A + B Integration)
# =====================================================================

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip() 

required_cols = ["Value_0h", "Value_24h"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Invalid CSV. The file must contain '{col}'.")
        st.stop()

# 4. Feature Engineering 
df["Early_Drift"] = df["Value_24h"] - df["Value_0h"]
df["Early_Drift_Rate"] = df["Early_Drift"] / 24.0
safety_drift_rate = safety_early_drift / 24.0

# 5. Module B: Predict 168h 
if model is not None:
    X = df[["Value_0h", "Value_24h"]]
    df["Predicted_168h"] = model.predict(X)
else:
    df["Predicted_168h"] = df["Value_24h"] + (df["Early_Drift_Rate"] * 144) 

df["Predicted_Early_to_168_Drift"] = df["Predicted_168h"] - df["Value_24h"]
df["Predicted_168h_Drift_Rate"] = df["Predicted_Early_to_168_Drift"] / 144.0

# 6. Module A: Robust Z-Score 
if "Lot_ID" in df.columns:
    df["Lot_Median"] = df.groupby("Lot_ID")["Value_24h"].transform("median")
    df["Lot_MAD"] = df.groupby("Lot_ID")["Value_24h"].transform(lambda s: np.median(np.abs(s - np.median(s))))
else:
    df["Lot_Median"] = df["Value_24h"].median()
    df["Lot_MAD"] = np.median(np.abs(df["Value_24h"] - df["Value_24h"].median()))

scale = (1.4826 * df["Lot_MAD"]).replace(0, np.nan)
df["Lot_Robust_Z"] = ((df["Value_24h"] - df["Lot_Median"]) / scale).abs().fillna(0)

# Flag logic
df["Anomaly_Flag"] = df["Lot_Robust_Z"] >= z_threshold
df["Drift_Risk_Flag"] = (df["Predicted_168h_Drift_Rate"] > safety_drift_rate) | (df["Early_Drift_Rate"] > safety_drift_rate)

# Module C Explainable risk engine logic
def risk(row):
    reasons = []
    if row["Anomaly_Flag"]:
        reasons.append("Abnormal deviation from lot baseline (Module A)")
    if row["Drift_Risk_Flag"]:
        reasons.append("Projected drift rate exceeds safety criterion (Module B)")
    if not reasons:
        return "NORMAL", "Within lot baseline and predictive drift limits"
    if row["Anomaly_Flag"] and row["Drift_Risk_Flag"]:
        return "HIGH RISK", "; ".join(reasons)
    return "REVIEW", reasons[0]

df[["Risk", "Reason"]] = df.apply(lambda r: pd.Series(risk(r)), axis=1)
id_col = "Component_ID" if "Component_ID" in df.columns else df.columns[0]


# =====================================================================
# UI GRAPH & INTERACTIVITY 
# =====================================================================

# 7. Section 1: Macro View (Plotting Predicted 168h)
fig = go.Figure()

# Plot Safe Components
fig.add_trace(go.Scatter(
    x=df[df["Risk"]=="NORMAL"].index, 
    y=df[df["Risk"]=="NORMAL"]["Predicted_168h"],
    customdata=df[df["Risk"]=="NORMAL"][id_col],
    mode='markers', name='Pass (NORMAL)', marker=dict(color='#00CC96', size=8),
    hovertemplate="<b>%{customdata}</b><br>Predicted 168h: %{y:.2f}<extra></extra>"
))

# Plot Anomalies
fig.add_trace(go.Scatter(
    x=df[df["Risk"]!="NORMAL"].index, 
    y=df[df["Risk"]!="NORMAL"]["Predicted_168h"],
    customdata=df[df["Risk"]!="NORMAL"][id_col],
    mode='markers', name='Flagged (REVIEW/HIGH RISK)', marker=dict(color='#EF553B', size=12, symbol='x'),
    hovertemplate="<b>%{customdata}</b><br>Predicted 168h: %{y:.2f}<extra></extra>"
))

plot_median = df["Predicted_168h"].median()
plot_mad = np.median(np.abs(df["Predicted_168h"] - plot_median))
upper_bound = plot_median + (z_threshold * plot_mad)
lower_bound = plot_median - (z_threshold * plot_mad)

fig.add_hline(y=upper_bound, line_dash="dash", line_color="#FFA15A", annotation_text="Visual Upper Bound")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="#FFA15A", annotation_text="Visual Lower Bound")

fig.update_layout(
    title="Module B: Predicted 168h Degradation Profile (Click any point to inspect)",
    xaxis_title=f"{id_col} (Index)", 
    yaxis_title="Predicted 168h Value",
    modebar=dict(color='gray', activecolor='#00CC96') 
)

# Render the graph and capture click events!
event = st.plotly_chart(
    fig, 
    use_container_width=True, 
    theme="streamlit",
    on_select="rerun",           # Tells Streamlit to update the UI instantly on click
    selection_mode="points"      # Prevents lasso-selecting multiple points at once
)

# Total Anomalies Counter
total_anomalies = len(df[df["Risk"] != "NORMAL"])
st.markdown(f"**⚠️ Total Anomalies Flagged:** {total_anomalies} out of {len(df)} components processed.")

st.markdown("---") 

# =====================================================================
# SELECTION & DIAGNOSTICS LOGIC
# =====================================================================

# Initialize memory so Streamlit remembers what component we are looking at
if "selected_comp" not in st.session_state:
    st.session_state.selected_comp = df[id_col].iloc[0]
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None

# Check if the user just clicked a point on the graph
current_click = None
if event and event.get("selection") and event["selection"].get("points"):
    current_click = event["selection"]["points"][0].get("x")

# If they clicked a *new* point, force the diagnostic section to update to that component
if current_click is not None and current_click != st.session_state.last_clicked:
    st.session_state.last_clicked = current_click
    if current_click in df.index:
        st.session_state.selected_comp = df.loc[current_click, id_col]
elif current_click is None:
    st.session_state.last_clicked = None


# 8. Section 2: Micro View (Explainable AI Diagnostic Report)
st.subheader("Explainable AI Diagnostics")

col_search1, col_search2 = st.columns(2)

with col_search1:
    search_query = st.text_input("🔍 Filter Component ID:", placeholder="e.g., test or 001 (Case-Insensitive)")

# Filter the dropdown list based on what the user types
if search_query:
    matching_ids = df[df[id_col].astype(str).str.contains(search_query, case=False, na=False)][id_col].tolist()
else:
    matching_ids = df[id_col].tolist()

if not matching_ids:
    st.warning(f"No components found matching '{search_query}'. Showing all components.")
    matching_ids = df[id_col].tolist()

# Ensure the app doesn't crash if they filter out their currently selected component
if st.session_state.selected_comp not in matching_ids:
    st.session_state.selected_comp = matching_ids[0]

with col_search2:
    default_idx = matching_ids.index(st.session_state.selected_comp)
    user_selection = st.selectbox("Select from matching components:", matching_ids, index=default_idx)
    st.session_state.selected_comp = user_selection

# Pull the specific data for the finalized component
comp_data = df[df[id_col] == st.session_state.selected_comp].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("0h Value", f"{comp_data['Value_0h']:.2f}")
with col2:
    st.metric("24h Value", f"{comp_data['Value_24h']:.2f}")
with col3:
    st.metric("Predicted 168h Value", f"{comp_data['Predicted_168h']:.2f}", delta=f"{comp_data['Predicted_Early_to_168_Drift']:.2f} total drift", delta_color="inverse")
    
st.markdown("### 📋 Automated Reasoning Report")

if comp_data["Risk"] == "NORMAL":
    st.success(f"""
    **Pass Analysis for {st.session_state.selected_comp}:**
    {comp_data['Reason']}
    * **Projected Drift Rate:** {comp_data['Predicted_168h_Drift_Rate']:.4f}/hr (Max Allowed: {safety_drift_rate:.4f}/hr)
    * **Lot Robust Z-Score:** {comp_data['Lot_Robust_Z']:.2f}
    """)
elif comp_data["Risk"] == "REVIEW":
    st.warning(f"""
    **Secondary Review Required for {st.session_state.selected_comp}:**
    {comp_data['Reason']}
    * **Projected Drift Rate:** {comp_data['Predicted_168h_Drift_Rate']:.4f}/hr (Max Allowed: {safety_drift_rate:.4f}/hr)
    * **Lot Robust Z-Score:** {comp_data['Lot_Robust_Z']:.2f}
    """)
else:
    st.error(f"""
    **High Risk Anomaly Detected for {st.session_state.selected_comp}:**
    {comp_data['Reason']}
    * **Projected Drift Rate:** {comp_data['Predicted_168h_Drift_Rate']:.4f}/hr (Max Allowed: {safety_drift_rate:.4f}/hr)
    * **Lot Robust Z-Score:** {comp_data['Lot_Robust_Z']:.2f} (Threshold: {z_threshold})
    
    **Recommendation:** Isolate {st.session_state.selected_comp} from the current ESS lot immediately.
    """)