**SignalForge**

#**AI-Driven Anomaly Detection in Component Burn-in and Screening**

The main idea of our project is to find components that may show abnormal behaviour during **Environmental Stress Screening(ESS)**,even if they are still within the normal fixed limits.

#**Problem Statement**

In traditional ESS,the measured value of a component is compared with a fixed limit.

If the value is within the limit - **Pass**

If the value crosses the limit - **Fail**

But sometimes a component can pass the fixed limit and still behave differently from other components in the same lot.

Our project tries to identify this type of abnormal behaviour at an early stage.

#**Proposed Solution**

We use statistics and machine learning to analyse ESS data.

The system :

1.Processes ESS measurement data.

2.Creates a dynamic baseline for normal behaviour.

3.Finds abnormal component behaviour.

4.Uses early measurements like **0h and 24h** to predict the behaviour at **168h**.

5.Combines these results to give a risk level.

The final output can be:

-Normal

-Review

-High Risk

6.Provides an explainable result.

#**WorkFlow**

Component Data - Preprocessing - Dynamic Baseline - Anomaly Detection + Drift Prediction - Risk Engine - Normal/Review/High Risk

#**Parameters**

The system works with ESS parameters such as:

-Iddq

-Leakage

-Delay

Measurements are considered at different time points:

-O hours

-24 hours

-96 hours

-168 hours

#**Technologies Used**

-Python

-Pandas

-Numpy

-Scipy

-Statsmodels

-Scikit-learn

-Matplotlib/Plotly

-Streamlit

-FastAPI

#**Key Features**

1.Dynamic Baseline

-Instead of only using fixed limits,we try to understand what normal behaviour looks like for a particular lot.

2.Anomaly Detection

-We compare an individual component with the behaviour of the lot and identify unusual patterns.

3.168h Drift Prediciton

-We use early measurements,mainly **oh and 24h**,to estimate how the component may behave **168h**.

4.Risk Score

-Anomaly and drift information are combined to generate a risk decision.

5.Explanation

-The system provides the reason for the risk,such as abnormal deviation or excessive grift.

6.Human-in-the-Loop QA Support

-The system supports QA Engineers by providing the risk level and supporting evidence.The final decision is reviewed by a human instead of being completely automated.

#**Expected Impact**

Our project aims to:

-Identify abnormal component behaviour earlier.

-Give an early warning by predicting 168h drift using early measurements.

-Reduce the risk of defective components escaping screening process.

-Help QA Engineers make more decision.

-Provides an explainable risk decision instead of simple PASS/FAIL result.

#**Team**

**Prototype:** https://q9kbbqptxac9nhhaapnp6z.streamlit.app/

**Team Name**: SignalForge

**Problem StatementID**: SIH26170

**Theme**: Smart Automation

**PS Category**: Software
