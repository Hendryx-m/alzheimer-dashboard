import streamlit as st
import pandas as pd
import plotly.express as px

# Loading the longitudinal dataset
df = pd.read_csv("longitudinal_data.csv")
st.title("Longitudinal MMSE Trends Dashboard")

# Sidebar: select a PatientID to view their data
patient_ids = sorted(df["PatientID"].unique())
selected_patient = st.sidebar.selectbox("Select PatientID", patient_ids)

# Filtering data for the selected patinet
patient_data = df[df["PatientID"]== selected_patient]

st.subheader(f"Patient {selected_patient} MMSE Over Time")
fig = px.line(patient_data, x="Visit", y="MMSE", markers=True,
              title=f"MMSE Trend for Patient {selected_patient}",
              labels={"Visit": "Visit (Time)", "MMSE": "MMSE Score"})
st.plotly_chart(fig)

# Additional visualization: Group level average MMSE over Visits
avg_mmse = df.groupby("Visit")["MMSE"].mean().reset_index()
st.subheader("Average MMSE Over Visits")
fig2 = px.line(avg_mmse, x="Visit", y="MMSE", markers=True,
               title="Average MMSE Trend Across All Patinets")
st.plotly_chart(fig2)