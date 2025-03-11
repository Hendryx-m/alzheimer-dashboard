# Alzheimer's Disease Synthetic Data & Dashboard

This repository contains a synthetic longitudinal dataset and an interactive dashboard designed to simulate and analyze factors associated with Alzheimer's disease. The project integrates advanced data simulation techniques, mixed effects modeling, and classification models with an interactive visualization built using Streamlit and Plotly.

## Overview

The project aims to:

- **Simulate Synthetic Data:**  
  Generate a synthetic dataset that includes patient demographics, lifestyle factors, clinical measurements, cognitive assessments, and additional environmental, socioeconomic, and treatment variables.

- **Longitudinal Data Simulation:**  
  Create multiple visits for each patient to simulate changes over time (e.g., cognitive decline measured by MMSE, clinical fluctuations, and changes in diagnosis).

- **Statistical Modeling:**  
  Implement a mixed effects model to analyze the longitudinal MMSE trends, and a classification model to predict the diagnosis at the final visit.

- **Interactive Dashboard:**  
  Develop a dashboard with Streamlit that allows users to explore individual patient trajectories and view group-level trends.

## Repository Structure

- **main.py:**  
  Contains code to generate the synthetic longitudinal dataset, perform data simulation, fit a mixed effects model, and build a classification model.

- **dashboard.py:**  
  Provides an interactive Streamlit dashboard to visualize longitudinal trends (e.g., MMSE over time) for individual patients as well as group-level averages.

- **requirements.txt:**  
  Lists all the Python dependencies required to run the project.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/alzheimer-dashboard.git
   cd alzheimer-dashboard
