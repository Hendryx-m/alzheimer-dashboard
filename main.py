import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_patients = 6900 - 4571 + 1  # 2150 patients
num_visits = 3  # baseline, 1-year, and 2-year follow-ups

# Generating patient IDs
patient_ids = np.arange(4571, 6900 + 1)

# Create the synthetic baseline dataset with extra variables
baseline_data = pd.DataFrame({
    'PatientID': patient_ids,

    # Demographic Details
    'Age': np.random.randint(60, 91, size=num_patients),
    'Gender': np.random.choice([0, 1], size=num_patients),  # 0: Male, 1: Female
    'Ethnicity': np.random.choice([0, 1, 2, 3], size=num_patients),  # 0: Caucasian, etc.
    'EducationLevel': np.random.choice([0, 1, 2, 3], size=num_patients),

    # Lifestyle Factors
    'BMI': np.round(np.random.uniform(15, 40, size=num_patients), 1),
    'Smoking': np.random.choice([0, 1], size=num_patients),
    'AlcoholConsumption': np.random.randint(0, 21, size=num_patients),
    'PhysicalActivity': np.round(np.random.uniform(0, 10, size=num_patients), 1),
    'DietQuality': np.random.randint(0, 11, size=num_patients),
    'SleepQuality': np.random.randint(4, 11, size=num_patients),

    # Medical History
    'FamilyHistoryAlzheimers': np.random.choice([0, 1], size=num_patients),
    'CardiovascularDisease': np.random.choice([0, 1], size=num_patients),
    'Diabetes': np.random.choice([0, 1], size=num_patients),
    'Depression': np.random.choice([0, 1], size=num_patients),
    'HeadInjury': np.random.choice([0, 1], size=num_patients),
    'Hypertension': np.random.choice([0, 1], size=num_patients),

    # Clinical Measurements
    'SystolicBP': np.random.randint(90, 181, size=num_patients),
    'DiastolicBP': np.random.randint(60, 121, size=num_patients),
    'CholesterolTotal': np.random.randint(150, 301, size=num_patients),
    'CholesterolLDL': np.random.randint(50, 201, size=num_patients),
    'CholesterolHDL': np.random.randint(20, 101, size=num_patients),
    'CholesterolTriglycerides': np.random.randint(50, 401, size=num_patients),

    # Cognitive and Functional Assessments
    'MMSE': np.random.randint(25, 31, size=num_patients),  # assume better scores at baseline
    'FunctionalAssessment': np.random.randint(7, 11, size=num_patients),
    'MemoryComplaints': np.random.choice([0, 1], size=num_patients),
    'BehavioralProblems': np.random.choice([0, 1], size=num_patients),
    'ADL': np.random.randint(7, 11, size=num_patients),

    # Symptoms
    'Confusion': np.random.choice([0, 1], size=num_patients),
    'Disorientation': np.random.choice([0, 1], size=num_patients),
    'PersonalityChanges': np.random.choice([0, 1], size=num_patients),
    'DifficultyCompletingTasks': np.random.choice([0, 1], size=num_patients),
    'Forgetfulness': np.random.choice([0, 1], size=num_patients),

    # Diagnosis Information (at baseline, before any follow-up)
    'Diagnosis': np.random.choice([0, 1], size=num_patients),

    # Confidential Information
    'DoctorInCharge': 'XXXConfid',

    # Additional Environmental & Socioeconomic Factors
    'Income': np.random.randint(20000, 150001, size=num_patients),  # annual income in USD
    'NeighborhoodQuality': np.random.randint(0, 11, size=num_patients),  # scale 0 (poor) to 10 (excellent)
    'EmploymentStatus': np.random.choice([0, 1, 2], size=num_patients),  # 0: Unemployed, 1: Employed, 2: Retired
    'MaritalStatus': np.random.choice([0, 1, 2, 3], size=num_patients),  # 0: Single, 1: Married, 2: Divorced, 3: Widowed

    # Medication or Treatment Data
    'Treatment': np.random.choice([0, 1], size=num_patients)  # 0: Placebo/No treatment, 1: Receiving treatment
})

# --------------------------
# Longitudinal Data Simulation
# --------------------------
rows = []
for _, patient in baseline_data.iterrows():
    # Set baseline values (that may change over time)
    mmse = patient['MMSE']
    functional = patient['FunctionalAssessment']
    adl = patient['ADL']
    systolic_base = patient['SystolicBP']
    diastolic_base = patient['DiastolicBP']
    cholesterol_total_base = patient['CholesterolTotal']

    diagnosed = 0  # assume not diagnosed at baseline
    for visit in range(num_visits):
        # At baseline (visit 0) values are as initialized;
        # for follow-up visits, simulate changes:
        if visit > 0:
            decline = np.random.randint(0, 3)
            mmse = max(0, mmse - decline)
            functional = max(0, functional - np.random.randint(0, 2))
            adl = max(0, adl - np.random.randint(0, 2))
        
        # Simulate slight fluctuations in clinical measurements
        systolic = systolic_base + np.random.randint(-5, 6)
        diastolic = diastolic_base + np.random.randint(-3, 4)
        cholesterol_total = cholesterol_total_base + np.random.randint(-10, 11)

        # Diagnosis simulation: probability increases as MMSE falls below 24
        if diagnosed == 0:
            p = 1 / (1 + np.exp(mmse - 24))
            if np.random.rand() < p:
                diagnosed = 1
        diagnosis = diagnosed

        # Create a longitudinal record (most baseline variables remain constant)
        record = patient.copy()
        record['Visit'] = visit
        record['MMSE'] = mmse
        record['SystolicBP'] = systolic
        record['DiastolicBP'] = diastolic
        record['CholesterolTotal'] = cholesterol_total
        record['Diagnosis'] = diagnosis
        record['FunctionalAssessment'] = functional
        record['ADL'] = adl
        rows.append(record)

# Build the full longitudinal DataFrame
longitudinal_df = pd.DataFrame(rows)
longitudinal_df.reset_index(drop=True, inplace=True)

# Save the dataset to CSV for later use (for example, in the dashboard)
longitudinal_df.to_csv("longitudinal_data.csv", index=False)

print("Longitudinal data preview:")
print(longitudinal_df.head(10))

# --------------------------
# Mixed Effects Model
# --------------------------
# We use the mixed linear model to predict MMSE using visit, Age, and Treatment.
# A random intercept is included for each PatientID.
md = smf.mixedlm("MMSE ~ Visit + Age + Treatment", longitudinal_df, groups=longitudinal_df["PatientID"])
mdf = md.fit()
print("\nMixed Effects Model Summary:")
print(mdf.summary())

# --------------------------
# Classification Model
# --------------------------
# We'll build a classification model to predict the final diagnosis.
# Here we use data from the last visit (Visit==2).
last_visit = longitudinal_df[longitudinal_df["Visit"] == (num_visits - 1)]
# Use some predictors, for example: Age, BMI, Treatment, and baseline MMSE.
X = last_visit[["Age", "BMI", "Treatment", "MMSE"]]
y = last_visit["Diagnosis"]

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Classification Accuracy:", accuracy_score(y_test, y_pred))
