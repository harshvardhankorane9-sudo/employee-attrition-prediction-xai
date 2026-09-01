import streamlit as st
import pandas as pd
import joblib

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ======================================
# LOAD MODEL
# ======================================

model = joblib.load("best_model.pkl")

# ======================================
# TITLE
# ======================================

st.title("📊 Employee Attrition Prediction System")

st.markdown("""
This AI system predicts whether an employee is likely to leave the company.
""")

# ======================================
# SIDEBAR
# ======================================

st.sidebar.header("Employee Details")

# ======================================
# INPUTS
# ======================================

Age = st.sidebar.slider("Age", 18, 65, 30)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

Years_at_Company = st.sidebar.slider(
    "Years at Company",
    0,
    40,
    5
)

Job_Role = st.sidebar.selectbox(
    "Job Role",
    [
        "Manager",
        "Developer",
        "Analyst",
        "Sales",
        "HR"
    ]
)

Monthly_Income = st.sidebar.number_input(
    "Monthly Income",
    1000,
    50000,
    5000
)

Work_Life_Balance = st.sidebar.slider(
    "Work-Life Balance",
    1,
    5,
    3
)

Job_Satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,
    5,
    3
)

Performance_Rating = st.sidebar.slider(
    "Performance Rating",
    1,
    5,
    3
)

Number_of_Promotions = st.sidebar.slider(
    "Number of Promotions",
    0,
    10,
    1
)

Overtime = st.sidebar.selectbox(
    "Overtime",
    ["Yes", "No"]
)

Distance_from_Home = st.sidebar.slider(
    "Distance from Home",
    1,
    50,
    10
)

Education_Level = st.sidebar.slider(
    "Education Level",
    1,
    5,
    3
)

Marital_Status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

Number_of_Dependents = st.sidebar.slider(
    "Number of Dependents",
    0,
    10,
    2
)

Job_Level = st.sidebar.slider(
    "Job Level",
    1,
    5,
    2
)

Company_Size = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

Company_Tenure = st.sidebar.slider(
    "Company Tenure",
    0,
    50,
    10
)

Remote_Work = st.sidebar.selectbox(
    "Remote Work",
    ["Yes", "No"]
)

Leadership_Opportunities = st.sidebar.slider(
    "Leadership Opportunities",
    1,
    5,
    3
)

Innovation_Opportunities = st.sidebar.slider(
    "Innovation Opportunities",
    1,
    5,
    3
)

Company_Reputation = st.sidebar.slider(
    "Company Reputation",
    1,
    5,
    3
)

Employee_Recognition = st.sidebar.slider(
    "Employee Recognition",
    1,
    5,
    3
)

# ======================================
# MANUAL ENCODING
# ======================================

Gender = 1 if Gender == "Male" else 0

Overtime = 1 if Overtime == "Yes" else 0

Remote_Work = 1 if Remote_Work == "Yes" else 0

# ======================================
# JOB ROLE ENCODING
# ======================================

job_role_map = {
    "Manager": 0,
    "Developer": 1,
    "Analyst": 2,
    "Sales": 3,
    "HR": 4
}

Job_Role = job_role_map[Job_Role]

# ======================================
# MARITAL STATUS ENCODING
# ======================================

marital_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

Marital_Status = marital_map[Marital_Status]

# ======================================
# COMPANY SIZE ENCODING
# ======================================

company_size_map = {
    "Small": 0,
    "Medium": 1,
    "Large": 2
}

Company_Size = company_size_map[Company_Size]

# ======================================
# CREATE DATAFRAME
# ======================================

input_df = pd.DataFrame({

    'Age': [Age],

    'Gender': [Gender],

    'Years at Company': [Years_at_Company],

    'Job Role': [Job_Role],

    'Monthly Income': [Monthly_Income],

    'Work-Life Balance': [Work_Life_Balance],

    'Job Satisfaction': [Job_Satisfaction],

    'Performance Rating': [Performance_Rating],

    'Number of Promotions': [Number_of_Promotions],

    'Overtime': [Overtime],

    'Distance from Home': [Distance_from_Home],

    'Education Level': [Education_Level],

    'Marital Status': [Marital_Status],

    'Number of Dependents': [Number_of_Dependents],

    'Job Level': [Job_Level],

    'Company Size': [Company_Size],

    'Company Tenure': [Company_Tenure],

    'Remote Work': [Remote_Work],

    'Leadership Opportunities': [Leadership_Opportunities],

    'Innovation Opportunities': [Innovation_Opportunities],

    'Company Reputation': [Company_Reputation],

    'Employee Recognition': [Employee_Recognition]

})

# ======================================
# PREDICT BUTTON
# ======================================

if st.button("Predict Attrition"):

    try:

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Employee Likely to Leave")

        else:

            st.success("✅ Employee Likely to Stay")

        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )

        # ==================================
        # ANALYSIS
        # ==================================

        st.subheader("Risk Analysis")

        reasons = []

        if Job_Satisfaction <= 2:
            reasons.append("Low Job Satisfaction")

        if Work_Life_Balance <= 2:
            reasons.append("Poor Work-Life Balance")

        if Overtime == 1:
            reasons.append("High Overtime")

        if Monthly_Income < 3000:
            reasons.append("Low Monthly Income")

        if Employee_Recognition <= 2:
            reasons.append("Low Employee Recognition")

        if Leadership_Opportunities <= 2:
            reasons.append("Low Leadership Opportunities")

        if len(reasons) == 0:

            st.success(
                "Employee profile appears stable."
            )

        else:

            st.write("Possible risk factors:")

            for r in reasons:
                st.write(f"- {r}")

    except Exception as e:

        st.error(f"Prediction Error: {e}")