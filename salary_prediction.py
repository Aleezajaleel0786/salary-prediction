import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as  np

# Load trained model and preprocessing tools
model = load_model("salary_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Sample input with dummy Surname and RowNumber
new_customer = {
    'CreditScore': 650,
    'Age': 30,
    'Tenure': 5,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 70000,
    'EducationLevel': 3,
    'Experience': 7,
    'SkillCount': 6,
    'Surname': 'TestUser',
    'RowNumber': 1
}

input_df = pd.DataFrame([new_customer])

# Drop unnecessary columns
input_df.drop(['Surname', 'RowNumber'], axis=1, inplace=True)

# Add missing training columns if any
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]


# Load saved objects
model = load_model("salary_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("Salary prediction Model")

# User inputs
geo = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender_male = st.selectbox("Gender", ['Male', 'Female']) == 'Male'
balance = st.number_input("Balance:", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", value=680)
age = st.slider("Age", 18,19)
tenure = st.slider("Tenure", 0,10,5)
num_products = st.slider("Number of Products", 1,4)
has_cr_card = st.selectbox(" Credit card?", options=[0, 1])

is_active = st.selectbox(' Active member?', ['0', '1'])
estimated_salary = st.number_input("Estimated Salary", value=60000)





# Prepare input dict
new_customer = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary,
    'Gender_Male': int(gender_male),
    'Geography_Germany': 1 if geo == 'Germany' else 0,
    'Geography_Spain': 1 if geo == 'Spain' else 0
}

if st.button("Predicted salary"):
    new_df = pd.DataFrame([new_customer])
    new_df = new_df.reindex(columns=feature_names, fill_value=0)
    new_scaled = scaler.transform(new_df)
    prediction = model.predict(new_scaled)
    st.success(f"Predicted Salary: {prediction[0][0]:,.2f}")

    
    
    
    
    
# Sample input with dummy Surname and RowNumber
new_customer = {
    'CreditScore': 650,
    'Age': 30,
    'Tenure': 5,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 70000,
    'EducationLevel': 3,
    'Experience': 7,
    'SkillCount': 6,
    'Surname': 'TestUser',
    'RowNumber': 1
}

input_df = pd.DataFrame([new_customer])

# Drop unnecessary columns
input_df.drop(['Surname', 'RowNumber'], axis=1, inplace=True)

# Add missing training columns if any
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# Scale input and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print(f"The customer's estimated balance is: {int(prediction[0][0])}")

