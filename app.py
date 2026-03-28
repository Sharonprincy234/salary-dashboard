import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("💼 Salary Prediction App")

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Convert text data to numbers
df = pd.get_dummies(df)

st.write("Columns:", df.columns)

# Split data
X = df.drop("Monthly_Salary", axis=1)
y = df["Monthly_Salary"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# User input
st.sidebar.header("Enter Details")

inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(col, value=0)

input_df = pd.DataFrame([inputs])

# Prediction
if st.button("Predict"):
    result = model.predict(input_df)
    st.success(f"💰 Salary: ₹{result[0]:,.2f}")
