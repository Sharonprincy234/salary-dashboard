import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Salary Prediction App")

df = pd.read_csv("employee_salary_dataset.csv")

X = df.drop("Monthly_Salary", axis=1)
y = df["Monthly_Salary"]

model = RandomForestRegressor()
model.fit(X, y)

st.sidebar.header("Enter Details")

inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(col, value=0)

input_df = pd.DataFrame([inputs])

if st.button("Predict"):
    result = model.predict(input_df)
    st.success(f"Salary: ₹{result[0]:,.2f}")
