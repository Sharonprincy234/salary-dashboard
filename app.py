import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("💼 Salary Prediction Dashboard")

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")

# Clean column names (remove spaces + lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert text data to numeric
df = pd.get_dummies(df)

# Show dataset (optional)
st.write("### Dataset Preview")
st.dataframe(df.head())

# Show columns (for safety)
st.write("Columns:", df.columns)

# -----------------------------
# IMPORTANT: Target column
# -----------------------------
# Change this ONLY if needed after checking column names
target_column = "monthly_salary"   # or "salary"

# Split data
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.sidebar.header("Enter Details")

inputs = {}

for col in X.columns:
    inputs[col] = st.sidebar.number_input(f"{col}", value=0)

input_df = pd.DataFrame([inputs])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
