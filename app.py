import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Dataset09-Employee-salary-prediction.csv")
df.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_Year', 'Salary']
df.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
df['Degree_Encoded'] = le.fit_transform(df['Degree'])
df['Job_Title_Encoded'] = le.fit_transform(df['Job_Title'])

# Scale numeric features
scaler = StandardScaler()
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])
df['Experience_Scaled'] = scaler.fit_transform(df[['Experience_Year']])

# Features and label
X = df[['Age_Scaled', 'Gender_Encoded', 'Degree_Encoded', 'Job_Title_Encoded', 'Experience_Scaled']]
y = df['Salary']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# Streamlit app interface
st.title("💼 Employee Salary Predictor")

age = st.slider("Select Age", 18, 65, 25)
gender = st.selectbox("Select Gender", df['Gender'].unique())
degree = st.selectbox("Select Degree", df['Degree'].unique())
job_title = st.selectbox("Select Job Title", df['Job_Title'].unique())
experience = st.slider("Years of Experience", 0, 40, 1)

if st.button("Predict Salary"):
    gender_encoded = le.transform([gender])[0]
    degree_encoded = le.transform([degree])[0]
    job_encoded = le.transform([job_title])[0]
    age_scaled = scaler.transform([[age]])[0][0]
    exp_scaled = scaler.transform([[experience]])[0][0]
    input_data = np.array([[age_scaled, gender_encoded, degree_encoded, job_encoded, exp_scaled]])
    salary = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{salary:,.2f}")

st.caption(f"Model Accuracy (R² Score): {r2*100:.2f}%")
