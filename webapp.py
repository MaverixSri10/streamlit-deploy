import streamlit as st
import pandas as pd
import joblib


#age                             75.0
#anaemia                          0.0
#creatinine_phosphokinase       582.0
#diabetes                         0.0
#ejection_fraction               20.0
#high_blood_pressure              1.0
#platelets                   265000.0
#serum_creatinine                 1.9
#serum_sodium                   130.0
#sex                              1.0
#smoking                          0.0
#time                             4.0
#DEATH_EVENT                      1.0

st.title('Heart Failure Prediction')

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# Creating the input fields
age = st.number_input("age")
anaemia = st.selectbox("anaemia", pd.unique(df["anaemia"]))
creatinine_phosphokinase = st.number_input("creatinine_phosphokinase")
diabetes = st.selectbox("diabetes", pd.unique(df["diabetes"]))
ejection_fraction = st.number_input("ejection_fraction")
high_blood_pressure = st.selectbox("high_blood_pressure", pd.unique(df["high_blood_pressure"]))
platelets = st.number_input("platelets")
serum_creatinine = st.number_input("serum_creatinine")
serum_sodium = st.number_input("serum_sodium")
sex = st.selectbox("sex", pd.unique(df["sex"]))
smoking = st.selectbox("smoking", pd.unique(df["smoking"]))
time = st.number_input("time")


input = {
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": diabetes,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": high_blood_pressure,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time
}

# Click prediction button

if st.button('Predction'):
    model = joblib.load("DT.pkl")

    x_input = pd.DataFrame(input, index=[0])

    prediction = model.predict(x_input)

    st.write(prediction)


