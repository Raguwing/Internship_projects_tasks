import streamlit as st
import numpy as np
from pickle import load
import pandas as pd

Std_scaler = load(open(r"D:\Btech project all\ML_Models\Std_scaler.pkl", 'rb'))
encoder_ohe = load(open(r"D:\Btech project all\ML_Models\encoder_ohe.pkl", 'rb'))
RF_regressor = load(open(r"D:\Btech project all\ML_Models\RF_regressor.pkl", 'rb'))

def main():
    st.title("Welcome to Model Prediction")
    'Hello My name is GUDLA RAGUWING, I have worked on ML Model development to predict Laptop prices for the given hstorical data'
    st.title("Medical Cost Prediction")
    html_temp ="""
    <div style="background-color:#025246 ;padding:10x">
    <h2 style="color:white;text-align:center;">Laptop Price Prediction</h2>
    </div>
    """

    age = st.text_input("AGE", placeholder="Enter value",)
    sex = st.text_input("GENDER", placeholder="Enter Gender")
    bmi = st.text_input("BMI", placeholder="Enter value")
    children = st.text_input("CHILDREN", placeholder="Enter no of children")
    smoker = st.text_input("SMOKER", placeholder="Enter")



    btn_click = st.button("Predict")

    if btn_click == True:
        if age and sex and bmi and children and smoker:
            query_point1 = np.array([ int(bmi),int(age),int(children)]).reshape(1, -1)
            query_point2 = np.array([str(sex),str(smoker)]).reshape(1, -1)
            query_point_trans1 = Std_scaler.transform(query_point1)
        
            query_point_trans2 =encoder_ohe.transform(query_point2)
            query_point_tran = np.concatenate((query_point_trans1,query_point_trans2), axis =1)


            pred =RF_regressor.predict(query_point_tran)

            st.success(pred)
        else:
            st.error("Enter the values properly.")



if __name__=='__main__':
    main()
