# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: siddhardhan
"""

import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image

# Load and preprocess the dataset
dataset = pd.read_csv("general.csv")
X = dataset.drop('Disease', axis=1)
y = dataset['Disease']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model1.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

def predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion):
    # Prepare user input as a single-row DataFrame
    user_input = pd.DataFrame({
        'Temp': [temp_f],
        'Pulserate': [pulse_rate_bpm],
        'Vomiting': [vomiting],
        'YellowishUrine': [yellowish_urine],
        'Indigestion': [indigestion]
    })

    # Standardize the user input
    user_input = scaler.transform(user_input)

    # Make prediction
    predicted_disease = model.predict(user_input)[0]
    disease_names = { 0: 'Heart Disease',1: 'Viral Fever/Cold', 2: 'Jaundice', 3: 'Food Poisoning',4: 'Normal'}
    return disease_names[predicted_disease]

def show_attribute_descriptions():
    attribute_descriptions = {
        "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
        "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
        "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
        "MDVP:Jitter(%)": "Several measures of variation in fundamental frequency",
        "MDVP:Jitter(Abs)": "Several measures of variation in fundamental frequency",
        "MDVP:RAP": "Several measures of variation in fundamental frequency",
        "MDVP:PPQ": "Several measures of variation in fundamental frequency",
        "Jitter:DDP": "Several measures of variation in fundamental frequency",
        "MDVP:Shimmer": "Several measures of variation in amplitude",
        "MDVP:Shimmer(dB)": "Several measures of variation in amplitude",
        "Shimmer:APQ3": "Several measures of variation in amplitude",
        "Shimmer:APQ5": "Several measures of variation in amplitude",
        "MDVP:APQ": "Several measures of variation in amplitude",
        "Shimmer:DDA": "Several measures of variation in amplitude",
        "NHR": "Two measures of ratio of noise to tonal components in the voice",
        "HNR": "Two measures of ratio of noise to tonal components in the voice",
        "status": "Health status of the subject (one) - Parkinson's, (zero) - healthy",
        "RPDE": "Two nonlinear dynamical complexity measures",
        "D2": "Two nonlinear dynamical complexity measures",
        "DFA": "Signal fractal scaling exponent",
        "spread1": "Three nonlinear measures of fundamental frequency variation",
        "spread2": "Three nonlinear measures of fundamental frequency variation",
        "PPE": "Three nonlinear measures of fundamental frequency variation",
    }

    st.header("Attribute Descriptions")
    for attribute, description in attribute_descriptions.items():
        st.write(f"**{attribute}**: {description}")


def calculate_bmi(weight, height):
    bmi = weight / (height / 100) ** 2
    return bmi

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# sidebar for navigation
def main():
    with st.sidebar:
        image = Image.open('images/navbar.png')
        st.image(image,width =200) 
        selected = option_menu('Disease Diagnosis and Recommendation System',
                              
                              ['GENERAL','Diabetes Prediction',
                               'Heart Disease Prediction',
                               'Parkinsons Prediction','BMI CALCULATOR'],
                              icons=['dashboard','activity','heart','person','line-chart'],
                              default_index=0)

    if(selected == 'GENERAL'):
        st.title("General Diagnosis") 
        st.write("Please enter the following information:")
        col1,col2 = st.columns([2,1])
        with col1:
         # Get user input
           temp_f = st.text_input("Temperature (F):", value=0)
           pulse_rate_bpm = st.text_input("Pulse rate (bpm):", value=0)
           st.write("Check the Symptoms you have below:")
           vomiting = st.checkbox("Vomiting")
           yellowish_urine = st.checkbox("Yellowish Urine")
           indigestion = st.checkbox("Indigestion")
   
       # Predict disease based on user input
           if st.button("Test Result"):
               predicted_disease = predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion)
               medicine_recommendations = {
       'Heart Disease': 'Follow a heart-healthy diet and exercise regularly.\nIt is crucial to attend all scheduled appointments with your doctor for proper monitoring and management.',
       'Viral Fever/Cold': 'Get plenty of rest and stay hydrated.\nIf your fever persists or is accompanied by severe symptoms, visit a doctor for proper evaluation and treatment.',
       'Jaundice': 'Rest, stay well-hydrated, and follow a balanced diet.\nIf you notice yellowing of the skin or eyes (jaundice), seek medical attention immediately for proper diagnosis and treatment.',
       'Food Poisoning': 'Stay hydrated and avoid solid foods until symptoms subside.\nIf you experience severe symptoms, seek medical attention promptly for proper evaluation and treatment.',
       'Normal': 'Maintain a healthy lifestyle with regular exercise and a balanced diet.\nEven if you are feeling well, have regular check-ups with your doctor to monitor your overall health.'
        } 
           
   
       
       # Show the pop-up box with disease prediction and medicine recommendation
               if predicted_disease in medicine_recommendations:
                   medicine_recommendation = medicine_recommendations[predicted_disease]
                   st.info(f"Predicted Disease: {predicted_disease}")
                   with st.expander("Medicine Recommendation:"):
                       st.info(f"Medicine Recommendation: {medicine_recommendation}")
               else:
                   st.warning("Unknown disease prediction. Please check your input and try again.")
               #st.write(f"Predicted Disease: {predicted_disease}")    
        
        with col2:
            image = Image.open('images/general.png')
            st.image(image,width =500)
        
    # Diabetes Prediction Page
    if (selected == 'Diabetes Prediction'):
        
        # page title
        st.title('Diabetes Prediction')
        
        
        # getting the input data from the user
        col1, col2, col3,col4= st.columns(4)
        
        with col1:
            Pregnancies = st.text_input('No of Pregnancies')
            
        with col2:
            Glucose = st.text_input('Glucose Level')
        
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        
        with col2:
            Insulin = st.text_input('Insulin Level')
        
        with col3:
            BMI = st.text_input('BMI')
        
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
        with col2:
            Age = st.text_input('Age')
        
        with col4:
            image = Image.open('images/diabetes.png')
            st.image(image,width = 400)  

        # code for Prediction
        diab_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
              st.success('The person is diabetic')
              with st.expander("Medicine Recommendation:"):
                    st.info(f"Medicine Recommendation: {'Please Consult a Medical Profesional. Please follow a balanced and healthy diet. It is important to exercise regularly.'}")
            else:
              st.success('The person is not diabetic')
        
           
        #st.success(diab_diagnosis)
    
    
    
    
    # Heart Disease Prediction Page
    if (selected == 'Heart Disease Prediction'):
        
        # page title
        st.title('Heart Disease Prediction')
        
        col1, col2, col3 ,col4= st.columns(4)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            sex_options = ['Male','Female']
            sex = st.selectbox('Sex',sex_options)
            
        with col3:
            cp = st.text_input('Chest Pain type(1,2,3,4)')
            
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
            
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
            
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
            
        with col3:
            exang = st.text_input('Exercise Induced Angina')
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
            
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
            
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
            
        with col1:
            thal = st.text_input('Results of Nuclear Stress Test(0,1,2)')
            
        with col4:
            image = Image.open('images/heart.png')
            st.image(image,width =350)

         
         
        # code for Prediction
        heart_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Heart Disease Test Result'):
            sex_mapping = {'Male': 1, 'Female': 0}
            sex_numeric = sex_mapping[sex]
            input_data = [float(age), float(sex_numeric), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                  float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            input_data = np.array(input_data).reshape(1, -1)
            heart_prediction = heart_disease_model.predict(input_data)
            if heart_prediction[0] == 1:
                st.success('The person is having heart disease')
                with st.expander("Medicine Recommendation:"):
                    st.info(f"Medicine Recommendation: {'SEEK IMMEDIATE MEDICAL ATTENTION. We encourage you to make positive lifestyle changes to reduce risk factors for Heart Disease.'}")
            else:
                st.success('The person does not have any heart disease')
            
        #st.success(heart_diagnosis)
            
         
        
    
    # Parkinson's Prediction Page
    if (selected == "Parkinsons Prediction"):
        
        # page title
        st.title("Parkinson's Disease Prediction")
        st.subheader('Enter the details of your Biomedical Voice Measurement Test:')
        col1, col2, col3, col4, col5,col6 = st.columns(6)  
        
        with col1:
            fo = st.text_input('Fo(Hz)')
            
        with col2:
            fhi = st.text_input('Fhi(Hz)')
            
        with col3:
            flo = st.text_input('Flo(Hz)')
            
        with col4:
            Jitter_percent = st.text_input('Jitter(%)')
            
        with col5:
            Jitter_Abs = st.text_input('Jitter(Abs)')
            
        with col1:
            RAP = st.text_input('RAP')
            
        with col2:
            PPQ = st.text_input('PPQ')
            
        with col3:
            DDP = st.text_input('DDP')
            
        with col4:
            Shimmer = st.text_input('Shimmer')
            
        with col5:
            Shimmer_dB = st.text_input('Shimmer(dB)')
            
        with col1:
            APQ3 = st.text_input('APQ3')
            
        with col2:
            APQ5 = st.text_input('APQ5')
            
        with col3:
            APQ = st.text_input('APQ')
            
        with col4:
            DDA = st.text_input('DDA')
            
        with col5:
            NHR = st.text_input('NHR')
            
        with col1:
            HNR = st.text_input('HNR')
            
        with col2:
            RPDE = st.text_input('RPDE')
            
        with col3:
            DFA = st.text_input('DFA')
            
        with col4:
            spread1 = st.text_input('spread1')
            
        with col5:
            spread2 = st.text_input('spread2')
            
        with col1:
            D2 = st.text_input('D2')
            
        with col2:
            PPE = st.text_input('PPE')
            
        with col6:
            image = Image.open('images/parkinsons.png')
            st.image(image,width =350)

        
        # code for Prediction
        parkinsons_diagnosis = ''
        
        # creating a button for Prediction    
        if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
              st.success('The person has Parkinsons disease')
              with st.expander("Medicine Recommendation:"):
                st.info(f"Medicine Recommendation: {'CONSULT A NEUROLOGIST. Please take the prescribed medications. We also recommend physical and occupational therapy to improve mobility.'}")
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease"
        if st.button("Show Attribute Descriptions"):
            show_attribute_descriptions()    

             
        #st.success(parkinsons_diagnosis)
        
    if (selected == 'BMI CALCULATOR'):
        
        st.title("BMI CALCULATOR")

        st.write("Body Mass Index (BMI) is a measure of body fat based on height and weight.")
        st.write("Use this calculator to find out your BMI category.")
        col1,col2 = st.columns([2,1])
        with col1:
            weight = st.text_input("Enter your weight (in kilograms)")
            height = st.text_input("Enter your height (in centimeters)")
        
            if st.button("Calculate BMI"):
                
                weight = float(weight)
                height = float(height)
                bmi = calculate_bmi(weight, height)
                category = interpret_bmi(bmi)
        
                st.write("### Results")
                st.write(f"Your BMI: {bmi:.2f}")
                st.write(f"Category: {category}")
        with col2:
            image = Image.open('images/bmi.png')
            st.image(image,width =350)   


    

if __name__ == "__main__":
    main()

st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.markdown("<p style = 'color:grey;'>This is a prediction web app for informational purposes only.\n It is not a substitute for professional medical advice.\nPlease consult a doctor or visit a hospital for proper diagnosis and treatment.</p>",unsafe_allow_html=True)
st.write("\n")
st.write("\n")
st.markdown('<p style="font-size:12px; color:#808080;">©2023 Internship Project by LISA BOJAMMA MS for DLithe</p>', unsafe_allow_html=True)




