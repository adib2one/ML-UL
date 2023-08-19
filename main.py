import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn import metrics
import warnings

# melakukan  Exploratory Data Analysis
data = pd.read_csv('diabetes.csv',sep=',')

X = data.iloc[:,0:8]
y = data.iloc[:,-1]

# Melakukan Normalisasi Data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Melakukan train test pada data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Melakukan Normalisasi data dan melakukan train test pada data yang sudah di normalisasikan

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melakukan pengolahan dengan algoritma Naive Bayes

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)

st.markdown("<style> .title { color:DarkTurquoise; font-size: 20px; border-radius: 25px; background-color: black; text-align : center; border: white;outline-color: white;} </style>",unsafe_allow_html= True)
st.markdown('<h1 class= "title"> Prediksi Penyakit Diabetes dengan Supervised Learning </h1>',unsafe_allow_html= True)

st.markdown('<style> .train { color:black; font-size:15px; }</style>',unsafe_allow_html= True)
st.markdown('<h3 class= "train">Akurasi pada Training Data</h3>',unsafe_allow_html=True)
st.text(f"hasil akurasi pada data traning adalah {metrics.accuracy_score(X_train_pred,y_train)}")

st.markdown('<style> .test { color:black; font-size:15px; }</style>',unsafe_allow_html= True)
st.markdown('<h3 class= "test">Akurasi pada Testing Data</h3>',unsafe_allow_html=True)
st.text(f"hasil akurasi pada data test adalah {metrics.accuracy_score(X_test_pred,y_test)}")


st.markdown('<style> .confusion{ color:black; font-size:15px; }</style>',unsafe_allow_html= True)
st.markdown('<h3 class= "confusion">Hasil Confusion Matrix Specificity </h3>',unsafe_allow_html=True)
#st.text(tuple({metrics.confusion_matrix(y_test,y_pred)}))
cons_matrix = metrics.confusion_matrix(y_test,y_pred)
st.table(cons_matrix)


st.markdown('<style >.predict { color:black;font-size:25px;border-style: solid;text-align:center;border-color: red,outline-color: blue;}<\style>',unsafe_allow_html=True)
st.markdown('<h2 class= "predict"> Melakukan Prediksi </h2>',unsafe_allow_html=True)

col1 , col2 = st.columns(2)

with col1 :
    Pregnancies = st.text_input('input nilai Pregnancies')
    
with col2 :
    Glucose = st.text_input(" input nilai Glucose")

with col1:
    BloodPressure = st.text_input("input nilai BloodPressure")

with col2:
 SkinThickness = st.text_input("input nilai SkinThickness")

with col1:  
    Insulin = st.text_input("input nilai Insulin")

with col2:
    BMI = st.text_input('input nilai BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input('input nilai DiabetesPedigreeFunction ')
    
with col2:
    Age = st.text_input('input nilai Age')

diagnosis = ''

if st.button('Tes prediksi Diabetes',):
    warnings.filterwarnings('ignore')
    diab_predicted = pd.to_numeric([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age],errors='coerce')
    array_predicted = np.array(diab_predicted)
    reshape_predicted = array_predicted.reshape(1,-1)
   # diab_input = np.array(diab_predicted)
   # diab_reshape = diab_input.reshape(1,-1)
   #diab_scaler = scaler.transform(diab_reshape)
   # numeric_predict = [float[item] for item in diab_predicted]
   # array_predict = np.array(numeric_predict,dtype= 'float64')
    model_predict = model.predict(reshape_predicted)
    st.write("""
             <style>
             .output {
            color:purple;
            text-size:25px;
            background-color:white;
            text-align:center;
             }
             
             
             </style>
             """,unsafe_allow_html=True)
    if (model_predict[0] == 0):
        #print('pasien negatif diabetes')
        st.write('<h3 class= "output">pasien negatif diabetes</h3>',unsafe_allow_html=True)
    else:
        #print('pasien positif diabetes')
        st.write('<h3 class= "output">pasien positif diabetes</h3>',unsafe_allow_html=True)
        
    








