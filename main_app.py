#Library imports

import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# import keras.models import load_model

# Loading the model and vectorizer
with open('vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

# Name of classes
Class_names = ['Not Spam','Spam']

# Setting title of app
st.title('Email Classifier')
st.write('Enter the email text')
email_input = st.text_area("Email Text", height = 200)

# Function to predict email classification
def classify_email(email):
    email_vector = vectorizer.transform([email])
    prediction  = model.predict(email_vector)
    return prediction[0]

if st.button("Classify"):
    if email_input:
        result = classify_email(email_input)
        st.write(f"The email is classified as: {Class_names[result]}")
    else:
        st.write('Please enter an email to classify')



