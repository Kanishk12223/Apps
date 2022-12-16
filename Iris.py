import base64
import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("rf.pk1", "rb")
rf = pickle.load(pickle_in)
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    prediction = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction
def main():
    st.title("Iris species prediction")
    with open('iris.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    sepal_length = st.slider("Sepal length", min_value= 4.3, max_value= 7.9, step= 0.1)
    sepal_width = st.slider("Sepal width", min_value= 2.0, max_value= 4.4, step= 0.1)
    petal_length = st.slider("Petal length", min_value= 1.0, max_value= 6.9, step= 0.1)
    petal_width = st.slider("Petal width", min_value= 0.1, max_value= 2.5, step= 0.1)
    result = ""
    if st.button("Predict"):
        result= predict_species(sepal_length, sepal_width, petal_length, petal_width)
        if result == 0:
            result = 'Setosa'
        elif result == 1:
            result = 'Versicolor'
        else:
            result = 'Virginica'
    st.success('The flower belongs to {}'.format(result))
if __name__ == '__main__':
    main()