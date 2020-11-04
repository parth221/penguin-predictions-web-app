import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.write("""
# Penguin Prediction App
""")
st.sidebar.header('User Input Features')


def user_selected_input():
    island = st.sidebar.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    feature = pd.DataFrame(data, index=[0])
    return feature


input_df = user_selected_input()
label_species = joblib.load('species')
label_sex = joblib.load('sex')
label_island = joblib.load('island')
model = joblib.load('classifier.pkl')
input_df['sex'] = label_sex.transform(input_df['sex'])
input_df['island'] = label_island.transform(input_df['island'])
preds = model.predict(input_df)
preds_pro = model.predict_proba(input_df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[preds])
st.write(label_species.classes_)

st.subheader('Prediction Probability')
final_preds = pd.DataFrame(preds_pro, columns=['Adelie', 'Chinstrap','Gentoo'])
st.write(final_preds)
