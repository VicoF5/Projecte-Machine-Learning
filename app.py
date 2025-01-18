#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!pip install --upgrade numpy pandas


# In[16]:


import pandas as pd   #Numpy: 1.23.3 Pandas: 1.4.4
import numpy as np


# In[12]:


get_ipython().system('pip install streamlit')
import streamlit as st
import pickle

# Cargar el modelo y el escalador desde archivos
with open('linear_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción del Riesgo Crediticio (Regresión Logistica)')

# Entrada de datos del usuario
balance = st.number_input('balance (en euros)')
loan = st.number_input('Loan (yes=1, no=0)')



# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'Balance': [balance],
    'Loan': [loan],
})

# Estandarizar las entradas
user_data_standardized = scaler.transform(user_data)

# Realizar la predicción
prediction = modelo.predict(user_data_standardized)

# Mostrar la predicción
st.write(f'Predicción de riesgo crediticio: {prediction[0]:.2f}')


# In[ ]:




