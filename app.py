#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!pip install numpy<2.0,>=1.26.0  pandas==2.2.3


# In[22]:


"""
Compatible versions:
Pandas version: 1.4.4
Numpy version: 1.23.3
Scipy version: 1.9.1
Streamlit version: 1.41.1
Scikit-learn version: 1.6.1
"""

import pandas as pd   
import numpy as np
import scipy
import streamlit as st  
import pickle




print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Scipy version:", scipy.__version__)
print("Streamlit version:", st.__version__)


# In[45]:


# Cargar el modelo y el escalador desde archivos
with open('log_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción del Riesgo Crediticio (Regresión Logistica)')

# Entrada de datos del usuario
balance = st.number_input('balance (en euros)', step=1.0)
loan = st.number_input('Loan (yes=1, no=0)', step=1)



# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'balance': [balance],
    'loan': [loan],
})

user_data['balance'] = pd.to_numeric(user_data['balance'], errors='coerce')

#chequeo tipo de dato
print(user_data['balance'].dtype)
print(user_data['loan'].dtype)


# In[49]:


# Estandarizar solo "balance"
columnas_stand = user_data['balance'].values.reshape(-1, 1)
user_data_standardized = scaler.transform(columnas_stand)

# Convierte el numpy.ndarray a DataFrame para concatenar con el otro input loan que no estandarize
user_data_standardized_df = pd.DataFrame(user_data_standardized, columns=['balance'])

# Concatenar ambos inputs en un dataframe
loan_column = pd.DataFrame(user_data['loan'], columns=['loan'])

final_df = pd.concat([user_data_standardized_df, loan_column], axis=1)


#visualizo
final_df.info()



# In[47]:


# Realizar la predicción

prediction = modelo.predict(final_df)

# Mostrar la predicción
st.write(f'Predicción de riesgo crediticio: {prediction[0]:.2f}')


# In[ ]:




