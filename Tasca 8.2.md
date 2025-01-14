# Preparación de datos: "bank_dataset.csv"

# 1 - Importación de Librerías


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install sweetviz
import sweetviz as sv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
```

    Requirement already satisfied: sweetviz in c:\users\victo\anaconda3\lib\site-packages (2.3.1)
    Requirement already satisfied: tqdm>=4.43.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (4.64.1)
    Requirement already satisfied: matplotlib>=3.1.3 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (3.5.2)
    Requirement already satisfied: pandas!=1.0.0,!=1.0.1,!=1.0.2,>=0.25.3 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.4.4)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.9.1)
    Requirement already satisfied: jinja2>=2.11.1 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (2.11.3)
    Requirement already satisfied: importlib-resources>=1.2.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (6.4.5)
    Requirement already satisfied: numpy>=1.16.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.21.5)
    Requirement already satisfied: zipp>=3.1.0 in c:\users\victo\anaconda3\lib\site-packages (from importlib-resources>=1.2.0->sweetviz) (3.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\victo\anaconda3\lib\site-packages (from jinja2>=2.11.1->sweetviz) (2.0.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (21.3)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (3.0.9)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (9.2.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (1.4.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas!=1.0.0,!=1.0.1,!=1.0.2,>=0.25.3->sweetviz) (2024.2)
    Requirement already satisfied: colorama in c:\users\victo\anaconda3\lib\site-packages (from tqdm>=4.43.0->sweetviz) (0.4.5)
    Requirement already satisfied: six>=1.5 in c:\users\victo\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.1.3->sweetviz) (1.16.0)
    

# 2 - Carga del dataset "raw" desde github


```python
df = pd.read_csv('https://raw.githubusercontent.com/ITACADEMYprojectes/projecteML/e8d1aab0a24ddf55af9dfd9e83b1ea79e34c1af9/bank_dataset.CSV')
```

# 3 -  Exploración del dataset


```python
# Información general del dataset
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11162 entries, 0 to 11161
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   age        11152 non-null  float64
     1   job        11162 non-null  object 
     2   marital    11157 non-null  object 
     3   education  11155 non-null  object 
     4   default    11162 non-null  object 
     5   balance    11162 non-null  int64  
     6   housing    11162 non-null  object 
     7   loan       11162 non-null  object 
     8   contact    11162 non-null  object 
     9   day        11162 non-null  int64  
     10  month      11162 non-null  object 
     11  duration   11162 non-null  int64  
     12  campaign   11162 non-null  int64  
     13  pdays      11162 non-null  int64  
     14  previous   11162 non-null  int64  
     15  poutcome   11162 non-null  object 
     16  deposit    11162 non-null  object 
    dtypes: float64(1), int64(6), object(10)
    memory usage: 1.4+ MB
    


```python
# Visualización primeras filas
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56.0</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.0</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.0</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.0</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Estadísticas descriptivas para variables numéricas
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>balance</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11152.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.235384</td>
      <td>1528.538524</td>
      <td>15.658036</td>
      <td>371.993818</td>
      <td>2.508421</td>
      <td>51.330407</td>
      <td>0.832557</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.914934</td>
      <td>3225.413326</td>
      <td>8.420740</td>
      <td>347.128386</td>
      <td>2.722077</td>
      <td>108.758282</td>
      <td>2.292007</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>-6847.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>122.000000</td>
      <td>8.000000</td>
      <td>138.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>550.000000</td>
      <td>15.000000</td>
      <td>255.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>1708.000000</td>
      <td>22.000000</td>
      <td>496.000000</td>
      <td>3.000000</td>
      <td>20.750000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95.000000</td>
      <td>81204.000000</td>
      <td>31.000000</td>
      <td>3881.000000</td>
      <td>63.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Estadísticas descriptivas para variables categóricas
df.describe(include=['object'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11162</td>
      <td>11157</td>
      <td>11155</td>
      <td>11162</td>
      <td>11162</td>
      <td>11162</td>
      <td>11162</td>
      <td>11162</td>
      <td>11162</td>
      <td>11162</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>12</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>management</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>may</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2566</td>
      <td>6349</td>
      <td>5474</td>
      <td>10994</td>
      <td>5881</td>
      <td>9702</td>
      <td>8042</td>
      <td>2824</td>
      <td>8326</td>
      <td>5873</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Recuento valores nulos por columna
df.isnull().sum()
```




    age          10
    job           0
    marital       5
    education     7
    default       0
    balance       0
    housing       0
    loan          0
    contact       0
    day           0
    month         0
    duration      0
    campaign      0
    pdays         0
    previous      0
    poutcome      0
    deposit       0
    dtype: int64



### Comentarios:

De un primer análisis se puede ver que el conjunto de datos tiene 17 columnas, 11162 registros. El dataset contiene información sobre diversas características relacionadas con interacciones bancarias, que incluyen tanto variables numéricas como categóricas.  

Las variables categóricas son 10 e incluyen: job, marital, education, default (la variable objetivo), housing, loan, contact, month, poutcome y deposit .
Las Variables numéricas son 7 e incluyen: age, balance, day, duration, campaign, pdays, y previous.
Hay 22 valores faltantes presentes en las variables age, marital y education.
Mi variable objetivo es "default": indica si el cliente tiene un crédito en "deafult" (sin pagar).


Para las variables númericas se incluyó un análisis descriptivo evaluando medidas de tendencia central (media, mediana) y dispersión (desviación estándar).  
Para las variables categóricas también se detalla una descripción de los datos. 



# 4 - Gestión valores nulos

### Comentarios:

Antes de dividir el conjunto de datos en entrenamiento y prueba voy a tratar valores faltantes utilizando la mediana para las variables numéricas y la moda para categóricas que es el valor que aparece con mayor frecuencia.


```python
#Para la variable númerica "age" uso la mediana
df['age'] = df['age'].fillna(df['age'].median())

#Para las variables categóricas "marital" y "education" uso la moda, que es el valor que aparece con mayor frecuencia
df['marital'] = df['marital'].fillna(df['marital'].mode()[0])
df['education'] = df['education'].fillna(df['education'].mode()[0])

#compruebo
df.isnull().sum()

```




    age          0
    job          0
    marital      0
    education    0
    default      0
    balance      0
    housing      0
    loan         0
    contact      0
    day          0
    month        0
    duration     0
    campaign     0
    pdays        0
    previous     0
    poutcome     0
    deposit      0
    dtype: int64



# 5 - Cambio de tipo de variable

### Comentarios

Previo a la división del dataset y para posteriormente poder correr el modelo de ML con mayor facilidad cambiare mi variable objetivo "default" de categórica a númerica, así como tambien de la variables "loan" (si tiene un crédito personal), housing (so tien un crédito hipotecario) y "deposit" (si tiene un deposito) que tiene como valores "sí" o "no" y entiendo me serán de utilidad mas adelante. Para evitar que me queden como "nan" tambíen las convertí de float a int.


```python
print("Valores únicos en 'default':", df['default'].unique())
print("Valores únicos en 'loan':", df['loan'].unique())
print("Valores únicos en 'deposit':", df['deposit'].unique())
print("Valores únicos en 'housing':", df['deposit'].unique())
```

    Valores únicos en 'default': ['no' 'yes']
    Valores únicos en 'loan': ['no' 'yes']
    Valores únicos en 'deposit': ['yes' 'no']
    Valores únicos en 'housing': ['yes' 'no']
    


```python
# Transformo los valores "yes" a 1 y "no" a 0 

df['default'] = df['default'].map({'yes': 1, 'no': 0})
df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})

# Convertir las columnas a enteros (int)
df['default'] = df['default'].astype(int)
df['loan'] = df['loan'].astype(int)
df['deposit'] = df['deposit'].astype(int)
df['housing'] = df['housing'].astype(int)

# Verificación
print('default:', df['default'].dtype)
print('loan:', df['loan'].dtype) 
print('deposit:',df['deposit'].dtype) 
print('housing:',df['housing'].dtype) 
```

    default: int32
    loan: int32
    deposit: int32
    housing: int32
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11162 entries, 0 to 11161
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   age        11162 non-null  float64
     1   job        11162 non-null  object 
     2   marital    11162 non-null  object 
     3   education  11162 non-null  object 
     4   default    11162 non-null  int32  
     5   balance    11162 non-null  int64  
     6   housing    11162 non-null  int32  
     7   loan       11162 non-null  int32  
     8   contact    11162 non-null  object 
     9   day        11162 non-null  int64  
     10  month      11162 non-null  object 
     11  duration   11162 non-null  int64  
     12  campaign   11162 non-null  int64  
     13  pdays      11162 non-null  int64  
     14  previous   11162 non-null  int64  
     15  poutcome   11162 non-null  object 
     16  deposit    11162 non-null  int32  
    dtypes: float64(1), int32(4), int64(6), object(6)
    memory usage: 1.3+ MB
    


```python
#verifo no tener nulos después del mapeo

df.isnull().sum()
```




    age          0
    job          0
    marital      0
    education    0
    default      0
    balance      0
    housing      0
    loan         0
    contact      0
    day          0
    month        0
    duration     0
    campaign     0
    pdays        0
    previous     0
    poutcome     0
    deposit      0
    dtype: int64



# 6 - División del dataset en "train" y "test"

### Comentarios

Utilizare el método "Train/Test" donde dividire el dataset en dos conjuntos de datos, el de entrenamiento con un 80% de los datos, y el de prueba con el restante 20%. Esta proporción es la standard. Con el de entrenamiento creare el modelo, y con el de prueba comprobare la precisión del modelo.También asegurare que la división sea reproducible con random_state=0 y que proporción de "default" sea la misma en ambos conjuntos con stratify=y.

Se verificó la estratificacion y el balance de las clases se mantiene.



```python
# Separo las columnas de las caracteristicas (X) y la variable objetivo (y) 
X = df.drop(columns=['default'])  # caracteristicas (features)
y = df['default']  # Variable objetivo (label)


# Divido los datasets en conjuntos de entrenamiento (train) y prueba(test)
train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=y)

#Verifico
print("Tamaño del conjunto de entrenamiento (train):", train.shape)
print("Tamaño del conjunto de prueba (test):", test.shape)
```

    Tamaño del conjunto de entrenamiento (train): (8929, 17)
    Tamaño del conjunto de prueba (test): (2233, 17)
    


```python
# Verifico el balance de clases (estratificación):
# Proporción de clases en el dataset original
print("Proporción de clases en el dataset original:")
print(y.value_counts(normalize=True))

# Proporción de clases en el conjunto de entrenamiento
print("\nProporción de clases en el conjunto de entrenamiento:")
print(train['default'].value_counts(normalize=True))

# Proporción de clases en el conjunto de prueba
print("\nProporción de clases en el conjunto de prueba:")
print(test['default'].value_counts(normalize=True))
```

    Proporción de clases en el dataset original:
    0    0.984949
    1    0.015051
    Name: default, dtype: float64
    
    Proporción de clases en el conjunto de entrenamiento:
    0    0.984993
    1    0.015007
    Name: default, dtype: float64
    
    Proporción de clases en el conjunto de prueba:
    0    0.984774
    1    0.015226
    Name: default, dtype: float64
    


```python
# Chequeo las primeras filas de ambos conjuntos
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2524</th>
      <td>49.0</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>telephone</td>
      <td>13</td>
      <td>aug</td>
      <td>640</td>
      <td>2</td>
      <td>101</td>
      <td>4</td>
      <td>success</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10329</th>
      <td>47.0</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>4040</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>11</td>
      <td>aug</td>
      <td>53</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2784</th>
      <td>65.0</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>571</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>20</td>
      <td>oct</td>
      <td>349</td>
      <td>1</td>
      <td>183</td>
      <td>2</td>
      <td>success</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5558</th>
      <td>36.0</td>
      <td>services</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>unknown</td>
      <td>6</td>
      <td>jun</td>
      <td>308</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>36.0</td>
      <td>technician</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>202</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>29</td>
      <td>jan</td>
      <td>313</td>
      <td>1</td>
      <td>120</td>
      <td>3</td>
      <td>failure</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>643</th>
      <td>50.0</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>0</td>
      <td>2284</td>
      <td>1</td>
      <td>0</td>
      <td>telephone</td>
      <td>31</td>
      <td>jul</td>
      <td>1088</td>
      <td>17</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6148</th>
      <td>43.0</td>
      <td>retired</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>7298</td>
      <td>0</td>
      <td>0</td>
      <td>unknown</td>
      <td>11</td>
      <td>jun</td>
      <td>36</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4334</th>
      <td>25.0</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>4718</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>24</td>
      <td>jul</td>
      <td>709</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1687</th>
      <td>33.0</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>unknown</td>
      <td>0</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>30</td>
      <td>apr</td>
      <td>482</td>
      <td>1</td>
      <td>161</td>
      <td>1</td>
      <td>failure</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>41.0</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>4004</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>17</td>
      <td>apr</td>
      <td>314</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 7 - Detección y gestion de outliers en "train"

### Comentarios

Hago la detección y tratamiento de outliers en el dataset de entrenamiento y luego aplico directamente él metodo al dataset de prueba. Elimino del análisis mi variable objetivo, así como las que convertí en 1 y 0 anteriormente: default, loan, deposit y housing para simplificar la visualización gráfica ya que son variables binarias. 


```python
# Lista de columnas númericas para detectar outliers
columnas_outlier = ['age', 'balance', 'day', 'duration','campaign','pdays','previous']
```


```python
# Detección de outliers con boxplots para las columnas definidas
for i in columnas_outlier:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[i])
    plt.xlabel(i)
    plt.grid(True)
    plt.show() 
```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    



    
![png](output_30_2.png)
    



    
![png](output_30_3.png)
    



    
![png](output_30_4.png)
    



    
![png](output_30_5.png)
    



    
![png](output_30_6.png)
    



```python
# En esta instancia para tratar los outliers aplicaré la Winsorización de forma de mantener todos los datos reduciendo el impacto de los valores atípcos

# Función para calcular límites IQR
def calculate_iqr_limits(df, columns):
    limits_dict = {}
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        limits_dict[column] = (lower_bound, upper_bound)
        
    return limits_dict


```


```python
# Función para winsorizar datos
def winsorize_data(df, columns):
    limits_dict = calculate_iqr_limits(df, columns)
    
    for column, (lower_bound, upper_bound) in limits_dict.items():
        # Winsorizar: Reemplazar valores menores que el límite inferior y mayores que el límite superior
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        
    return df
```


```python
# Defino las columnas a winsorizar: Veo que hay outliers en todas las columnas previamente seleccionadas (puntos fuera de los "bigotes") salvo en "day"
columnas_wins = ['age', 'balance', 'duration','campaign','pdays','previous']
```


```python
# Aplico winsorización a las columnas especificadas
train_winsorized = winsorize_data(train, columnas_wins)
```


```python
# Comprobamos que no haya outliers
for column in columnas_wins:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[column])
    plt.xlabel(column)
    plt.grid(True)
    plt.show() # No se detectan outliers
```


    
![png](output_35_0.png)
    



    
![png](output_35_1.png)
    



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    



    
![png](output_35_4.png)
    



    
![png](output_35_5.png)
    



```python
# Obtener una descripción estadística de las variables númericas del dataset `train`
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
      <td>8929.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.146545</td>
      <td>0.015007</td>
      <td>1116.586628</td>
      <td>0.471833</td>
      <td>0.128234</td>
      <td>15.598947</td>
      <td>353.191903</td>
      <td>2.251092</td>
      <td>7.123362</td>
      <td>0.483929</td>
      <td>0.471721</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.659209</td>
      <td>0.121588</td>
      <td>1347.587217</td>
      <td>0.499234</td>
      <td>0.334369</td>
      <td>8.401288</td>
      <td>285.539393</td>
      <td>1.531445</td>
      <td>14.037123</td>
      <td>0.890477</td>
      <td>0.499228</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-2248.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>138.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>545.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>254.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>1699.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>495.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74.500000</td>
      <td>1.000000</td>
      <td>4067.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>1030.500000</td>
      <td>6.000000</td>
      <td>31.500000</td>
      <td>2.500000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Comentarios

Luego de aplicar la winsorización , puedo comprobar gráficamente y estadisticamente que se hizo una correcta gestión de los valores atípcos al reemplazarlos con los valores mas cercanos. Por ejemplo, al inicio, "balance" tenía una std de 3225 y ahora tiene 1347

# 8 - Identificación de características númericas relevantes en "train"


```python
## Matriz de correlación para variables numéricas
train_numeric = train.select_dtypes(include=['number'])

corr_matrix = train_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de correlación")
plt.show()
```


    
![png](output_39_0.png)
    


### Comentarios

De la matriz puedo ver que mi variable objetivo no tiene relaciones fuerte con el resto de las variables, sin embargo encuentro que hay:
- una correlación lineal positiva moderada entre "loan" y "default"
- una correlación lineal negativa moderada entre "balance" y "default"
- una correlación lineal negativa debil entre "pdays", "previous", "deposit" y "default".

Por lo que las caracteristicas númericas seleccionadas como relevantes serán 'balance', 'loan','pdays','previous' y'deposit'

   


```python
# Elimino las variables numericas irrelevantes de mi dataset:

columnas_a_eliminar = ['age','housing', 'day', 'duration', 'campaign']
train = df.drop(columns=columnas_a_eliminar)

train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.015051</td>
      <td>1528.538524</td>
      <td>0.130801</td>
      <td>51.330407</td>
      <td>0.832557</td>
      <td>0.473840</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.121761</td>
      <td>3225.413326</td>
      <td>0.337198</td>
      <td>108.758282</td>
      <td>2.292007</td>
      <td>0.499338</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-6847.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>122.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>550.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1708.000000</td>
      <td>0.000000</td>
      <td>20.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>81204.000000</td>
      <td>1.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 9 - Codificación de variables categóricas en "train"

### Comentarios

Para la codificación de las variables categóricas que del dataset "train" utilizaré el método One-Hot Encoding ya que no existe un orden inherente en las categorías. Usare el mismo método para ambos datasets, identificando primero las variables categóricas y viendo si tienen demasiados valores únicos que signifique alguna modificación adicional, como una agrupación.



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11162 entries, 0 to 11161
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   age        11162 non-null  float64
     1   job        11162 non-null  object 
     2   marital    11162 non-null  object 
     3   education  11162 non-null  object 
     4   default    11162 non-null  int32  
     5   balance    11162 non-null  int64  
     6   housing    11162 non-null  int32  
     7   loan       11162 non-null  int32  
     8   contact    11162 non-null  object 
     9   day        11162 non-null  int64  
     10  month      11162 non-null  object 
     11  duration   11162 non-null  int64  
     12  campaign   11162 non-null  int64  
     13  pdays      11162 non-null  int64  
     14  previous   11162 non-null  int64  
     15  poutcome   11162 non-null  object 
     16  deposit    11162 non-null  int32  
    dtypes: float64(1), int32(4), int64(6), object(6)
    memory usage: 1.3+ MB
    


```python
# Identifico las variables categóricas
categoricas = train.select_dtypes(include=['object']).columns
print("Variables categóricas en train:", categoricas)
print()

# Veo los valores únicos de las variables categóricas
for col in categoricas:
    print(f"Valores únicos en '{col}':", train[col].unique())
```

    Variables categóricas en train: Index(['job', 'marital', 'education', 'contact', 'month', 'poutcome'], dtype='object')
    
    Valores únicos en 'job': ['admin.' 'technician' 'services' 'management' 'retired' 'blue-collar'
     'unemployed' 'entrepreneur' 'housemaid' 'unknown' 'self-employed'
     'student']
    Valores únicos en 'marital': ['married' 'single' 'divorced']
    Valores únicos en 'education': ['secondary' 'tertiary' 'primary' 'unknown']
    Valores únicos en 'contact': ['unknown' 'cellular' 'telephone']
    Valores únicos en 'month': ['may' 'jun' 'jul' 'aug' 'oct' 'nov' 'dec' 'jan' 'feb' 'mar' 'apr' 'sep']
    Valores únicos en 'poutcome': ['unknown' 'other' 'failure' 'success']
    


```python
#Reagrupaciones
# Diccionario de mapeo
month_to_season = {
    'mar': 'Primavera', 'apr': 'Primavera', 'may': 'Primavera',
    'jun': 'Verano', 'jul': 'Verano', 'aug': 'Verano',
    'sep': 'Otoño', 'oct': 'Otoño', 'nov': 'Otoño',
    'dec': 'Invierno', 'jan': 'Invierno', 'feb': 'Invierno'
}

# Reemplazar los valores de 'month' con las estaciones correspondientes
train['month'] = train['month'].map(month_to_season)

# Diccionario de mapeo
job_to_skill = {
    'admin.': 'Cualificado', 'technician': 'Cualificado', 'services': 'No cualificado',
    'management': 'Cualificado', 'retired': 'No cualificado', 'blue-collar': 'No cualificado',
    'unemployed': 'No cualificado', 'entrepreneur': 'Cualificado', 'housemaid': 'No cualificado',
    'unknown': 'Desconocido', 'self-employed': 'Cualificado', 'student': 'No cualificado'
}

# Reemplazar los valores de 'job' con las categorías correspondientes
train['job'] = train['job'].map(job_to_skill)


# Verifico los cambios:
for col in categoricas:
    print(f"Valores únicos en '{col}':", train[col].unique())
```

    Valores únicos en 'job': ['Cualificado' 'No cualificado' 'Desconocido']
    Valores únicos en 'marital': ['married' 'single' 'divorced']
    Valores únicos en 'education': ['secondary' 'tertiary' 'primary' 'unknown']
    Valores únicos en 'contact': ['unknown' 'cellular' 'telephone']
    Valores únicos en 'month': ['Primavera' 'Verano' 'Otoño' 'Invierno']
    Valores únicos en 'poutcome': ['unknown' 'other' 'failure' 'success']
    


```python
# Verificar los cambios en train
print(train[['month']].head())
print(train[['job']].head())
```

           month
    0  Primavera
    1  Primavera
    2  Primavera
    3  Primavera
    4  Primavera
                  job
    0     Cualificado
    1     Cualificado
    2     Cualificado
    3  No cualificado
    4     Cualificado
    


```python
#Codifico las 6 variables categoricas para el set de entrenamiento:
train = pd.get_dummies(train, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], dtype='int', drop_first=True) # elimino la primer columna para evitar multicolinealidad

# visualizo el nuevo dataset
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>job_Desconocido</th>
      <th>job_No cualificado</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>...</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>month_Otoño</th>
      <th>month_Primavera</th>
      <th>month_Verano</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2343</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1270</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2476</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>184</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Comentarios
Con la reagrupación solo quedaron 21 columnas es algo gestionable para mi modelo por lo cual no haré mas modificaciones a las variables categoricas. 

# 10 - Estandarización de variables númericas en "train"

### Comentarios

Aplicar estandarización númericas para que no haya problemas de escala. Lo hare para todas las variables menos para la variable objetivo porque quiero mantener la interpretación en su escala original.



```python
#Situación antes de estandarizar
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>job_Desconocido</th>
      <th>job_No cualificado</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>...</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>month_Otoño</th>
      <th>month_Primavera</th>
      <th>month_Verano</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>...</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.015051</td>
      <td>1528.538524</td>
      <td>0.130801</td>
      <td>51.330407</td>
      <td>0.832557</td>
      <td>0.473840</td>
      <td>0.006271</td>
      <td>0.415338</td>
      <td>0.569253</td>
      <td>0.315087</td>
      <td>...</td>
      <td>0.330138</td>
      <td>0.044436</td>
      <td>0.069342</td>
      <td>0.210177</td>
      <td>0.148181</td>
      <td>0.360419</td>
      <td>0.381204</td>
      <td>0.048110</td>
      <td>0.095951</td>
      <td>0.745924</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.121761</td>
      <td>3225.413326</td>
      <td>0.337198</td>
      <td>108.758282</td>
      <td>2.292007</td>
      <td>0.499338</td>
      <td>0.078946</td>
      <td>0.492802</td>
      <td>0.495203</td>
      <td>0.464571</td>
      <td>...</td>
      <td>0.470284</td>
      <td>0.206072</td>
      <td>0.254047</td>
      <td>0.407453</td>
      <td>0.355296</td>
      <td>0.480144</td>
      <td>0.485704</td>
      <td>0.214008</td>
      <td>0.294537</td>
      <td>0.435360</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-6847.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>122.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>550.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1708.000000</td>
      <td>0.000000</td>
      <td>20.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>81204.000000</td>
      <td>1.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>




```python
# Especificar las columnas numéricas a estandarizar, son solo 2 porque el resto de las variables númericas ya son binarias.
columnas_stand = ['balance','pdays']

# Inicializar el escalador
scaler = StandardScaler()

# Aplicar la estandarización solo a las columnas numéricas
train[columnas_stand] = scaler.fit_transform(train[columnas_stand])


```


```python
# Verificar estandarización. Recordar que no se estandariza la variable objetivo porque queremos mantener la interpretación en su escala original.
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>job_Desconocido</th>
      <th>job_No cualificado</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>...</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>month_Otoño</th>
      <th>month_Primavera</th>
      <th>month_Verano</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11162.000000</td>
      <td>1.116200e+04</td>
      <td>11162.000000</td>
      <td>1.116200e+04</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>...</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.015051</td>
      <td>9.961372e-17</td>
      <td>0.130801</td>
      <td>-8.213994e-15</td>
      <td>0.832557</td>
      <td>0.473840</td>
      <td>0.006271</td>
      <td>0.415338</td>
      <td>0.569253</td>
      <td>0.315087</td>
      <td>...</td>
      <td>0.330138</td>
      <td>0.044436</td>
      <td>0.069342</td>
      <td>0.210177</td>
      <td>0.148181</td>
      <td>0.360419</td>
      <td>0.381204</td>
      <td>0.048110</td>
      <td>0.095951</td>
      <td>0.745924</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.121761</td>
      <td>1.000045e+00</td>
      <td>0.337198</td>
      <td>1.000045e+00</td>
      <td>2.292007</td>
      <td>0.499338</td>
      <td>0.078946</td>
      <td>0.492802</td>
      <td>0.495203</td>
      <td>0.464571</td>
      <td>...</td>
      <td>0.470284</td>
      <td>0.206072</td>
      <td>0.254047</td>
      <td>0.407453</td>
      <td>0.355296</td>
      <td>0.480144</td>
      <td>0.485704</td>
      <td>0.214008</td>
      <td>0.294537</td>
      <td>0.435360</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-2.596850e+00</td>
      <td>0.000000</td>
      <td>-4.811841e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>-4.360996e-01</td>
      <td>0.000000</td>
      <td>-4.811841e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>-3.033975e-01</td>
      <td>0.000000</td>
      <td>-4.811841e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>5.564233e-02</td>
      <td>0.000000</td>
      <td>-2.811903e-01</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>2.470351e+01</td>
      <td>1.000000</td>
      <td>7.380638e+00</td>
      <td>58.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>



# 11 - Aplico las transformaciones en "test"

### Comentarios

Detallo las diferentes etapas y lo que hare:

- Imputación de Valores Faltantes: Se hizo previo a la separacion de datasets

- Gestión de outliers: aplico el mismo método de Winsorizacion sin necesidad de detección.

- Selección de Características Relevantes: mismo esquema

- Codificación de Variables Categóricas: mismo esquema

- Estandarización: mismo esquema


```python
#Situación inicial
test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.00000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.251232</td>
      <td>0.015226</td>
      <td>1463.669055</td>
      <td>0.47828</td>
      <td>0.141066</td>
      <td>15.894313</td>
      <td>370.142409</td>
      <td>2.392297</td>
      <td>52.338558</td>
      <td>0.862965</td>
      <td>0.482311</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.860337</td>
      <td>0.122479</td>
      <td>2829.226127</td>
      <td>0.49964</td>
      <td>0.348167</td>
      <td>8.495863</td>
      <td>346.320067</td>
      <td>2.446044</td>
      <td>109.589447</td>
      <td>2.256221</td>
      <td>0.499799</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-6847.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>130.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>137.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>563.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>259.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>1740.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>501.000000</td>
      <td>3.000000</td>
      <td>56.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>37127.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>3284.000000</td>
      <td>43.000000</td>
      <td>842.000000</td>
      <td>37.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Gestión de outliers (Winsorización)
# Aplicar winsorización a las columnas especificadas
test_winsorized = winsorize_data(test, columnas_wins)


```


```python
# Obtengo una descripción estadística del DataFrame `test`
test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.00000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
      <td>2233.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.182714</td>
      <td>0.015226</td>
      <td>1119.070309</td>
      <td>0.47828</td>
      <td>0.141066</td>
      <td>15.894313</td>
      <td>352.058665</td>
      <td>2.188536</td>
      <td>31.329154</td>
      <td>0.491267</td>
      <td>0.482311</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.643015</td>
      <td>0.122479</td>
      <td>1355.534132</td>
      <td>0.49964</td>
      <td>0.348167</td>
      <td>8.495863</td>
      <td>281.676480</td>
      <td>1.478585</td>
      <td>56.690294</td>
      <td>0.901035</td>
      <td>0.499799</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-2285.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>130.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>137.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>563.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>259.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>1740.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>501.000000</td>
      <td>3.000000</td>
      <td>56.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74.500000</td>
      <td>1.000000</td>
      <td>4155.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>1047.000000</td>
      <td>6.000000</td>
      <td>141.500000</td>
      <td>2.500000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Identificación de Características Relevantes:

# Elimino las variables numericas irrelevantes de mi dataset:

columnas_a_eliminar = ['age','housing', 'day', 'duration', 'campaign']
test = df.drop(columns=columnas_a_eliminar)

test.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
      <td>11162.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.015051</td>
      <td>1528.538524</td>
      <td>0.130801</td>
      <td>51.330407</td>
      <td>0.832557</td>
      <td>0.473840</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.121761</td>
      <td>3225.413326</td>
      <td>0.337198</td>
      <td>108.758282</td>
      <td>2.292007</td>
      <td>0.499338</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-6847.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>122.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>550.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1708.000000</td>
      <td>0.000000</td>
      <td>20.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>81204.000000</td>
      <td>1.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reagrupaciones
# Reemplazar los valores de 'month' con las estaciones correspondientes
test['month'] = test['month'].map(month_to_season)

# Reemplazar los valores de 'job' con las categorías correspondientes
test['job'] = test['job'].map(job_to_skill)


# Verifico los cambios:
for col in categoricas:
    print(f"Valores únicos en '{col}':", test[col].unique())
    
# Codificación de Variables Categóricas

#Codifico las 6 variables categoricas para el set de test:
test = pd.get_dummies(test, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], dtype='int', drop_first=True) # elimino la primer columna para evitar multicolinealidad


# visualizo el nuevo dataset
test.head()
```

    Valores únicos en 'job': ['Cualificado' 'No cualificado' 'Desconocido']
    Valores únicos en 'marital': ['married' 'single' 'divorced']
    Valores únicos en 'education': ['secondary' 'tertiary' 'primary' 'unknown']
    Valores únicos en 'contact': ['unknown' 'cellular' 'telephone']
    Valores únicos en 'month': ['Primavera' 'Verano' 'Otoño' 'Invierno']
    Valores únicos en 'poutcome': ['unknown' 'other' 'failure' 'success']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>job_Desconocido</th>
      <th>job_No cualificado</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>...</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>month_Otoño</th>
      <th>month_Primavera</th>
      <th>month_Verano</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2343</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1270</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2476</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>184</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Estandarizar


# Aplico la estandarización solo a las columnas numéricas
columnas_stand = ['balance','pdays']
test[columnas_stand] = scaler.transform(test[columnas_stand])


# Verifico estandarización. Recordar que no se estandariza la variable objetivo
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>loan</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>job_Desconocido</th>
      <th>job_No cualificado</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>...</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>month_Otoño</th>
      <th>month_Primavera</th>
      <th>month_Verano</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.252525</td>
      <td>0</td>
      <td>-0.481184</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>-0.459974</td>
      <td>0</td>
      <td>-0.481184</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-0.080160</td>
      <td>0</td>
      <td>-0.481184</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.293762</td>
      <td>0</td>
      <td>-0.481184</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-0.416876</td>
      <td>0</td>
      <td>-0.481184</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# 12 - Entreno el modelo


Luego de haber realizado todo el procesamiento de datos, entiendo que mis datasets "train" y "test" están en un formato adecuado para el modelado, garantizando calidad y consistencia.


```python
# Especificar las columnas para características y variable objetivo
feature_columns = ['balance', 'loan', 'pdays', 'previous', 'deposit']
target_column = 'default'
```


```python
#Verifico los valores
print(type(feature_columns))
print(feature_columns)
print()
print(type(target_column))
print(target_column)
```

    <class 'list'>
    ['balance', 'loan', 'pdays', 'previous', 'deposit']
    
    <class 'str'>
    default
    


```python
# Separo características (X) y variable objetivo (y) en el conjunto de entrenamiento. 
X_train = train[feature_columns]
y_train = train[target_column]

# Separo características (X) y variable objetivo (y) en el conjunto de prueba
X_test = test[feature_columns]
y_test = test[target_column]


```


```python
# Creo el modelo de Regresión Logística
modelo = LogisticRegression().fit(X_train, y_train)
np.set_printoptions(suppress=True)

print(modelo.predict(X_test))
print()
print(modelo.predict_proba(X_test))
print()
print(modelo.score(X_test, y_test))
print()
print(modelo.intercept_, modelo.coef_)
```

    [0 0 0 ... 0 0 0]
    
    [[0.99911016 0.00088984]
     [0.97822318 0.02177682]
     [0.99601286 0.00398714]
     ...
     [0.97184707 0.02815293]
     [0.98780323 0.01219677]
     [0.97071416 0.02928584]]
    
    0.9846801648450099
    
    [-5.66774523] [[-4.51746843  0.75240902 -0.05376684 -0.31205271 -0.24093429]]
    


```python
# Realiza predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcula las métricas
cm = confusion_matrix(y_test, y_pred) # Matriz de confusión
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary') 
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Imprime las métricas
print("Confusion Matrix:")
print(cm)
print()
print(f"Accuracy: {accuracy:.2f}")
print()
print(f"Precision: {precision:.2f}")
print()
print(f"Recall: {recall:.2f}")
print()
print(f"F1-score: {f1:.2f}")
```

    Confusion Matrix:
    [[10990     4]
     [  167     1]]
    
    Accuracy: 0.98
    
    Precision: 0.20
    
    Recall: 0.01
    
    F1-score: 0.01
    
    AUC-ROC: 0.86
    

### Interpretación de los resultados:

#### Confusion Matrix:

-> Clase negativa (no positiva):
  - 10,990 predicciones negativas correctas (True Negatives, TN).
  - 4 falsos positivos (False Positives, FP): casos predichos como positivos, pero son negativos.

-> Clase positiva:
 - 167 falsos negativos (False Negatives, FN): casos positivos reales que el modelo no detectó.
 - 1 predicción positiva correcta (True Positive, TP).

El modelo tiene una fuerte tendencia a predecir la clase negativa (mayoría) y detecta solo 1 caso positivo correctamente, lo que indica problemas para identificar la clase positiva.

####  Accuracy (0.98):

El modelo clasifica correctamente el 98% de los ejemplos totales. 
Sin embargo como mi conjunto es muy desbalanceado, esta métrica puede ser engañosa ya que la clase negativa domina (casi el 99% del total), lo que significa que el modelo puede lograr un alto "accuracy" simplemente prediciendo casi siempre la clase negativa.

#### Precision (0.20):
Solo el 20% de las predicciones positivas del modelo son correctas. El modelo solo hizo 5 predicciones positivas (TP + FP = 1 + 4 = 5), y de esas, solo 1 fue correcta (TP). Esto significa que el modelo rara vez predice la clase positiva, pero cuando lo hace, la mayoría de las veces está equivocado.

#### Recall (0.01):
El modelo identifica correctamente solo el 1% de los casos positivos reales. Hay 168 casos positivos reales (TP + FN = 1 + 167 = 168), pero solo detecta 1 como positivo (TP). Lo que indica nuevamente que el modelo no es efectivo para identificar la clase positiva.
####  F1-score (0.01):
Un valor bajo indica que el modelo tiene un balance deficiente entre precisión y recall para la clase positiva.


Como grandes cuestiones del modelo tenemos que hay una dominancia de la clase negativa, donde el modelo clasifica casi todo como negativo y una clase positiva subrepresentada, lo que afecta negativamente las métricas. 

Pareciera que el modelo actual no es efectivo para identificar la clase positiva, "default" = yes, que es exactamente lo que mas nos interesa para poder evaluar el riegro crediticio de un cliente. 









```python

```
