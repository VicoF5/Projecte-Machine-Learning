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
    Requirement already satisfied: numpy>=1.16.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.21.5)
    Requirement already satisfied: tqdm>=4.43.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (4.64.1)
    Requirement already satisfied: pandas!=1.0.0,!=1.0.1,!=1.0.2,>=0.25.3 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.4.4)
    Requirement already satisfied: matplotlib>=3.1.3 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (3.5.2)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (1.9.1)
    Requirement already satisfied: importlib-resources>=1.2.0 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (6.4.5)
    Requirement already satisfied: jinja2>=2.11.1 in c:\users\victo\anaconda3\lib\site-packages (from sweetviz) (2.11.3)
    Requirement already satisfied: zipp>=3.1.0 in c:\users\victo\anaconda3\lib\site-packages (from importlib-resources>=1.2.0->sweetviz) (3.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\victo\anaconda3\lib\site-packages (from jinja2>=2.11.1->sweetviz) (2.0.1)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (2.8.2)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (9.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (0.11.0)
    Requirement already satisfied: packaging>=20.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (21.3)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (3.0.9)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.1.3->sweetviz) (1.4.2)
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



# 6 - Gestión del desbalance de clases

### Comentarios:

Como ya se podría prever en los primeros análisis existe un gran desbalance en la variable "default". Luego de volver a reflexionar sobre el trabajo, dado que la cantidad de "Trues" de la variable "default" en el dataset (168/11162 = 1.5%) es muy pequeña decidí la técnica de sobremuestreo sobre la misma, pasando de 168 a 10994 registros.


```python
# Distribución de la variable objetivo "default"
print(df['default'].value_counts(normalize=True))

# Visualización de la distribución
sns.countplot(data=df, x='default')
plt.title("Distribución de la variable objetivo")
plt.show()
```

    0    0.984949
    1    0.015051
    Name: default, dtype: float64
    


    
![png](output_23_1.png)
    



```python
#Creación de los sets

default_no= df[df['default'] == 0]
default_yes= df[df['default'] == 1]

print(default_no.count())

print(default_yes.count())
```

    age          10994
    job          10994
    marital      10994
    education    10994
    default      10994
    balance      10994
    housing      10994
    loan         10994
    contact      10994
    day          10994
    month        10994
    duration     10994
    campaign     10994
    pdays        10994
    previous     10994
    poutcome     10994
    deposit      10994
    dtype: int64
    age          168
    job          168
    marital      168
    education    168
    default      168
    balance      168
    housing      168
    loan         168
    contact      168
    day          168
    month        168
    duration     168
    campaign     168
    pdays        168
    previous     168
    poutcome     168
    deposit      168
    dtype: int64
    


```python
# Aplicación de técnica de sobremuestreo:

sobremuestreo_yes = default_yes.sample(n=10994, replace=True, random_state=0)
print(sobremuestreo_yes.count())

```

    age          10994
    job          10994
    marital      10994
    education    10994
    default      10994
    balance      10994
    housing      10994
    loan         10994
    contact      10994
    day          10994
    month        10994
    duration     10994
    campaign     10994
    pdays        10994
    previous     10994
    poutcome     10994
    deposit      10994
    dtype: int64
    


```python
# comprobación datasets balanceados

print("Registros False:", default_no.shape[0])

print("Registros True:", sobremuestreo_yes.shape[0])
```

    Registros False: 10994
    Registros True: 10994
    


```python
#Creo el nuevo dataframe sobremuestreado
df = pd.concat([sobremuestreo_yes, default_no])
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
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.599463</td>
      <td>0.500000</td>
      <td>739.390986</td>
      <td>0.496998</td>
      <td>0.232445</td>
      <td>16.196380</td>
      <td>358.819265</td>
      <td>2.848235</td>
      <td>35.320584</td>
      <td>0.506140</td>
      <td>0.393396</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.913957</td>
      <td>0.500011</td>
      <td>2499.290136</td>
      <td>0.500002</td>
      <td>0.422401</td>
      <td>8.524699</td>
      <td>317.319084</td>
      <td>3.361692</td>
      <td>96.776847</td>
      <td>1.782127</td>
      <td>0.488515</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-6847.000000</td>
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
      <td>-4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>132.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.500000</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>246.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>762.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>513.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95.000000</td>
      <td>1.000000</td>
      <td>81204.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>3881.000000</td>
      <td>63.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribución de la variable objetivo "default"
print(df['default'].value_counts(normalize=True))

# Visualización de la distribución
sns.countplot(data=df, x='default')
plt.title("Distribución de la variable objetivo")
plt.show()
```

    1    0.5
    0    0.5
    Name: default, dtype: float64
    


    
![png](output_28_1.png)
    


# 7 - División del dataset en "train" y "test"

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

    Tamaño del conjunto de entrenamiento (train): (17590, 17)
    Tamaño del conjunto de prueba (test): (4398, 17)
    


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
    1    0.5
    0    0.5
    Name: default, dtype: float64
    
    Proporción de clases en el conjunto de entrenamiento:
    1    0.5
    0    0.5
    Name: default, dtype: float64
    
    Proporción de clases en el conjunto de prueba:
    1    0.5
    0    0.5
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
      <th>5354</th>
      <td>27.0</td>
      <td>housemaid</td>
      <td>married</td>
      <td>secondary</td>
      <td>1</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>jul</td>
      <td>59</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1555</th>
      <td>45.0</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>1</td>
      <td>-443</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>20</td>
      <td>apr</td>
      <td>691</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10234</th>
      <td>53.0</td>
      <td>admin.</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>7</td>
      <td>may</td>
      <td>176</td>
      <td>1</td>
      <td>322</td>
      <td>5</td>
      <td>other</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1116</th>
      <td>55.0</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>1</td>
      <td>-308</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>2</td>
      <td>feb</td>
      <td>781</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4921</th>
      <td>36.0</td>
      <td>services</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>cellular</td>
      <td>15</td>
      <td>jul</td>
      <td>687</td>
      <td>7</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
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
      <th>8279</th>
      <td>56.0</td>
      <td>blue-collar</td>
      <td>divorced</td>
      <td>primary</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>18</td>
      <td>jul</td>
      <td>304</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2142</th>
      <td>26.0</td>
      <td>student</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>26</td>
      <td>may</td>
      <td>445</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8980</th>
      <td>36.0</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>1</td>
      <td>-508</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>17</td>
      <td>apr</td>
      <td>832</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7754</th>
      <td>39.0</td>
      <td>self-employed</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>1</td>
      <td>-103</td>
      <td>0</td>
      <td>1</td>
      <td>unknown</td>
      <td>5</td>
      <td>jun</td>
      <td>210</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4654</th>
      <td>32.0</td>
      <td>student</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1138</td>
      <td>0</td>
      <td>0</td>
      <td>telephone</td>
      <td>10</td>
      <td>feb</td>
      <td>402</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 8 - Detección y gestion de outliers en "train"

### Comentarios

Hago la detección y tratamiento de outliers en el dataset de entrenamiento y luego aplico directamente él metodo al dataset de prueba. Elimino del análisis mi variable objetivo, así como las que convertí en 1 y 0 anteriormente: default, loan, deposit y housing para simplificar la visualización gráfica ya que son variables binarias. 


```python
# Lista de columnas númericas para detectar outliers
columnas_outlier = ['age', 'balance', 'day', 'duration','campaign'] #pdays','previous'
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


    
![png](output_38_0.png)
    



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    



    
![png](output_38_3.png)
    



    
![png](output_38_4.png)
    



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
columnas_wins = ['age', 'balance', 'duration','campaign'] 
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


    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    



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
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
      <td>17590.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.542638</td>
      <td>0.500000</td>
      <td>377.010041</td>
      <td>0.496873</td>
      <td>0.232348</td>
      <td>16.172257</td>
      <td>348.210375</td>
      <td>2.378624</td>
      <td>35.086754</td>
      <td>0.502729</td>
      <td>0.392496</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.700959</td>
      <td>0.500014</td>
      <td>773.755964</td>
      <td>0.500004</td>
      <td>0.422342</td>
      <td>8.524241</td>
      <td>281.922811</td>
      <td>1.640641</td>
      <td>96.365141</td>
      <td>1.788736</td>
      <td>0.488320</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-1150.125000</td>
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
      <td>-3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>131.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.500000</td>
      <td>83.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>244.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>761.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>510.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72.000000</td>
      <td>1.000000</td>
      <td>1908.875000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>1078.500000</td>
      <td>6.000000</td>
      <td>854.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Comentarios

Luego de aplicar la winsorización , puedo comprobar gráficamente y estadisticamente que se hizo una correcta gestión de los valores atípcos al reemplazarlos con los valores mas cercanos. Por ejemplo, al inicio, "balance" tenía una std de 2499,29 y ahora tiene 773,76

# 9 - Identificación de características númericas relevantes en "train"


```python
## Matriz de correlación para variables numéricas
train_numeric = train.select_dtypes(include=['number'])

corr_matrix = train_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de correlación")
plt.show()
```


    
![png](output_47_0.png)
    


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
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>739.390986</td>
      <td>0.232445</td>
      <td>35.320584</td>
      <td>0.506140</td>
      <td>0.393396</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500011</td>
      <td>2499.290136</td>
      <td>0.422401</td>
      <td>96.776847</td>
      <td>1.782127</td>
      <td>0.488515</td>
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
      <td>-4.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>762.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
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
    </tr>
  </tbody>
</table>
</div>



# 10 - Codificación de variables categóricas en "train"

### Comentarios

Para la codificación de las variables categóricas que del dataset "train" utilizaré el método One-Hot Encoding ya que no existe un orden inherente en las categorías. Usare el mismo método para ambos datasets, identificando primero las variables categóricas y viendo si tienen demasiados valores únicos que signifique alguna modificación adicional, como una agrupación.



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21988 entries, 5036 to 11161
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   age        21988 non-null  float64
     1   job        21988 non-null  object 
     2   marital    21988 non-null  object 
     3   education  21988 non-null  object 
     4   default    21988 non-null  int32  
     5   balance    21988 non-null  int64  
     6   housing    21988 non-null  int32  
     7   loan       21988 non-null  int32  
     8   contact    21988 non-null  object 
     9   day        21988 non-null  int64  
     10  month      21988 non-null  object 
     11  duration   21988 non-null  int64  
     12  campaign   21988 non-null  int64  
     13  pdays      21988 non-null  int64  
     14  previous   21988 non-null  int64  
     15  poutcome   21988 non-null  object 
     16  deposit    21988 non-null  int32  
    dtypes: float64(1), int32(4), int64(6), object(6)
    memory usage: 2.7+ MB
    


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
    
    Valores únicos en 'job': ['self-employed' 'blue-collar' 'management' 'entrepreneur' 'retired'
     'technician' 'admin.' 'housemaid' 'services' 'unemployed' 'student'
     'unknown']
    Valores únicos en 'marital': ['married' 'single' 'divorced']
    Valores únicos en 'education': ['secondary' 'unknown' 'tertiary' 'primary']
    Valores únicos en 'contact': ['cellular' 'unknown' 'telephone']
    Valores únicos en 'month': ['nov' 'jul' 'may' 'feb' 'jun' 'aug' 'sep' 'jan' 'apr' 'oct' 'dec' 'mar']
    Valores únicos en 'poutcome': ['unknown' 'failure' 'other' 'success']
    


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
    Valores únicos en 'education': ['secondary' 'unknown' 'tertiary' 'primary']
    Valores únicos en 'contact': ['cellular' 'unknown' 'telephone']
    Valores únicos en 'month': ['Otoño' 'Verano' 'Primavera' 'Invierno']
    Valores únicos en 'poutcome': ['unknown' 'failure' 'other' 'success']
    


```python
# Verificar los cambios en train
print(train[['month']].head())
print(train[['job']].head())
```

              month
    5036      Otoño
    8803     Verano
    6203  Primavera
    7938   Invierno
    328      Verano
                     job
    5036     Cualificado
    8803  No cualificado
    6203     Cualificado
    7938     Cualificado
    328      Cualificado
    


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
      <th>5036</th>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8803</th>
      <td>1</td>
      <td>67</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6203</th>
      <td>1</td>
      <td>101</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
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
      <th>7938</th>
      <td>1</td>
      <td>75</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>328</th>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
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

# 11 - Estandarización de variables númericas en "train"

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
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>...</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>739.390986</td>
      <td>0.232445</td>
      <td>35.320584</td>
      <td>0.506140</td>
      <td>0.393396</td>
      <td>0.005958</td>
      <td>0.414544</td>
      <td>0.552529</td>
      <td>0.306940</td>
      <td>...</td>
      <td>0.308577</td>
      <td>0.049254</td>
      <td>0.047071</td>
      <td>0.283245</td>
      <td>0.116882</td>
      <td>0.329225</td>
      <td>0.464572</td>
      <td>0.032700</td>
      <td>0.048708</td>
      <td>0.839776</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500011</td>
      <td>2499.290136</td>
      <td>0.422401</td>
      <td>96.776847</td>
      <td>1.782127</td>
      <td>0.488515</td>
      <td>0.076958</td>
      <td>0.492654</td>
      <td>0.497244</td>
      <td>0.461235</td>
      <td>...</td>
      <td>0.461917</td>
      <td>0.216403</td>
      <td>0.211796</td>
      <td>0.450585</td>
      <td>0.321287</td>
      <td>0.469943</td>
      <td>0.498755</td>
      <td>0.177853</td>
      <td>0.215263</td>
      <td>0.366822</td>
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
      <td>-4.000000</td>
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
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>81.000000</td>
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
      <td>1.000000</td>
      <td>762.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
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
      <td>21988.000000</td>
      <td>2.198800e+04</td>
      <td>21988.000000</td>
      <td>2.198800e+04</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>...</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>1.016652e-14</td>
      <td>0.232445</td>
      <td>2.356699e-15</td>
      <td>0.506140</td>
      <td>0.393396</td>
      <td>0.005958</td>
      <td>0.414544</td>
      <td>0.552529</td>
      <td>0.306940</td>
      <td>...</td>
      <td>0.308577</td>
      <td>0.049254</td>
      <td>0.047071</td>
      <td>0.283245</td>
      <td>0.116882</td>
      <td>0.329225</td>
      <td>0.464572</td>
      <td>0.032700</td>
      <td>0.048708</td>
      <td>0.839776</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500011</td>
      <td>1.000023e+00</td>
      <td>0.422401</td>
      <td>1.000023e+00</td>
      <td>1.782127</td>
      <td>0.488515</td>
      <td>0.076958</td>
      <td>0.492654</td>
      <td>0.497244</td>
      <td>0.461235</td>
      <td>...</td>
      <td>0.461917</td>
      <td>0.216403</td>
      <td>0.211796</td>
      <td>0.450585</td>
      <td>0.321287</td>
      <td>0.469943</td>
      <td>0.498755</td>
      <td>0.177853</td>
      <td>0.215263</td>
      <td>0.366822</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-3.035487e+00</td>
      <td>0.000000</td>
      <td>-3.753109e-01</td>
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
      <td>-2.974476e-01</td>
      <td>0.000000</td>
      <td>-3.753109e-01</td>
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
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>-2.634372e-01</td>
      <td>0.000000</td>
      <td>-3.753109e-01</td>
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
      <td>1.000000</td>
      <td>9.046380e-03</td>
      <td>0.000000</td>
      <td>-3.753109e-01</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
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
      <td>3.219572e+01</td>
      <td>1.000000</td>
      <td>8.459648e+00</td>
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



# 12 - Aplico las transformaciones en "test"

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
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.568213</td>
      <td>0.500000</td>
      <td>690.014325</td>
      <td>0.497499</td>
      <td>0.232833</td>
      <td>16.292860</td>
      <td>360.633242</td>
      <td>2.795134</td>
      <td>36.255798</td>
      <td>0.519782</td>
      <td>0.396999</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.898471</td>
      <td>0.500057</td>
      <td>2227.569689</td>
      <td>0.500051</td>
      <td>0.422685</td>
      <td>8.526815</td>
      <td>316.258356</td>
      <td>3.230386</td>
      <td>98.411804</td>
      <td>1.755577</td>
      <td>0.489331</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-6847.000000</td>
      <td>0.000000</td>
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
      <td>-15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>132.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.500000</td>
      <td>79.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>252.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>764.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>525.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>37127.000000</td>
      <td>1.000000</td>
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
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
      <td>4398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.512733</td>
      <td>0.500000</td>
      <td>367.194577</td>
      <td>0.497499</td>
      <td>0.232833</td>
      <td>16.292860</td>
      <td>352.006935</td>
      <td>2.366758</td>
      <td>36.255798</td>
      <td>0.519782</td>
      <td>0.396999</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.713767</td>
      <td>0.500057</td>
      <td>785.312464</td>
      <td>0.500051</td>
      <td>0.422685</td>
      <td>8.526815</td>
      <td>282.310153</td>
      <td>1.632860</td>
      <td>98.411804</td>
      <td>1.755577</td>
      <td>0.489331</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>-1184.250000</td>
      <td>0.000000</td>
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
      <td>-15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>132.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.500000</td>
      <td>79.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>252.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>764.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>525.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72.000000</td>
      <td>1.000000</td>
      <td>1933.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>1114.500000</td>
      <td>6.000000</td>
      <td>842.000000</td>
      <td>37.000000</td>
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
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
      <td>21988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>739.390986</td>
      <td>0.232445</td>
      <td>35.320584</td>
      <td>0.506140</td>
      <td>0.393396</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500011</td>
      <td>2499.290136</td>
      <td>0.422401</td>
      <td>96.776847</td>
      <td>1.782127</td>
      <td>0.488515</td>
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
      <td>-4.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>762.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
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
    Valores únicos en 'education': ['secondary' 'unknown' 'tertiary' 'primary']
    Valores únicos en 'contact': ['cellular' 'unknown' 'telephone']
    Valores únicos en 'month': ['Otoño' 'Verano' 'Primavera' 'Invierno']
    Valores únicos en 'poutcome': ['unknown' 'failure' 'other' 'success']
    




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
      <th>5036</th>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8803</th>
      <td>1</td>
      <td>67</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6203</th>
      <td>1</td>
      <td>101</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
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
      <th>7938</th>
      <td>1</td>
      <td>75</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>328</th>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
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
      <th>5036</th>
      <td>1</td>
      <td>-0.295847</td>
      <td>0</td>
      <td>-0.375311</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8803</th>
      <td>1</td>
      <td>-0.269039</td>
      <td>0</td>
      <td>-0.375311</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6203</th>
      <td>1</td>
      <td>-0.255435</td>
      <td>0</td>
      <td>-0.375311</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
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
      <th>7938</th>
      <td>1</td>
      <td>-0.265838</td>
      <td>1</td>
      <td>-0.375311</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>328</th>
      <td>1</td>
      <td>-0.295847</td>
      <td>0</td>
      <td>-0.375311</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# 13 - Entreno el modelo


Luego de haber realizado todo el procesamiento de datos, entiendo que mis datasets "train" y "test" están en un formato adecuado para el modelado, garantizando calidad y consistencia.

En este caso, elijo un modelo de regresión logística ya que está específicamente diseñada para problemas de clasificación binaria . Como mi objetivo es predecir una variable categórica con dos clases no = 0, yes =1, es las elección mas conveniente.

Selecciono mis "features" (variables independientes) según las matriz de correlación respecto a mi variable objetivo.


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

    [1 1 1 ... 1 1 1]
    
    [[0.4432215  0.5567785 ]
     [0.38352328 0.61647672]
     [0.39736465 0.60263535]
     ...
     [0.36827477 0.63172523]
     [0.44645817 0.55354183]
     [0.35681057 0.64318943]]
    
    0.7899308713843914
    
    [-0.67621046] [[-4.2753299   0.8415658  -0.00160041 -0.24258949 -0.36113642]]
    


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
    [[7548 3446]
     [1173 9821]]
    
    Accuracy: 0.79
    
    Precision: 0.74
    
    Recall: 0.89
    
    F1-score: 0.81
    

### Interpretación de los resultados:

Para este problema necesitamos predecir si el cliente tiene un historial de incumplimiento crediticio (yes o no) respecto al pago de un préstamo (variable objetivo: default), lo cual es problema de clasificación binaria. En estos caso un algoritimo de regresión logística fue el  adecuado ya que es un modelo simple y facilmente interpretable. Sin embargo se hizo un fuerte trabajo de pre-procesamiento de los datos para que el modelo funcione correctamente.

Dado que el costo de incumplimiento crediticio ("defuault") para la institución bancaria es alto, a continuación evaluaré el rendimiento del modelo en base a la Matriz de Confusión:
#### Confusion Matrix:

-> Clase negativa (no positiva):
  - 7548 predicciones negativas correctas (True Negatives, TN).
  - 3446 falsos positivos (False Positives, FP): casos predichos como positivos, pero son negativos.

-> Clase positiva:
 - 1173 falsos negativos (False Negatives, FN): casos positivos reales que el modelo no detectó.
 - 9821 predicción positiva correcta (True Positive, TP).

La matriz muestra que el modelo tiene un buen rendimiento global, pero con más errores al predecir la clase negativa (3446 FP) que la clase positiva (1173 FN). Esto es aceptable porque mi objetivo principal es identificar la clase positiva (default = yes) con mayor prioridad.

####  Accuracy (0.79):

El modelo clasifica correctamente el 79% de los ejemplos totales. 
Sin embargo como mi conjunto estaba muy desbalanceado, esta métrica puede ser engañosa.

#### Precision (0.74):
La precisión mide la proporción de predicciones positivas que realmente son correctas. Con un valor de 74%, el modelo es razonablemente preciso, pero el 26% de las predicciones positivas son incorrectas. Esto podría ser un problema si mi costo de los falsos positivos es alto, significando la perdida de potenciales clientes para un préstamo.

#### Recall (0.89):
El recall mide la proporción de verdaderos positivos correctamente identificados. Un recall del 89% indica que el modelo identifica bien la mayoría de las instancias de la clase positiva, lo que es crucial en casos de incumplimiento créditicio
. 
####  F1-score (0.81):
El F1-score es el promedio armónico entre precisión y recall. Con un valor de 81%, el modelo logra un buen equilibrio entre capturar la mayor parte de los positivos reales (recall) y minimizar los falsos positivos (precisión).


Luego de balancear las clases, el modelo ha mejorado su performance considerablemente. 


# 14 - Siguientes pasos

Se podría evaluar mas profundamente el modelo, analizando los umbrales de probabilidad para asi ajustar la aceptación o rechazo de clientes en función de los riesgos de incumplimiento, a través de la curva ROC y AUC. Asó como sería  útil hacer una cross-validation para verificar que el modelo es robusto frente a distintas particiones de los datos.

Claramente el dataset está muy desbalanceado, por lo que con el sobremuestro pudo haber casusado sobreajuste y una generación de datos poco realistas. A mi entender es necesario ampliar la base de datos para poder tener mas seguridad en las predicciones del modelo, ya que la variable objetivo es crucial para el negocio.
