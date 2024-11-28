# Ejemplo de documentación del Proceso de Recolección de Datos para el Proyecto de Predicción de Incumplimiento de Credito en el Banco

## 1. Fuentes

**Identificación de Fuentes:**
- Base de datos interna del CRM

**Descripción de las Fuentes:**
- La base de datos CRM contiene los registros bancarios y la campaña de marketing telefónico. Se recopilan variables demográficas, financieras, y de historial de contacto.

## 2. Métodos de recolección de datos

**Procedimientos y Herramientas:**
- Se exporta en formato CSV a un repositorio de GitHub. Esta tarea la realiza el equipo de análisis de datos del Banco.

**Frecuencia de Recolección:**
- Los datos se recolectan al finalizar la campaña de marketing.

**Scripts de Descarga:**

```python

importe pandas as pd

# Cargar el archivo 
csv_url = https://github.com/ITACADEMYprojectes/projecteML/blob/e8d1aab0a24ddf55af9dfd9e83b1ea79e34c1af9/bank_dataset.CSV
df = pd.read_csv(csv_url)

# Exploración inicial
print(df.info())

````

## 3. Formato y Estructura de los Datos

**Tipo de Datos:**
- Numéricos: `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`
- Binarios: `default`, `housing`, `loan`, `deposit`
- Categórico: `job`, `marital`, `education`,  `contact`, `poutcome`
- Fecha: `day`, `month`

**Formato de Almacenamiento:**
- Datos tabulares almacenados en archivo csv de 17 columnas.

## 4. Limitaciones de los datos

- Valores nulos en las columnas `age`, `marital` y `education`
- Diferentes espacios temporal: Las campañas se hizo en distintas fechas por lo que puede generar un poco de distorsion en los datos comparativos de los clientes.

## 5. Consideraciones sobre Datos Sensibles

**Tipo de Datos Sensibles:**
- Información Personal Identificable (PII): -
- Información Demográfica Sensible: `age`,  `marital`, `education`
- Información Financiera Sensible: `balance`, `default`, `housing`, `loan`, `deposit`
- Datos Comportamentales Sensibles: `duration`

**Medidas de Protección:**
- **Anonimización y Pseudonimización:**
 - Se aplica agrupacion en intervalos para `age`y se hacen generalizaciones para `marital`y `education`
- **Acceso Restringido:**
 - Acceso a datos sensibles restringido sólo a personal autorizado.
- **Cumplimiento de Regulaciones:**
 - Cumplimiento con la legislación aplicable (RGDP en la UE)


```python

```
