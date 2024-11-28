## Exercici 1. Consumir una API

***Consumo la API de Aplha Vantage para obtener precios una acción en una fecha determinada***


```python
# Importo libreria "requests"
import requests

# Defino el símbolo de la acción (por ejemplo, F para Ford) y la fecha para la cual quiero el precio
symbol = "F"
target_date = "2024-11-25"

# Indico la URL de la cual voy a consumir la API
url = f"https://www.alphavantage.co/query"

# Defino los parámetros de la solicitud para la función de TIME_SERIES_DAILY (precios diarios)
params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "apikey": api_key,
    "outputsize": "compact" 
}

# Realizo la solicitud GET
response = requests.get(url, params=params)

# Verifico si la solicitud fue exitosa
if response.status_code == 200:
    data = response.json()

    # Extraigo la serie temporal (precios diarios)
    time_series = data.get("Time Series (Daily)", {})

    # Busco los precios en la fecha deseada
    if target_date in time_series:
        values = time_series[target_date]
        print(f"Datos para {symbol} en la fecha {target_date}:")
        print(f"  Apertura: {values['1. open']} USD")
        print(f"  Máximo: {values['2. high']} USD")
        print(f"  Mínimo: {values['3. low']} USD")
        print(f"  Cierre: {values['4. close']} USD")
        print(f"  Volumen: {values['5. volume']}")
    else:
        print(f"No se encontraron datos para la fecha {target_date}.")
else:
    print(f"Error al llamar a la API: {response.text}")
```

    Datos para F en la fecha 2024-11-25:
      Apertura: 11.3000 USD
      Máximo: 11.5200 USD
      Mínimo: 11.2799 USD
      Cierre: 11.4000 USD
      Volumen: 63469313
    

## Exercici 2. Obtenir dades amb Web Scraping

***Hago web scraping en una pagina de frases ("http://quotes.toscrape.com/") y elijo solo "scrapear" las de un autor especifico (ej:Albert Einstein) y si no existe que me diga "autor no encontrado"***


```python
# Importo libreria "requests" y "BeautiflSoup" de "bs4"
import requests
from bs4 import BeautifulSoup

# Indico la URL donde voy a hacer el "web scraping"
url = "http://quotes.toscrape.com/"

# Realizo la solicitud GET 
response = requests.get(url)

# Verifico si la respuesta fue exitosa
if response.status_code == 200:
    # Parseo el contenido HTML de la página
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extraigo las citas y los autores
    quotes = soup.find_all("div", class_="quote")
    for quote in quotes:
        # Extraer la cita
        text_element = quote.find("span", class_="text")
        text = text_element.get_text() if text_element else "Cita no encontrada"
        
        # Extraer el autor (verificar que existe)
        author_element = quote.find("small", class_="author")
        author = author_element.get_text() if author_element else "Autor no encontrado"
        
        # Filtro solo las citas de Albert Einstein
        if author == "Albert Einstein":
            print(f"Cita: {text}")
            print(f"Autor: {author}")
            print("-" * 80)
        else:
            print("Autor no encontrado")
        break
else:
    print(f"Error al acceder al sitio web: {response.text}")
```

    Cita: “The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”
    Autor: Albert Einstein
    --------------------------------------------------------------------------------
    


```python

```
