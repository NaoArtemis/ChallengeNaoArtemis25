'''
python2
'''

# Modules
import requests
import json

# Dati da inviare in formato JSON

# Specifica l'URL di destinazione
url = "http://127.0.0.1:5010/api/catalogo"

# Invia la richiesta HTTP POST con i dati JSON
response = requests.get(url)

# Stampa la risposta
print(response.text)