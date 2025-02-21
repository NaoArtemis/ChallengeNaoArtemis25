'''
python2
'''

# Modules
import requests
import json

# Dati da inviare in formato JSON
data = 2

# Specifica l'URL di destinazione
url = "http://127.0.0.1:5010/api/descrizione_prodotto/" + str(data)

# Invia la richiesta HTTP POST con i dati JSON
response = requests.get(url, json=data)

# Stampa la risposta
print(response.text)