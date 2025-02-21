'''
python2
'''

# Modules
import requests
import json

# Dati da inviare in formato JSON
data = {"id_cliente":9, "id_oggetto":1}

# Specifica l'URL di destinazione
url = "http://127.0.0.1:5010/api/carrello/" + str(data)

# Invia la richiesta HTTP POST con i dati JSON
response = requests.delete(url, json=data)

# Stampa la risposta
print(response.text)