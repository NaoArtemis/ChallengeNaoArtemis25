'''
python2
'''

# Modules
import requests
import json

# Dati da inviare in formato JSON
data = {
        "det": "send_det",
        "age": "send_age",
        "gen": "send_gen",
        "emo_dom": "send_emo_dom",
        "emo": "emotions",
        "att": "attention"
};

# Specifica l'URL di destinazione
url = "http://127.0.0.1:5010/api/morphcast"

# Invia la richiesta HTTP POST con i dati JSON
response = requests.post(url, json=data)

# Stampa la risposta
print(response.text)