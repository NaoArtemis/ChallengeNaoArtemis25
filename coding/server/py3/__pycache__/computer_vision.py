from flask import Flask, request, jsonify
import os
import requests
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, storage
import time

app = Flask(__name__)
UPLOAD_FOLDER = "vid"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/receive_video', methods=['POST'])
def receive_video():
    if 'file' not in request.files:
        return jsonify({"error": "Nessun file fornito"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, f"video_{int(time.time())}.mp4")
    
    try:
        file.save(filepath)
        print(f"Video ricevuto e salvato: {filepath}")
    except Exception as e:
        return jsonify({"error": f"Errore durante il salvataggio del file: {e}"}), 500
    
    data = request.json
    color1 = data.get("color1", "orange")  # Colore predefinito
    color2 = data.get("color2", "black")   # Colore predefinito
    
    try:
        # Chiamata alla funzione analyze_video senza analisi effettiva
        analyze_video(filepath, color1, color2)
        return jsonify({"message": "Video ricevuto"}), 200
    except Exception as e:
        return jsonify({"error": f"Errore durante l'elaborazione del video: {e}"}), 500

def analyze_video(video_path, color1, color2):
    """
    Funzione vuota per l'analisi del video.
    Puoi aggiungere qui la logica di analisi in futuro, se necessario.
    """
    print(f"Video salvato in: {video_path}")
    print(f"Colori ricevuti: {color1}, {color2}")
    # Nessuna analisi viene eseguita

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.json
    nao_ip = data.get("nao_ip")
    nao_port = data.get("nao_port")

    if not nao_ip or not nao_port:
        return jsonify({"error": "nao_ip e nao_port sono obbligatori"}), 400

    try:
        response = requests.post("http://localhost:5000/start_recording", json={"nao_ip": nao_ip, "nao_port": nao_port})
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": f"Errore durante la richiesta di avvio: {e}"}), 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        response = requests.post("http://localhost:5000/stop_recording")
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": f"Errore durante la richiesta di stop: {e}"}), 500

@app.route('/get_video', methods=['POST'])
def get_video():
    try:
        response = requests.post("http://localhost:5000/get_video")
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": f"Errore durante la richiesta del video: {e}"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)