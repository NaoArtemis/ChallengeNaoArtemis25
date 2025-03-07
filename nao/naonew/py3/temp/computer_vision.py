# Questo file contiene codici temporanei che poi verranno integrati nel main.py py3
#################################
# FUNZIONI FLASK SERVER Python3 #
#################################
from flask import Flask, request, jsonify
import os
import requests
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, storage

app_py3 = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app_py3.route('/receive_video', methods=['POST'])
def receive_video():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, f"video_{int(time.time())}.mp4")
    
    file.save(filepath)
    print(f"Video ricevuto e salvato: {filepath}")
    
    # Analizza il video
    data = request.json
    color1 = data.get("color1", "orange")  # Colore predefinito
    color2 = data.get("color2", "black")   # Colore predefinito
    analyze_video(filepath, color1, color2)
    return jsonify({"message": "Video ricevuto"}), 200

def analyze_video(video_path, color1, color2):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    data = []  # Lista per memorizzare i dati

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rileva i giocatori con YOLO
        results = model(frame)
        players = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Classe 0 è 'persona' in YOLO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    players.append((x1, y1, x2, y2))

        # Filtra i giocatori in base al colore della maglia
        filtered_players = []
        for (x1, y1, x2, y2) in players:
            roi = frame[y1:y2, x1:x2]
            color = np.mean(roi, axis=(0, 1))  # Colore medio della ROI
            if is_color_close(color, color1) or is_color_close(color, color2):
                filtered_players.append((x1, y1, x2, y2, color))

        # Assegna un ID univoco (numero di maglia) e salva i dati
        for i, (x1, y1, x2, y2, color) in enumerate(filtered_players):
            player_id = i + 1  # ID univoco (numero di maglia)
            data.append([player_id, (x1 + x2) // 2, (y1 + y2) // 2, color])

    # Salva i dati in un file CSV
    csv_path = os.path.join(UPLOAD_FOLDER, f"data_{int(time.time())}.csv")
    df = pd.DataFrame(data, columns=["ID Giocatore", "Posizione X", "Posizione Y", "Colore"])
    df.to_csv(csv_path, index=False)

    # Carica il video e il CSV su Firebase
    upload_to_firebase(video_path, f"video_{int(time.time())}.mp4")
    upload_to_firebase(csv_path, f"data_{int(time.time())}.csv")

def is_color_close(color, target_color):
    # Implementa la logica per confrontare i colori
    return True  # Placeholder

@app_py3.route('/start_recording', methods=['POST'])
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

@app_py3.route('/stop_recording', methods=['POST'])
def stop_recording():

    try:
        response = requests.post("http://localhost:5000/stop_recording")
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": f"Errore durante la richiesta di stop: {e}"}), 500

@app_py3.route('/get_video', methods=['POST'])
def get_video():
    try:
        response = requests.post("http://localhost:5000/get_video")
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": f"Errore durante la richiesta del video: {e}"}), 500

if __name__ == '__main__':
    app_py3.run(port=5001, debug=True)