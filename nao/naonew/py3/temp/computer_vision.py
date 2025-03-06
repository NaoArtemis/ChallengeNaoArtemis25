# Questo file contiene codici temporanei che poi verranno integrati nel main.py py3 e py2
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from flask import Flask, request, jsonify
import os
import threading
import time
import random
from naoqi import ALProxy
import requests
import firebase_admin
from firebase_admin import credentials, storage

# Inizializzazione di Firebase
cred = credentials.Certificate("path/to/your/firebase-key.json")  # Sostituisci con il percorso della tua chiave Firebase
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-bucket-name.appspot.com'  # Sostituisci con il nome del tuo bucket Firebase
})

#################################
# FUNZIONI FLASK SERVER Python2 #
#################################

app_py2 = Flask(__name__)

# Variabili globali per gestire la registrazione
is_recording = False
current_video_path = None
video_lock = threading.Lock()
last_send_time = 0

# Cartella per salvare i chunk di video
CHUNKS_FOLDER = "chunks"
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

def nao_get_image(nao_ip, nao_port):
    """
    Acquisisce un frame dalla telecamera del NAO.
    """
    video_proxy = ALProxy("ALVideoDevice", nao_ip, nao_port)

    # Configurazione della telecamera
    name_id = "video_image_" + str(random.randint(0, 100))  # Nome univoco per la connessione
    camera_id = 0  # 0 = telecamera superiore, 1 = telecamera inferiore
    resolution = 1  # 320x240 px
    color_space = 13  # RGB
    camera_fps = 15  # fps
    brightness_value = 55  # Luminosità predefinita

    video_proxy.setParameter(camera_id, 0, brightness_value)  # Luminosità

    video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, color_space, camera_fps)

    try:
        image = video_proxy.getImageRemote(video_client)
        if image:
            # Converti l'immagine in un formato utilizzabile da OpenCV
            image_width = image[0]
            image_height = image[1]
            image_data = np.frombuffer(image[6], dtype=np.uint8).reshape((image_height, image_width, 3))

            resized_image = cv2.resize(image_data, (640, 480))  # Da 320x240 a 640x480
            return resized_image
    except Exception as e:
        return None
    finally:
        video_proxy.unsubscribe(video_client)

def get_video_chunk(output_video, nao_ip, nao_port, start_time, duration=300):

    global is_recording

    # Configura il VideoWriter per salvare il video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 15, (640, 480))

    while is_recording and (time.time() - start_time) < duration:
        frame = nao_get_image(nao_ip, nao_port)
        if frame is not None:
            out.write(frame)

    out.release()

def upload_to_firebase(file_path, destination_name):

    bucket = storage.bucket()
    blob = bucket.blob(destination_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} caricato su Firebase come {destination_name}.")

@app_py2.route('/start_recording', methods=['POST'])
def start_recording():

    global is_recording, current_video_path, last_send_time

    data = request.json
    nao_ip = data.get("nao_ip")
    nao_port = data.get("nao_port")

    if not nao_ip or not nao_port:
        return jsonify({"error": "nao_ip e nao_port sono obbligatori"}), 400

    with video_lock:
        if not is_recording:
            is_recording = True
            current_video_path = os.path.join(CHUNKS_FOLDER, f"temp_chunk_{int(time.time())}.mp4")
            last_send_time = time.time()
            threading.Thread(target=record_and_send_video, args=(current_video_path, nao_ip, nao_port)).start()
            return jsonify({"message": "Registrazione avviata"}), 200
        else:
            return jsonify({"error": "La registrazione è già in corso"}), 400

@app_py2.route('/stop_recording', methods=['POST'])
def stop_recording():

    global is_recording, current_video_path

    with video_lock:
        if is_recording:
            is_recording = False
            if current_video_path and os.path.exists(current_video_path):
                if send_video(current_video_path):
                    os.remove(current_video_path)
                    return jsonify({"message": "Registrazione fermata e video inviato"}), 200
                else:
                    return jsonify({"error": "Errore durante l'invio del video"}), 500
            return jsonify({"error": "Nessun video disponibile"}), 400
        else:
            return jsonify({"error": "Nessuna registrazione in corso"}), 400

@app_py2.route('/get_video', methods=['POST'])
def get_video():
    global is_recording, current_video_path, last_send_time

    with video_lock:
        if is_recording and current_video_path and os.path.exists(current_video_path):
            if send_video(current_video_path):
                os.remove(current_video_path)
                last_send_time = time.time()
                current_video_path = os.path.join(CHUNKS_FOLDER, f"temp_chunk_{int(time.time())}.mp4")
                return jsonify({"message": "Video corrente inviato. Continuo a registrare i frame successivi."}), 200
            else:
                return jsonify({"error": "Errore durante l'invio del video"}), 500
        else:
            return jsonify({"error": "Nessun video disponibile"}), 400

def record_and_send_video(video_path, nao_ip, nao_port):
    global is_recording, current_video_path, last_send_time

    while is_recording:
        get_video_chunk(video_path, nao_ip, nao_port, last_send_time)
        if send_video(video_path):
            os.remove(video_path)
        current_video_path = os.path.join(CHUNKS_FOLDER, f"temp_chunk_{int(time.time())}.mp4")

if __name__ == '__main__':
    app_py2.run(port=5000, debug=True)

#################################
# FUNZIONI FLASK SERVER Python3 #
#################################

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