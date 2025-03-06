# Questo file contiene codici temporanei che poi verranno integrati nel main.py py2
# Inizializzazione di Firebase
cred = credentials.Certificate("path/to/your/firebase-key.json")  # Sostituisci con il percorso della tua chiave Firebase
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-bucket-name.appspot.com'  # Sostituisci con il nome del tuo bucket Firebase
})

#################################
# FUNZIONI FLASK SERVER Python2 #
#################################
from flask import Flask, request, jsonify
import os
import threading
import time
import random
from naoqi import ALProxy
import requests
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import numpy as np
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

