#questo file contiene codici temporanei che poi verranno integrati nel main.py py3 e py2
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from flask import Flask, request

#requirements 
'''
opencv-python
opencv-python-headless
flask
flask_login
yieldfrom
PyYAML
numpy
requests
SpeechRecognition
paramiko
psycopg2
psycopg2-binary
pandas
ultralytics
scikit-learn
scipy
'''

#entrabmi i server requirements
#################################
# FUNZIONI FLASK SERVER Python2 #
#################################
#funzione per registrare il video
from flask import Flask, request
import requests
import cv2
import os
import threading
import time
import numpy as np
from naoqi import ALProxy
import random

app = Flask(__name__)

# Variabili globali che vengono utilizzate per gesitre la durata del video
is_recording = False
current_video_path = None
video_lock = threading.Lock()
last_send_time = 0  


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

    # Imposta la luminosità
    video_proxy.setParameter(camera_id, 0, brightness_value)

    # Sottoscrizione al flusso video
    video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, color_space, camera_fps)

    try:
        # Acquisisci un frame dalla telecamera
        image = video_proxy.getImageRemote(video_client)
        if image:
            # Converti l'immagine in un formato utilizzabile da OpenCV
            image_width = image[0]
            image_height = image[1]
            image_data = np.frombuffer(image[6], dtype=np.uint8).reshape((image_height, image_width, 3))

            # Ridimensiona l'immagine a 640x480
            resized_image = cv2.resize(image_data, (640, 480))
            return resized_image
    except Exception as e:
        print(f"Errore durante l'acquisizione dell'immagine: {e}")
    finally:
        # Annulla la sottoscrizione al flusso video
        video_proxy.unsubscribe(video_client)

def get_video_chunk(output_video, start_time, duration=300):
    """
    Registra un chunk di video dalla telecamera del NAO.
    """
    global is_recording

    # Configura il VideoWriter per salvare il video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 15, (640, 480))

    while is_recording and (time.time() - start_time) < duration:
        # Acquisisci un frame dalla telecamera del NAO
        frame = nao_get_image(nao_ip, nao_port)
        if frame is not None:
            # Scrivi il frame nel video
            out.write(frame)

    # Rilascia le risorse
    out.release()

def send_video(video_path):
    """
    Invia il video al server Python 3.
    """
    with open(video_path, "rb") as f:
        try:
            requests.post("http://localhost:5001/receive_video", files={"file": f})
            print(f"Video {video_path} inviato al server Python 3.")
        except Exception as e:
            print(f"Errore durante l'invio del video: {e}")

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """
    Avvia la registrazione del video.
    """
    global is_recording, current_video_path, last_send_time

    with video_lock:
        if not is_recording:
            is_recording = True
            current_video_path = f"temp_chunk_{int(time.time())}.mp4"
            last_send_time = time.time()
            threading.Thread(target=record_and_send_video, args=(current_video_path,)).start()
            return "Registrazione avviata!", 200
        else:
            return "La registrazione è già in corso!", 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """
    Ferma la registrazione del video e invia l'ultimo video al server Python 3.
    """
    global is_recording, current_video_path

    with video_lock:
        if is_recording:
            is_recording = False
            if current_video_path and os.path.exists(current_video_path):
                send_video(current_video_path)
                os.remove(current_video_path)
            return "Registrazione fermata e video inviato!", 200
        else:
            return "Nessuna registrazione in corso!", 400

@app.route('/get_video', methods=['POST'])
def get_video():
    """
    Richiede il video corrente e continua a registrare i frame successivi.
    """
    global is_recording, current_video_path, last_send_time

    with video_lock:
        if is_recording and current_video_path and os.path.exists(current_video_path):
            # Invia il video corrente
            send_video(current_video_path)
            os.remove(current_video_path)

            # Aggiorna il tempo dell'ultimo invio
            last_send_time = time.time()

            # Crea un nuovo file video per i frame successivi
            current_video_path = f"temp_chunk_{int(time.time())}.mp4"
            return "Video corrente inviato. Continuo a registrare i frame successivi.", 200
        else:
            return "Nessun video disponibile!", 400

def record_and_send_video(video_path):
    """
    Funzione che registra e invia i video in modo continuo.
    """
    global is_recording, current_video_path, last_send_time

    while is_recording:
        # Registra i frame successivi all'ultimo invio
        get_video_chunk(video_path, last_send_time)
        send_video(video_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        current_video_path = f"temp_chunk_{int(time.time())}.mp4"

if __name__ == '__main__':
    app.run(port=5000, debug=True)

#################################
# FUNZIONI FLASK SERVER Python3 #
#################################
# video 
from flask import Flask, request
import os
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/receive_video', methods=['POST'])
def receive_video():
    """
    Riceve il video dal server Python 2 e lo salva.
    """
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, f"video_{int(time.time())}.mp4")
    
    file.save(filepath)
    print(f"Video ricevuto e salvato: {filepath}")
    
    # Analizza il video
    analyze_video(filepath)
    return "Video ricevuto!", 200

def analyze_video(video_path):
    """
    Analizza il video utilizzando YOLO.
    """
    print(f"Analizzando {video_path}...")
    # Integra qui il tuo modello YOLO
    pass

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """
    Invia una richiesta al server Python 2 per avviare la registrazione.
    """
    try:
        response = requests.post("http://localhost:5000/start_recording")
        return response.text, response.status_code
    except Exception as e:
        return f"Errore durante la richiesta di avvio: {e}", 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """
    Invia una richiesta al server Python 2 per fermare la registrazione.
    """
    try:
        response = requests.post("http://localhost:5000/stop_recording")
        return response.text, response.status_code
    except Exception as e:
        return f"Errore durante la richiesta di stop: {e}", 500

@app.route('/get_video', methods=['POST'])
def get_video():
    """
    Invia una richiesta al server Python 2 per ottenere il video corrente.
    """
    try:
        response = requests.post("http://localhost:5000/get_video")
        return response.text, response.status_code
    except Exception as e:
        return f"Errore durante la richiesta del video: {e}", 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)


































# Inizializzazione del modello YOLO
model = YOLO("yolov8n.pt")

# Parametri per Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_frame = None
prev_points = None

# Inizializzazione delle variabili
data = []  # Lista per memorizzare i dati
frame_count = 0
fps = 30  # FPS del video
player_id_counter = 0
players_dict = {}  # Dizionario per memorizzare gli ID dei giocatori
substitutions = {}  # Dizionario per gestire le sostituzioni

# Funzione per salvare i dati in CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["ID Giocatore", "Secondo", "Posizione X", "Posizione Y", "Frequenza Cardiaca"])
    df.to_csv(filename, index=False)

# Funzione per stimare la frequenza cardiaca (PPG)
def estimate_heart_rate(roi):
    """
    Stima la frequenza cardiaca utilizzando il canale verde della ROI.
    """
    green_channel = roi[:, :, 1]
    signal = np.mean(green_channel, axis=(0, 1))
    peaks, _ = find_peaks(signal, distance=10)
    heart_rate = len(peaks) * 6  # Fattore di scala
    return heart_rate

# Inizializzazione di Flask
app = Flask(__name__)

@app.route("/substitute", methods=["GET"])
def substitute():
    """
    Endpoint per gestire le sostituzioni.
    Esempio di richiesta: /substitute?old_id=4&new_id=6
    """
    old_id = int(request.args.get("old_id"))
    new_id = int(request.args.get("new_id"))
    substitutions[old_id] = new_id
    return f"Sostituzione registrata: ID {old_id} -> ID {new_id}"

# Apri il video
cap = cv2.VideoCapture("video.mp4")

# Loop principale
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = frame_count // fps  # Tempo in secondi
    
    # Rileva i giocatori con YOLO
    results = model(frame)
    players = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Classe 0 è 'persona' in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                players.append((x1, y1, x2, y2))
    
    # Filtra i giocatori in base al colore (esempio)
    filtered_players = []
    for (x1, y1, x2, y2) in players:
        roi = frame[y1:y2, x1:x2]
        color = np.mean(roi, axis=(0, 1))  # Colore medio della ROI
        filtered_players.append((x1, y1, x2, y2, color))
    
    # Traccia il movimento con Optical Flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray
        prev_points = np.array([[x1 + (x2 - x1) // 2, y2] for (x1, y1, x2, y2, _) in filtered_players], dtype=np.float32)
    else:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_points, None, **lk_params)
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        prev_frame = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
    
    # Stima la frequenza cardiaca (PPG) per ogni giocatore
    for i, (x1, y1, x2, y2, color) in enumerate(filtered_players):
        roi = frame[y1:y2, x1:x2]
        heart_rate = estimate_heart_rate(roi)
        
        # Assegna un ID univoco al giocatore
        if i not in players_dict:
            players_dict[i] = player_id_counter
            player_id_counter += 1
        
        # Gestione delle sostituzioni
        player_id = players_dict[i]
        if player_id in substitutions:
            player_id = substitutions[player_id]
        
        # Salva i dati
        data.append([player_id, current_time, (x1 + x2) // 2, (y1 + y2) // 2, heart_rate])
    
    # Salva i dati ogni 300 secondi
    if current_time > 0 and current_time % 300 == 0:
        file_number = current_time // 300
        save_to_csv(data, f"output_{file_number}.csv")
        data = []  # Resetta la lista dei dati
    
    # Visualizza i risultati
    for (x1, y1, x2, y2, _) in filtered_players:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salva i dati rimanenti
if data:
    save_to_csv(data, "output_final.csv")

cap.release()
cv2.destroyAllWindows()

# Avvia Flask
if __name__ == "__main__":
    app.run(debug=True)
