#questo file contiene codici temporanei che poi verranno integrati nel main.py py3
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
