# -*- coding: utf-8 -*-

'''
python3 64 bit
pip3 install -r requirements.txt
python3 main.py
versione Python 3.12.0 64 bit

NOTA BENE!!!  
Avviare prima server Nao: python2 main.py
Il server nao comunica con il Nao attraverso python2.
Questo server si interfaccia con l'utente, il database e AI attraverso python3.
''' 

# Modules
import time
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file
from hashlib import md5, sha256
from datetime import datetime
import requests
import os
import csv
import cv2
import cv2.aruco as aruco
import numpy as np
import utilities
from helpers.config_helper import Config
from helpers.logging_helper import logger
from helpers.speech_recognition_helper import SpeechRecognition
from helpers.db_helper import DB
from openai import OpenAI
from pathlib import Path
from flask_cors import CORS
import whisper
import threading
import sqlite3
import supervision as sv
from supervision import EllipseAnnotator, LabelAnnotator, ColorLookup, Position
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
import torch
from transformers import AutoProcessor, AutoModel
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Voronoi, voronoi_plot_2d
import io
from sklearn.cluster import KMeans
import shutil
import seaborn as sns
import matplotlib
matplotlib.use("Agg")    
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import random






config_helper  = Config()
db_helper      = DB(config_helper)

nao_ip         = config_helper.nao_ip
nao_port       = config_helper.nao_port
nao_user       = config_helper.nao_user
nao_password   = config_helper.nao_password
nao_api_openai = config_helper.api_openai

face_detection = True
face_tracker   = True
local_db_dialog = []
local_rec       = []

#api_key
OpenAI.api_key = config_helper.api_openai

app  = Flask(__name__)
CORS(app)
# flask-login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/'
login_manager.session_protection = 'strong'

def make_md5(s):
    encoding = 'utf-8'
    return md5(s.encode(encoding)).hexdigest()


def make_sha256(s):
    encoding = 'utf-8'
    return sha256(s.encode(encoding)).hexdigest()



#################################
#      computer vision          #
#################################
# variabili globali
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path_mp4 = os.path.join(script_dir, "recordings", "partita", "partita.mp4")
all_positions = []  # <-- Memorizziamo le posizioni dei giocatori 
voronoi_path = os.path.join(script_dir, "recordings", "voronoi", "voronoi_video.mp4")
tactics = "iniziale"

MODEL_PLAYERS = None
MODEL_LINES = None

def inizializza_modelli():
    global MODEL_PLAYERS, MODEL_LINES
    model_dir = os.path.join(script_dir, "models")
    MODEL_PLAYERS = YOLO(os.path.join(model_dir, "yolov8n_persone.pt"), verbose=False)
    MODEL_LINES = YOLO(os.path.join(model_dir, "yolo_lines_8n.pt"), verbose=False)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi
import cv2


def genera_video_voronoi(all_positions, output_path):
    """
    Genera un video con diagrammi di Voronoi su un campo da calcio stilizzato.

    Parametri:
        all_positions: lista di frame, ciascuno contenente una lista di tuple (x, y, team)
        output_path
    """
    field_length, field_width = 30, 15  # Dimensioni del campo in metri
    video_width, video_height = 600, 300  # Risoluzione del video in pixel
    fps = 10  # Fotogrammi al secondo

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    for frame_data in all_positions:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_aspect('equal')
        ax.axis('off')

        # Sfondo verde
        ax.add_patch(patches.Rectangle((0, 0), field_length, field_width, color='green'))

        # Linee del campo
        # Linea di metà campo
        ax.plot([field_length / 2, field_length / 2], [0, field_width], color='white', linewidth=1)
        # Cerchio di centrocampo
        ax.add_patch(patches.Circle((field_length / 2, field_width / 2), 1, edgecolor='white', facecolor='none', linewidth=1))
        # Aree di rigore
        ax.add_patch(patches.Rectangle((0, (field_width - 16.5) / 2), 16.5, 16.5, edgecolor='white', facecolor='none', linewidth=1))
        ax.add_patch(patches.Rectangle((field_length - 16.5, (field_width - 16.5) / 2), 16.5, 16.5, edgecolor='white', facecolor='none', linewidth=1))
        # Aree di porta
        ax.add_patch(patches.Rectangle((0, (field_width - 7.32) / 2), 5.5, 7.32, edgecolor='white', facecolor='none', linewidth=1))
        ax.add_patch(patches.Rectangle((field_length - 5.5, (field_width - 7.32) / 2), 5.5, 7.32, edgecolor='white', facecolor='none', linewidth=1))

        # Estrai le posizioni e i team
        points = [(x, y) for x, y, team in frame_data if team in ["blue", "red"]]
        teams = [team for x, y, team in frame_data if team in ["blue", "red"]]

        if len(points) > 2:
            vor = Voronoi(points)
            for point_idx, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if not -1 in region and len(region) > 0:
                    polygon = [vor.vertices[i] for i in region]
                    team = teams[point_idx]
                    color = 'blue' if team == 'blue' else 'red'
                    ax.fill(*zip(*polygon), alpha=0.3, color=color)

            for (x, y), team in zip(points, teams):
                ax.plot(x, y, 'o', color='blue' if team == 'blue' else 'red')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]  # rimuove canale alpha (RGBA → RGB)
        frame = cv2.resize(frame, (video_width, video_height))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        plt.close(fig)

    out.release()


def analizza_partita():
    global all_positions
    all_positions = []
    inizializza_modelli()

    output_path = os.path.join(script_dir, "recordings", "annotato.mp4")
    if not os.path.exists(video_path_mp4):
        return

    cap = cv2.VideoCapture(video_path_mp4)
    if not cap.isOpened():
        return

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = sv.ByteTrack()
    ellipse_annotator = EllipseAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
    label_annotator = LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_position=Position.BOTTOM_CENTER, text_scale=0.5)

    homography_matrix = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if homography_matrix is None:
            lines = MODEL_LINES(frame, conf=0.3)
            det = sv.Detections.from_ultralytics(lines[0])
            points = [[(b[0]+b[2])/2, b[3]] for b in det.xyxy[:4]]
            if len(points) == 4:
                homography_matrix, _ = cv2.findHomography(np.array(points, np.float32), np.array([[0,0],[30,0],[0,15],[30,15]], np.float32))

        results = MODEL_PLAYERS(frame, conf=0.2)
        det = sv.Detections.from_ultralytics(results[0])
        players = det[det.class_id != 0].with_nms(threshold=0.5)

        frame_data = []
        if len(players) > 0:
            tracked = tracker.update_with_detections(players)
            colors = [np.mean(frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])].reshape(-1, 3), axis=0) if frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])].size > 0 else [0, 0, 0] for b in tracked.xyxy]
            if len(colors) >= 2:
                kmeans = KMeans(n_clusters=2, n_init=10).fit(colors)
                teams = ["blue" if l == 0 else "red" for l in kmeans.labels_]
            else:
                teams = ["unknown"] * len(tracked.xyxy)
            tracked.data["team"] = teams

            labels = [f"#{tid} ({team})" for tid, team in zip(tracked.tracker_id, teams)]
            annotated = ellipse_annotator.annotate(scene=frame, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            for i, box in enumerate(tracked.xyxy):
                center = np.array([(box[0]+box[2])/2, box[3], 1])
                if homography_matrix is not None:
                    pt = homography_matrix @ center
                    pt /= pt[2]
                else:
                    pt = center
                frame_data.append((float(pt[0]), float(pt[1]), teams[i]))
            frame = annotated

        all_positions.append(frame_data)
        out.write(frame)

    cap.release()
    out.release()
    genera_video_voronoi(all_positions, voronoi_path)
                
@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    if file and file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        os.makedirs(os.path.dirname(video_path_mp4), exist_ok=True)
        file.save(video_path_mp4)
        return jsonify({"status": "ok", "message": "Video caricato con successo."}), 200
    return jsonify({"status": "error", "message": "Formato non valido."}), 400

@app.route('/analyze', methods=['GET'])
def analyze():
    print("inizio analisi")
    analizza_partita()
    return jsonify({"status": "ok", "message": "Analisi completata."}), 200

@app.route('/stream_voronoi', methods=['GET'])
def stream_voronoi():
    path = os.path.join(script_dir, "recordings", "annotato_video.mp4")
    if os.path.exists(path):
        return send_file(path, mimetype='video/mp4')
    return "Video non trovato", 404


@app.route('/sostituzione', methods=['GET'])
def sostituzione():
    dati = get_player_details()
    
    performance = {}
    for player in dati:
        id_player = player['id_player']
        total = 0
        for bpm, passi, velocita, id in player['dati']:
            total += bpm + velocita  #più alto il valore (bpm+velocità) più alto rendimento giocatore
        media = total / len(player['dati'])
        performance[id_player] = media

    # Trova il giocatore con la performance media più bassa
    peggior_id = min(performance, key=performance.get)
    giocatore = db_helper.get_player_by_id(peggior_id)

    text = f"Sostituire il giocatore {giocatore[0]} {giocatore[1]}"

    speech_ai(text)
    return jsonify({"status": "ok"}), 200


@app.route('/tactics', methods=['GET'])
def tactics():
    dati = get_player_details()  # lista di giocatori con dati biometrici
    score = {
        "audace": global_score_audace,
        "ospiti": global_score_ospite
    }

    giocatori_stanchi = []

    for player in dati:
        id_player = player['id_player']
        dati = player['dati']

        bpm_tot = sum([x[0] for x in dati])
        passi_tot = sum([x[1] for x in dati])
        velocita_tot = sum([x[2] for x in dati])

        n = len(dati)

        media_bpm = bpm_tot / n
        media_passi = passi_tot / n
        media_velocita = velocita_tot / n


        # Definizione di "stanco"
        if media_bpm > 130 or media_velocita < 4.2 :
            giocatori_stanchi.append(id_player)


    # Scelta in base a numero di giocatori stanchi e score
    modulo = ""
    tattica = ""

    num_stanchi = len(giocatori_stanchi)
    differenza = score["audace"] - score["ospiti"]

    modulo = ""
    tattica = ""

    num_stanchi = len(giocatori_stanchi)

    if num_stanchi >= 4:
        if differenza >= 1:
            # Vantaggio + squadra stanca 
            modulo = "2-2"
            tattica = "Difesa compatta, ritmo lento e possesso palla"
        elif differenza <= -1:
            # Svantaggio + stanchezza 
            modulo = "1-2-1"
            tattica = "Costruzione ragionata ed evitare sprint"
        else:
            # Parità + stanchezza
            modulo = "2-2"
            tattica = "Gestione equilibrata con rotazioni frequenti"

    elif num_stanchi >= 2 and num_stanchi <= 3:
        if differenza >= 1:
            # Vantaggio + fatica moderata
            modulo = "2-1-1"
            tattica = "Difesa solida, ripartenze rapide ma non esagerare"
        elif differenza <= -1:
            # Svantaggio + fatica moderata
            modulo = "1-2-1"
            tattica = "Pressing organizzato e sfruttare gli spazi"
        else:
            # Parità
            modulo = "1-2-1"
            tattica = "Tattica bilanciata e cercare di ottnere il  controllo del centrocampo"

    elif num_stanchi == 1:
        giocatore = giocatore = db_helper.get_player_by_id(id_player)
        sub = f"sostituire il giocatore {giocatore[0]} {giocatore[1]}"
        if differenza >= 1:
            # Vantaggio + squadra fresca
            modulo = "2-1-1"
            tattica = "Controllo palla con avanzate sicure ad alta intesnità"
        elif differenza <= -1:
            # Svantaggio + squadra fresca
            modulo = "0-2-2"
            tattica = "Pressing alto con attacco diretto"
        else:
            # Parità + squadra fresca
            modulo = "1-1-2"
            tattica = "Pressing medio con attacchi improvvisi e lanci lunghi"

    elif num_stanchi < 1:
        if differenza >= 1:
            # Vantaggio + squadra fresca
            modulo = "2-1-1"
            tattica = "Controllo palla con avanzate sicure ad alta intesnità"
        elif differenza <= -1:
            # Svantaggio + squadra fresca
            modulo = "0-2-2"
            tattica = "Pressing alto con attacco diretto"
        else:
            # Parità + squadra fresca
            modulo = "1-1-2"
            tattica = "Pressing medio con attacchi improvvisi e lanci lunghi"

    else:
        modulo = "1-2-1"
        tattica = "Modulo standard con gestione flessibile, badare agli errori"

    
    if num_stanchi == 1:
        text = f"sostituire {sub} e poi gochiamo con un {modulo} e {tattica}"
    else:
        text = f"giochiamo con un {modulo} e {tattica}"

    nao_animatedSayText(text)
    return jsonify({"status": "ok"}), 200



#################################
#            Tribuna            #
#################################
# VARIABILI GLOBALI task 2
global time_1
global time_2
global time_3
global minutes
global seconds
task_2 = False
global global_timer_running 
global_timer_running = False
global global_timer_start 
global_timer_start =0
global global_game_time 
global_game_time =0
global global_score_audace 
global_score_audace =0
global global_score_ospite
global_score_ospite=0
global counter
counter=0
global prima_disponibile
prima_disponibile=False

@app.route('/api/start_timer', methods=['POST'])
def start_timer():
    global global_timer_running, global_timer_start
    if not global_timer_running:
        global_timer_start = time.time() - global_game_time
        global_timer_running = True
    return jsonify({"status": "started"})

@app.route('/api/stop_timer', methods=['POST'])
def stop_timer():
    global global_timer_running, global_game_time
    if global_timer_running:
        global_game_time = time.time() - global_timer_start
        global_timer_running = False
    return jsonify({"status": "stopped"})

@app.route('/api/get_status', methods=['GET'])
def get_status():
    global global_timer_running, global_game_time, global_score_audace, global_score_ospite, time_1,time_3,minutes,seconds
    current_time = global_game_time
    if global_timer_running:
        current_time = time.time() - global_timer_start
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    time_1 = f"{minutes:02d}:{seconds:02d}"
    return jsonify({
        "time_1": f"{minutes:02d}:{seconds:02d}",
        "audace": global_score_audace,
        "ospite": global_score_ospite
    })

@app.route('/api/increment_score/<team>', methods=['POST'])
def increment_score(team):
    global global_score_audace, global_score_ospite
    if team == 'audace':
        global_score_audace += 1
    elif team == 'ospite':
        global_score_ospite += 1
    return jsonify({"status": "score updated"})

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    global global_timer_running, global_timer_start, global_game_time
    global global_score_audace, global_score_ospite,counter,posto_t,riga_t
    global_timer_running = False
    global_timer_start = 0
    global_game_time = 0
    global_score_audace = 0
    global_score_ospite = 0
    counter=0
    posto_t=1
    riga_t=1
    return jsonify({"status": "game reset"})


def tempo_di_pausa():
    print("Ora aspetta")
    timer = threading.Timer(5, ritorno_a_false)
    timer.start()

def ritorno_a_false():
    task_2 = False

def nao_dance_1(): #baletto richiamato in nao_coro
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_dance_1/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

@app.route('/nao_points', methods=['GET'])
def nao_points():
    global task_2
    global global_score_audace, global_score_ospite
    if global_score_audace ==global_score_ospite and global_score_audace==0:
        text = "Siamo ancora 0 pari, forza audace presto segneremo"
        speech_ai(text)
    elif global_score_audace ==global_score_ospite:
        text = "Siamo "+ str(global_score_audace) +"pari, forza audace presto segneremo"
        speech_ai(text)
    elif global_score_audace<global_score_ospite and (global_score_ospite-global_score_audace)==1:
        text = "Siamo sotto di un solo gol, possiamo recuperare"
        speech_ai(text)
    elif global_score_audace<global_score_ospite and (global_score_ospite-global_score_audace)>1:
        text = "Peccato siamo sotto di"+str(global_score_ospite-global_score_audace)+"gol, non perdiamo la speranza audace"
        speech_ai(text)
    elif global_score_audace>global_score_ospite and (global_score_audace-global_score_ospite)==1:
        text = "Siamo in vantaggio di un gol, evviva"
        speech_ai(text)
    elif global_score_audace>global_score_ospite and (global_score_audace-global_score_ospite)>1:
        text = "Siamo in vantaggio di " + str(global_score_audace - global_score_ospite) + " gol, nessuno ci può battere, evviva"
        speech_ai(text)
    tempo_di_pausa()


def get_seat():
    url      = "http://127.0.0.1:5011/get_seat"
    response = requests.get(url)
    logger.info(str(response.text))
    return response.json()['counter']
    

@app.route('/nao_seat', methods=['GET'])
def nao_seat():
    counter = get_seat()

    global task_2
    global prima_disponibile
    # Numero di righe e colonne
    righe = 6
    colonne = 10
    riga_t=1
    posto_t=1
    matrice = [[True for _ in range(colonne)] for _ in range(righe)]
    posti_max = righe*colonne
    posti_occupati =  counter

    # Imposta a False i primi 'posti_occupati' posti, riga per riga
    contatore = 0
    for i in range(righe):
        for j in range(colonne):
            if (contatore < posti_occupati and posti_occupati<=posti_max) or posti_occupati!=0:
                matrice[i][j] = False
                contatore += 1
            else:
                break
    for i in range(righe):
        for j in range(colonne):
            if matrice[i][j] == True:
                riga_t=i+1
                posto_t=j+1
                prima_disponibile=True
                break
        if prima_disponibile:
            break
    text="Puoi sederti nel posto {} della fila {}".format(posto_t, riga_t)
    nao_animatedSayText(text)
    tempo_di_pausa()
    

@app.route('/nao_time_match', methods=['GET'])
def nao_time_match():
    global task_2, time_3, time_2, minutes,seconds
    time_3 = get_status()
    time_3_data = f"{minutes:02d}:{seconds:02d}"
    time_2 = time_3_data
    if time_2=="00:00":
        text ="La partita non è ancora iniziata"
    else:
        text = "La partita è iniziata da "+ str(minutes)+"minuti "+ str(seconds)+"secondi"
    nao_animatedSayText(text)
    tempo_di_pausa()

@app.route('/nao_cori', methods=['GET'])
def nao_cori():  
    global task_2
    text = random.choice([
    "Alè alè oh oh, alè alè oh oh, Audace siamo noi, Audace siamo noi, Audace siamo noooi",
    "Questa passione la dedico a te, Audace mia, non ti lascerò mai",
    "Amarsi ancora, per questi colori, per questa città",
    "Sempre con te, ovunque andrai, Audace nel cuore, non ti lascerò mai",
    "Curva Sud canta per te, Audace Audace, orgoglio della città",
    "Forza Audace, vinci per noi, la Curva Sud è sempre con te"])
    nao_SayText(text)
    nao_dance_1()
    
    tempo_di_pausa() 

 
#################################
# FUNZIONI FLASK SERVER Python2 #
#################################


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     # Carica il classificatore Haar per il rilevamento dei volti
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                    # Converti il frame in scala di grigi
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))    # Rileva i volti nel frame
    for (x, y, w, h) in faces:                                                                              # Disegna un rettangolo attorno ai volti rilevati
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

@app.route('/webcam', methods=['GET'])
def webcam():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_webcam/" + str(data) 
    response = requests.get(url, json=data, stream=True)

    # face detection
    def generate_frames():
        boundary     = b'--frame\r\n'
        content_type = b'Content-Type: image/jpeg\r\n\r\n'
        frame_data   = b''

        for chunk in response.iter_content(chunk_size=1024):
            frame_data += chunk
            if boundary in frame_data:
                # Estrai il frame
                parts = frame_data.split(boundary)
                for part in parts[:-1]:                    
                    if content_type in part:
                        frame_data = part.split(content_type)[-1]
                        # Decodifica il frame
                        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame    = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                        # Esegui il rilevamento facciale
                        if face_detection:
                            frame_with_faces = detect_faces(frame)
                            # Codifica di nuovo il frame con i volti rilevati come JPEG
                            _, buffer = cv2.imencode('.jpg', frame_with_faces)
                        else:
                            # Codifica di nuovo il frame 
                            _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_data = parts[-1]

    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_aruco', methods=['GET'])
def webcam_aruco():
    ### per recuperare frame dal nao tramite py2 ###
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_webcam/" + str(data) 
    response = requests.get(url, json=data, stream=True)

    ### per recuperare frame tramite webcam collegata al pc ###
    
    #recupero variabili
    # Inizializza il dizionario ArUco
    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters()

    # Inizializza il rilevatore ArUco
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # aruco detection

    
    def generate_frames():

        boundary     = b'--frame\r\n'
        content_type = b'Content-Type: image/jpeg\r\n\r\n'
        frame_data   = b''

        for chunk in response.iter_content(chunk_size=1024):
            frame_data += chunk
            if boundary in frame_data:
                # Estrai il frame   
                parts = frame_data.split(boundary)

                for part in parts[:-1]:
                    if content_type in part:
                        frame_data = part.split(content_type)[-1]
                        # Decodifica il frame
                        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame    = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                        # Conversione del frame in scala di grigi per migliorare il rilevamento dei marker
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

                        # Se vengono rilevati marker, disegnali sul frame
                        if marker_ids is not None:
                            # Disegna i marker rilevati sul frame
                            aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

                            # TASK 2

                            #SCORE PARTITA
                            global task_2
                            if 184 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_points()

                            #Statistiche della partita
                            elif 185 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_time_match() #time

                            #Posti a sedere
                            elif 186 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_seat()

                            #cori
                            elif 187 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_cori() #festeggiamo

                        #ricodifica e invia il frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                frame_data = parts[-1]

    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



def nao_move_back(angle):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "angle":angle}
    url      = "http://127.0.0.1:5011/nao_move_back/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))  


def nao_move_fast(angle):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "angle":angle}
    url      = "http://127.0.0.1:5011/nao_move_fast/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))   


def nao_move_fast_stop():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_move_fast_stop/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))  


def nao_get_sensor_data():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_get_sensor_data/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))  
    return eval(str(response.text))['data']


nao_train_move_start = True
theta_speed = 0.0
def nao_train_move():
    duration   = 10
    filename   = 'nao_training_data.csv'
    start_time = time.time()
    data       = []

    #nao_move_fast(theta_speed)
    while nao_train_move_start:
        sensors = nao_get_sensor_data()
        x_speed = 1.0
        y_speed = 0.0
        data.append(sensors + [x_speed, y_speed, theta_speed])
        time.sleep(0.1)
    nao_move_fast_stop() 

    # Salva i dati in un file CSV
    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['gyro_x', 'gyro_y', 'acc_x', 'acc_y', 'acc_z', 'x_speed', 'y_speed', 'theta_speed'])
        writer.writerows(data)



@app.route('/webcam_aruco_pose_estimate', methods=['GET'])
def webcam_aruco_pose_estimate():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_webcam/" + str(data) 
    response = requests.get(url, json=data, stream=True)

    # Ottieni le dimensioni del frame dalla webcam
    width    = 640
    height   = 480
    center_x = width  // 2
    center_y = height // 2

    # Inizializza il dizionario ArUco
    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters()

    # Inizializza il rilevatore ArUco
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # Dimensione reale del marker in metri (ad esempio: 0.05 per 5 cm)
    marker_size = 0.025

    # Focale della camera in pixel (esempio, deve essere calibrata per la tua camera specifica)
    focal_length = 800

    # aruco detection
    def generate_frames():
        boundary     = b'--frame\r\n'
        content_type = b'Content-Type: image/jpeg\r\n\r\n'
        frame_data   = b''

        for chunk in response.iter_content(chunk_size=1024):
            frame_data += chunk
            if boundary in frame_data:
                # Estrai il frame
                parts = frame_data.split(boundary)
                for part in parts[:-1]:                    
                    if content_type in part:
                        frame_data = part.split(content_type)[-1]
                        # Decodifica il frame
                        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame    = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                        # Conversione del frame in scala di grigi per migliorare il rilevamento dei marker
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Rileva i marker ArUco nel frame usando ArucoDetector
                        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

                        # Se vengono rilevati marker, determina la posizione e la distanza
                        if marker_ids is not None:
                            for marker_id in marker_ids:
                                if marker_id == 449:
                                    for corners in marker_corners:
                                        # Calcola il centro del marker
                                        marker_center_x = int(corners[0][:, 0].mean())
                                        marker_center_y = int(corners[0][:, 1].mean())

                                        # Calcola la lunghezza apparente del marker in pixel
                                        pixel_size = cv2.norm(corners[0][0] - corners[0][1])

                                        # Calcola la distanza
                                        distance = (marker_size * focal_length) / pixel_size

                                        # Determina la deviazione dal centro dell'immagine
                                        deviation_x = marker_center_x - center_x
                                        deviation_y = marker_center_y - center_y

                                        # Stabilisci la direzione
                                        if distance > 0.20:
                                            nao_move_fast(0)
                                            
                                            if abs(deviation_x) < 10:  # Tolleranza per essere considerato "dritto"
                                                direction = "Dritto"
                                                nao_move_fast(0)
                                            elif deviation_x > 0:
                                                direction = "Storto a destra"
                                                nao_move_fast(10)
                                            else:
                                                direction = "Storto a sinistra"
                                                nao_move_fast(-10)
                                        else:
                                            nao_move_fast_stop()
                                            #pass

                                        print(f"Distanza dal marker: {distance:.2f} metri, Posizione marker: ({marker_center_x}, {marker_center_y}), Direzione: {direction}")

                                        # Disegna i marker rilevati sul frame
                                        aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

                                        # Disegna una linea tra il centro del frame e il centro del marker
                                        cv2.line(frame, (center_x, center_y), (marker_center_x, marker_center_y), (0, 255, 0), 2)
                        

                        # Disegna il centro del frame
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                        
                        # Codifica di nuovo il frame 
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_data = parts[-1]

    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def nao_audiorecorder(sec_sleep):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "nao_user":nao_user, "nao_password":nao_password, "sec_sleep":sec_sleep}
    url      = "http://127.0.0.1:5011/nao_audiorecorder/" + str(data) 
    response = requests.get(url, json=data, stream=True)

    local_path = f'recordings/microphone_audio.wav'
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)   
        logger.info("File audio ricevuto: " + str(response.status_code))
    else:
        logger.error("File audio non ricevuto: " + str(response.status_code))

    while True:
        speech_recognition = SpeechRecognition(local_path)
        if (speech_recognition.result != None or speech_recognition.result != ''):
            break
    
    logger.info("nao_audiorecorder: " + str(speech_recognition.result))
    return str(speech_recognition.result)

@app.route('/nao_touch_head_audiorecorder', methods=['GET'])
def nao_touch_head_audiorecorder():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "nao_user":nao_user, "nao_password":nao_password}
    url      = "http://127.0.0.1:5011/nao_touch_head_audiorecorder/" + str(data) 
    response = requests.get(url, json=data, stream=True)
    print("Ordine ricevuto")
    local_path = f'recordings/microphone_audio.wav'
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)   
        logger.info("File audio ricevuto: " + str(response.status_code))
    else:
        logger.error("File audio non ricevuto: " + str(response.status_code))

    # Utilizzo della libreria openai-whisper per eseguire lo speech-to-text
    model = whisper.load_model("base")
    
    try:
        result = model.transcribe(path_audio, language="it")
    except Exception as e:
        print(f"Errore durante la trascrizione: {e}")
        raise # In caso di errore, stampa e rilancia
    
    
    ordine = result.get("text", "").strip()
    punteggiatura = "!.?;,:-–—'\"()[]{}<>/\\@#€%&*+_=^°§"
    for simbolo in punteggiatura:
        ordine = ordine.replace(simbolo, "")

    if ordine =="dove posso sedermi" or ordine =="nao dove posso sedermi":
        nao_seat()
    elif ordine == "nao a quano siamo" or ordine == "a quanto siamo":
        nao_points()
    elif ordine == "nao festeggiamo":
        nao_cori()
    elif ordine == "nao quanto tempo è passato da inizio partita" or ordine == "quanto tempo è passato da inizio partita":
        nao_time_match()

@app.route('/nao_touch_head_counter', methods=['GET'])
def nao_touch_head_counter():
    # Prepara i dati da inviare
    data = {
        "nao_ip":       nao_ip,
        "nao_port":     nao_port,
        "nao_user":     nao_user,
        "nao_password": nao_password
    }

    url = "http://127.0.0.1:5011/nao_touch_head_counter/" + str(data) 
    response = requests.get(url, json=data, stream=True)

    if response.status_code == 200:
        app.logger.info("Tocco ricevuto da Python 2 (200).")
        return jsonify({'code': 200, 'message': 'Tocco registrato con successo'}), 200
    else:
        app.logger.error(f"Errore dal server Python 2: {response.status_code}")
        return jsonify({'code': 500, 'message': 'Errore remoto'}), 500


def nao_face_tracker():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_face_tracker/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_stop_face_tracker():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_stop_face_tracker/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))




def nao_SayText(text_to_say):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "text_to_say":text_to_say}
    url      = "http://127.0.0.1:5011/nao_SayText/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_standInit():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_standInit/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                      


def nao_stand():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_stand/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_volume_sound(volume_level):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "volume_level":str(volume_level)}
    url      = "http://127.0.0.1:5011/nao_volume_sound/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_tts_audiofile(filename): # FILE AUDIO NELLA CARTELLA tts_audio DI PY2
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "filename":filename, "nao_user":nao_user, "nao_password":nao_password}
    url      = "http://127.0.0.1:5011/nao_tts_audiofile/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


# PAGINE WEB
# Per impedire all'utente di tornare indietro dopo aver fatto il logout
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

class User(UserMixin):
    def __init__(self, id = None, username = ''):
        self.id = id

users = {'1': {'id': '1', 'username': 'admin', 'password': '21232f297a57a5a743894a0e4a801fc3'},  # md5(admin)
        '2': {'id': '2', 'username': '1', 'password': 'c4ca4238a0b923820dcc509a6f75849b'},     # md5(1)
        '3': {'id': '3', 'username': '2', 'password': 'c81e728d9d4c2f636f067f89cc14862c'}}      # md5(2)

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = make_md5(request.form["password"])

        user = next((u for u in users.values() if u['username'] == username and u['password'] == password), None)
        if user:
            user_obj = User(user['id'])
            login_user(user_obj)
            # Reindirizza in base allo username
            if username == "1":
                return redirect(url_for('home'))
            elif username == "2":
                return redirect(url_for('home2'))
            else:
                # Default: se lo username non corrisponde alle condizioni specificate
                return redirect(url_for('home'))
                
    return render_template('login.html')

@app.route("/logout", methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/home', methods=['GET'])
@login_required
def home():
    return render_template('home.html')

@app.route('/salute', methods=['GET'])
@login_required
def salute():
    return render_template('salute.html')

@app.route('/joystick', methods=['GET'])
@login_required
def joystcik():
    return render_template('joystick.html')

@app.route('/competition', methods=['GET'])
@login_required
def competition():
    return render_template('competition.html')

@app.route('/partita', methods=['GET'])
@login_required
def partita():
    #players = db_helper.select_players()     , players=players
    return render_template('partita.html')

@app.route('/registra', methods = ['GET']) 
# tramite questa pagina creai nuovi utenti e salvi nel database tramite funzione api_app_utenti(id):
@login_required
def database():
    return render_template('registra.html')

@app.route('/fans', methods = ['GET'])
@login_required 
def fans():
    return render_template('fans.html')

# API
@app.route('/api', methods=['GET'])
def api():
    return render_template('api.html')


@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({'code': 200, 'status': 'online', 'elapsed time': utilities.getElapsedTime(startTime)}), 200


@app.route('/api/audio_rec', methods=['GET'])
def api_audio_rec():
    if request.method == 'GET':
        try:
            return jsonify({'code': 200, 'message': 'OK', 'recordings': local_rec}), 200
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500


@app.route('/api/dialogo', methods=['GET'])
def api_dialogo():
    if request.method == 'GET':
        try:
            return jsonify({'code': 200, 'message': 'OK', 'data': local_db_dialog}), 200
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500


@app.route('/tts_to_nao', methods=['POST'])
def tts_to_nao():
    if request.method == "POST":
        text = request.form["message"]
        nao_animatedSayText(text)
    return redirect('/home')

@app.route('/tts_to_nao_ai', methods=['POST'])
def tts_to_nao_ai():
    if request.method == "POST":
        #collegamento a openai
        text = request.form["message_ai"]
        client = OpenAI(api_key = nao_api_openai)
        speech_file_path = Path(__file__).parent.parent / "py2/tts_audio/speech.mp3"
        response = client.audio.speech.create(model="tts-1",voice="nova",input=text)
        response.stream_to_file(speech_file_path)
        nao_tts_audiofile("speech.mp3")
        
    return redirect('/home')


# funione da usare a livello locale nel server
def speech_ai(text):
    client = OpenAI(api_key = nao_api_openai)
    speech_file_path = Path(__file__).parent.parent / "py2/tts_audio/speech.mp3"
    response = client.audio.speech.create(model="tts-1",voice="nova",input=text)
    response.stream_to_file(speech_file_path)
    nao_tts_audiofile("speech.mp3")

    return redirect("#")


@app.route('/set_volume', methods=['POST'])
def set_volume():
    data = request.get_json()
    volume_level = data.get('volume_level')
    
    if volume_level is None:
        return jsonify({"error": "Missing volume level"}), 400
    
    # Chiama la funzione che manda la richiesta al server Py2
    try:
        nao_volume_sound(int(volume_level))
        return jsonify({"status": "Volume aggiornato", "volume": volume_level}), 200
    except Exception as e:
        logger.error("Errore nel settaggio del volume: " + str(e))
        return jsonify({"error": "Errore interno"}), 500


# MOVEMENTS (Haseeb ha pagato gente per lavorare al suo posto)
@app.route('/api/movement/start', methods=['GET'])
def api_movement_start():
    nao_move_fast(0)
    return redirect('/home')


@app.route('/api/movement/stop', methods=['GET'])
def api_movement_stop():
    nao_move_fast_stop()
    return redirect('/home')

@app.route('/api/movement/left', methods=['GET'])
def api_movement_left():
    global theta_speed
    theta_speed = 10
    nao_move_fast(10)
    return redirect('/home')

@app.route('/api/movement/right', methods=['GET'])
def api_movement_right():
    global theta_speed
    theta_speed = -10
    nao_move_fast(-10)
    return redirect('/home')

@app.route('/api/movement/back', methods=['GET'])
def api_movement_back():
    nao_move_back(0)
    return redirect('/home')

@app.route('/api/movement/stand', methods=['GET'])
def api_movement_stand():
    nao_stand()
    return redirect('/home')

@app.route('/api/movement/standInit', methods=['GET'])
def api_movement_standInit():
    nao_standInit()
    return redirect('/home')

@app.route('/api/movement/nao_train_move', methods=['GET'])
def api_movement_nao_train_move():
    global nao_train_move_start 
    nao_train_move_start = True
    nao_train_move()
    return redirect('/home')

@app.route('/api/movement/nao_train_move_stop', methods=['GET'])
def api_movement_nao_train_move_stop():
    global nao_train_move_start 
    nao_train_move_start = False
    return redirect('/home')

@app.route('/api/movement/nao_sitdown', methods=['GET'])
def nao_sitdown():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_sitdown/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))   

@app.route('/api/movement/nao_autonomous_life', methods=['GET'])
def nao_autonomous_life():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_autonomous_life/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                                                     

@app.route('/api/movement/nao_autonomous_life_state', methods=['GET'])
def nao_autonomous_life_state():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_autonomous_life_state/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))  

@app.route('/api/movement/nao_wakeup',methods=['GET'])
def nao_wakeup():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_wakeup/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                                                   

@app.route('/api/movement/red_eye',methods=['GET'])
def nao_eye_red():
    data     = {"nao_ip": nao_ip, "nao_port": nao_port, "r": 255, "g": 0, "b": 0}  # Red color
    url      = "http://127.0.0.1:5011/nao_eye/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

@app.route('/api/movement/green_eye',methods=['GET'])
def nao_eye_green():
    data     = {"nao_ip": nao_ip, "nao_port": nao_port, "r": 0, "g": 255, "b": 0}  # Green color
    url      = "http://127.0.0.1:5011/nao_eye/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

@app.route('/api/movement/blue_eye',methods=['GET'])
def nao_eye_blue():
    data     = {"nao_ip": nao_ip, "nao_port": nao_port, "r": 0, "g": 0, "b": 255}  # Blue color
    url      = "http://127.0.0.1:5011/nao_eye/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

@app.route('/api/movement/nao_eye',methods=['GET'])
def nao_eye_white():
    data     = {"nao_ip": nao_ip, "nao_port": nao_port, "r": 255, "g": 255, "b": 255}  # White color
    url      = "http://127.0.0.1:5011/nao_eye/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

@app.route('/nao/battery',methods=['GET'])
def nao_battery_level():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_battery/" + str(data)
    try:
        response = requests.get(url, json=data)
        logger.info(str(response.text))
        battery_info = response.json()
        battery_level = battery_info["battery_level"]
        return jsonify({'battery_level': battery_level}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/nao_animatedSayText', methods = ['GET'])
def nao_animatedSayText(text_to_say):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "text_to_say":text_to_say}
    url      = "http://127.0.0.1:5011/nao_animatedSayText/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

###  database  ###
@app.route('/api/app/utente/<id>', methods=['POST'])
def api_app_utente(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"username":value, "password":value}
                json = request.json
                username = json["username"]
                password = json["password"]
                data = db_helper.select_account_player(username, password)
                return jsonify({'code': 200, 'message': 'OK', 'data': data}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500

@app.route('/api/app/dati/<id>', methods=['POST'])
def api_app_dati(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"id_player":id, "bpm":98, "passi":72, "velocita":4"}
                json = request.json
                id_player = json["id_player"]
                bpm = json["bpm"]
                passi = json["passi"]
                velocità = json["velocita"]
                db_helper.insert_dati(id_player, bpm, passi, velocità)
                
                if bpm >= 185:
                    player = db_helper.get_player_by_id(id_player) # ricevo una lista con [nome, cognome]
                    text = f"il giocatore {player[0]} {player[1]} deve abbassare il ritmo"
                    nao_animatedSayText(text)

                return jsonify({'code': 200, 'message': 'OK',}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            logger.error('No id argument passed')
            return jsonify({'code': 500, 'message': 'No id was passed'}), 500

@app.route('/api/app/infortuni/<id>', methods=['POST'])
def api_app_infortuni(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"id_player":n, "ammonizone":False, "infortunio":True}
                json = request.json
                id_player = json["id_player"]
                ammonizione = json["ammonizione"]
                infortunio = json["infortunio"]
                db_helper.insert_disponibilità(id_player, ammonizione, infortunio)
                return jsonify({'code': 200, 'message': 'OK',}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            logger.error('No id argument passed')
            return jsonify({'code': 500, 'message': 'No id was passed'}), 500

@app.route('/api/app/utenti', methods=['POST'])
def api_app_utenti():
    try:
        # Esempio JSON: {"username": "mt", "password": "0987", "nome": "marco", "cognome": "tomazzoli", "posizione": "laterale"}
        data = request.get_json()

        username = data["username"]
        password = data["password"]
        nome = data["nome"]
        cognome = data["cognome"]
        posizione = data["posizione"]

        db_helper.insert_utente(username, password, nome, cognome, posizione)

        return jsonify({'code': 200, 'message': 'OK'}), 200

    except Exception as e:
        logger.error(f"Errore in /api/app/utenti: {str(e)}")
        return jsonify({'code': 500, 'message': str(e)}), 500



@app.route('/api/db/dati', methods=['GET'])
def get_all_dati():
    try:
        results = db_helper.select_all_dati()
        return jsonify({'code': 200, 'data': results}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({'code': 500, 'message': str(e)}), 500
    

@app.route('/api/db/dati/clear', methods = ['GET'])
def clear_dati():
        db_helper.delete_dati()
        return jsonify({'code': 200}), 200


def get_player_details():
    dati = db_helper.get_players_with_last5()
    '''
    dati ha questa struttura:
    [
    {
        'id_player': 1,
        'dati': [
            (78, 1200, 4.5, 105),
            (80, 1300, 4.7, 104),
            (76, 1100, 4.2, 103),
            (82, 1250, 4.6, 102),
            (79, 1180, 4.4, 101)
        ]
    },
    {
        'id_player': 2,
        'dati': [
            (90, 1500, 5.0, 205),
            (88, 1480, 4.9, 204),
            (92, 1520, 5.1, 203),
            (89, 1490, 5.0, 202),
            (91, 1510, 5.2, 201)
        ]
    },
          .......
    ]
    '''
    return dati

'''
CODICI JSON
200 messaggio inviato
201 messaggio ricevuto
500 errore
'''

def nao_start():
    nao_volume_sound(80)
    nao_autonomous_life()
    nao_eye_white()
    nao_wakeup()
    
    nao_animatedSayText("Ciao sono Peara!")
    
    nao_stand()
    if face_tracker:
        nao_face_tracker()
    else:
        nao_stop_face_tracker()

if __name__ == "__main__":
    startTime  = time.time()
    #nao_autonomous_life_state()

    db_helper.insert_dati(1, 150, 1320, 4.5)
    db_helper.insert_dati(1, 153, 1350, 4.6)
    db_helper.insert_dati(1, 151, 1330, 4.4)
    db_helper.insert_dati(1, 149, 1300, 4.3)
    db_helper.insert_dati(1, 155, 1370, 4.7)

    db_helper.insert_dati(2, 165, 1460, 5.0)
    db_helper.insert_dati(2, 168, 1490, 5.1)
    db_helper.insert_dati(2, 170, 1510, 5.2)
    db_helper.insert_dati(2, 166, 1470, 5.0)
    db_helper.insert_dati(2, 169, 1500, 5.1)

    db_helper.insert_dati(3, 142, 1150, 3.9)
    db_helper.insert_dati(3, 144, 1160, 4.0)
    db_helper.insert_dati(3, 140, 1130, 3.8)
    db_helper.insert_dati(3, 143, 1140, 3.9)
    db_helper.insert_dati(3, 145, 1170, 4.0)

    db_helper.insert_dati(4, 160, 1280, 4.4)
    db_helper.insert_dati(4, 162, 1300, 4.5)
    db_helper.insert_dati(4, 158, 1260, 4.3)
    db_helper.insert_dati(4, 161, 1290, 4.4)
    db_helper.insert_dati(4, 159, 1270, 4.3)

    db_helper.insert_dati(5, 134, 960, 3.3)
    db_helper.insert_dati(5, 136, 980, 3.4)
    db_helper.insert_dati(5, 133, 950, 3.2)
    db_helper.insert_dati(5, 135, 970, 3.3)
    db_helper.insert_dati(5, 137, 990, 3.4)

    #nao_start()
    #nao_autonomous_life()
    #nao_eye_white()
    #nao_wakeup()
    
    #nao_tts_audiofile("speech01.mp3")
    #nao_touch_head_audiorecorder()
    #nao_audiorecorder(5)
    #nao_train_move()

    #db_helper.insert_utente("nabihaseeb","ciao","nabi","haseeb")
    #logger.info("Result query: %s , id=%s", oggetto, id)

    #utenti = db_helper.select_utenti()

    #prova = db_helper.create_tables()
    #prova1 = db_helper.insert_player(12,49,0.27,0.30,"red")

    app.secret_key = os.urandom(12)
    app.run(host=config_helper.srv_host, port=config_helper.srv_port, debug=config_helper.srv_debug)
 