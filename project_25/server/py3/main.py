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
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from hashlib import md5, sha256
from datetime import datetime
import requests
import time
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

#pre impostazione omografia
'''
molto importante posizionare bene gli aruco, una non lettura o poszione sbagliata può causare errore 
nell'applicazione della omografia cio può alterare im modo negativo la determinazione delle posizioni 
dei giocatori.
'''
#grazie a questo dizionario e aruco(id:1,2,3,4) riusciamo ad definire la matriche homography
id_to_coord = {
    1: (0, 0),       # Angolo in alto a sinistra
    2: (30, 0),      # Angolo in alto a destra
    3: (0, 15),      # Angolo in basso a sinistra
    4: (30, 15)      # Angolo in basso a destra
}

def inizializzazione():
    global partita_iniziata, partita_pausa, partita_secondo_tempo, partita_finita, start_time, tempo_fine_primo_tempo
    global MODEL_PLAYERS, tracker, processor, model, homography_matrix
    global TEAM_A_COLOR, TEAM_B_COLOR
    global omografia_pronta, task_2
    global frame_id, last_yolo_detections, last_yolo_frame

    # Stato partita
    partita_iniziata = False
    partita_pausa = False
    partita_secondo_tempo = False
    partita_finita = False
    start_time = None
    tempo_fine_primo_tempo = 0  # per gestire cambio campo

    ### Oggetti AI e tracking ###

    #inizializzazione modello yolo
    #se metto verbose true nella console python mi verebbe stampata l'analisi di ogni frame
    MODEL_PLAYERS = YOLO("project_25/server/py3/models/yolov8n_persone.pt", verbose=False) 
    tracker = sv.ByteTrack()

    #inizializzazione modello siglip
    processor = AutoProcessor.from_pretrained("project_25/server/py3/models/siglip")
    model = AutoModel.from_pretrained("project_25/server/py3/models/siglip")

    #matrice omografica
    homography_matrix = None
    #flag omografia
    omografia_pronta = False

    # Costanti
    TEAM_A_COLOR = "gray"
    TEAM_B_COLOR = "blue"

    #gestione analisi frame
    frame_id = 0
    last_yolo_detections = None
    last_yolo_frame = None

inizializzazione()

def global_variabili():
    global partita_iniziata, partita_pausa, partita_secondo_tempo, partita_finita, start_time, tempo_fine_primo_tempo
    global MODEL_PLAYERS, tracker, processor, model, homography_matrix
    global task_2
    global game_time, game_time_text, coords
    global ellipse_annotator, label_annotator, triangle_annotator
    global frame, np_frame, img_bytes, parts
    global x_center, y_bottom, ball_box
    global pixel_np, real_np, cx, cy, marker_ids
    global speech_file_path, speech_recognition
    global user_obj, client, username, password
    global frame_with_faces, x, y, x_speed, y_speed
    global frame_id, last_yolo_detections, last_yolo_frame
    global omografia_pronta
    # inizializzo se non esistono (protezione)
    if 'coords' not in globals(): coords = []
    if 'game_time' not in globals(): game_time = 0
    if 'game_time_text' not in globals(): game_time_text = "00:00"

def apply_homography(boxes):
    #omografia tecnica utilizzata per trasfromare coordinate pixel in coordinate reali
    global_variabili()
    coords = []
    if not omografia_pronta or homography_matrix is None or homography_matrix.shape != (3, 3):
        return coords  # ritorna lista vuota senza errore

    for box in boxes:
        x = (box[0] + box[2]) / 2
        y = box[3]
        pt = np.array([x, y, 1])
        transformed = homography_matrix @ pt
        transformed /= transformed[2]
        coords.append((transformed[0], transformed[1]))
    return coords

def classify_with_siglip(detections, frame):
    global_variabili()
    teams = []
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            teams.append("unknown")
            continue
        inputs_img = processor(images=crop, return_tensors="pt")
        inputs_text = processor(
            text=[f"a soccer player with {TEAM_A_COLOR} jersey", f"a soccer player with {TEAM_B_COLOR} jersey"],
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs_img)
            text_emb = model.get_text_features(**inputs_text)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ text_emb.T)[0]
        teams.append(TEAM_A_COLOR if sim[0] > sim[1] else TEAM_B_COLOR)
    detections.data["team"] = teams
    return detections

def analyze_frame(frame):
    start_time = time.time() # per contare gli fps
    global tracked, coords # le rendo globali oper vornoi_diagram
    global_variabili()

    global frame_id, last_yolo_detections, last_yolo_frame
    frame_id += 1
    run_yolo = frame_id % 10 == 0

    #se partita già iniziata calcolo il tempo, else imposto il tempo a 00:00
    if partita_iniziata and not partita_finita:
        # se siamo nel secondo tempo, aggiungiamo il tempo del primo tempo per mantenere continuità
        now_time = time.time()
        if start_time is not None:
            game_time = now_time - start_time + tempo_fine_primo_tempo if partita_secondo_tempo else now_time - start_time
        else:
            game_time = 0
        game_time_text = "PAUSA" if partita_pausa else f"{int(game_time // 60):02d}:{int(game_time % 60):02d}"
    else:
        game_time_text = "00:00"

    #applico il modello per la detection di persone
    
    if run_yolo:
        results_players = MODEL_PLAYERS(frame, conf=0.2)
        detections = sv.Detections.from_ultralytics(results_players[0])
        last_yolo_detections = detections
        last_yolo_frame = frame
    else:
        detections = last_yolo_detections if last_yolo_detections is not None else sv.Detections.empty()
    
    
    ball_detections = detections[detections.class_id == 0]
    player_detections = detections[detections.class_id != 0].with_nms(threshold=0.5, class_agnostic=True)
    

    
    if len(player_detections) > 0:
        tracked = tracker.update_with_detections(player_detections)
        tracked = classify_with_siglip(tracked, frame)
        coords = apply_homography(tracked.xyxy)

        #salvo le coordinate nel database
        for i, c in enumerate(coords):
            db_helper.insert_player(
                f"player_{tracked.tracker_id[i]}",
                game_time,
                c[0],
                c[1],
                tracked.data['team'][i]
            )
            logger.info("Result query: %s , id=%s", "player", tracked.tracker_id[i]) # funzione richimata dal db_helper

    else:
        tracked = player_detections

    if len(ball_detections) > 0:
        # Prende solo la prima palla rilevata, per non andare in conflitto con i palloni a bordo campo
        ball_box = ball_detections.xyxy[0]
        x_center = (ball_box[0] + ball_box[2]) / 2
        y_bottom = ball_box[3]

        pt = np.array([x_center, y_bottom, 1])
        transformed = homography_matrix @ pt
        transformed /= transformed[2]

        db_helper.insert_player(
            "ball",
            game_time,
            transformed[0],
            transformed[1],
            "none"
        )
        logger.info("Result query: %s , id=%s", "ball", "ball")
    
    annotated = frame
    
    if len(tracked) > 0:
        labels = [f"#{id} ({team})" for id, team in zip(tracked.tracker_id, tracked.data["team"])]
        ellipse_annotator = sv.EllipseAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_position=sv.Position.BOTTOM_CENTER, text_scale=0.5)
        annotated = ellipse_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
    if len(ball_detections) > 0:
        triangle_annotator = sv.TriangleAnnotator(color=sv.Color.RED, thickness=2)
        annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
    
    cv2.putText(annotated, f"Game Time: {game_time_text}", (10, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 255, 255), 2)
    
    # Calcola il tempo impiegato
    elapsed_time = (time.time() - start_time) * 1000  # lo trasformo in millisecondi
    latency_text = f"Latenza: {elapsed_time:.2f} ms"
    # Scrive il tempo impiegato sull'immagine
    cv2.putText(annotated, str(latency_text), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)             
    return annotated


class WebcamUSBResponseSimulator:
    def __init__(self, cam_index=0): # cam_index = 0, prima webcam dispobinile, per ecellenza quella integrata
        self.cap = cv2.VideoCapture(cam_index)

    def iter_content(self, chunk_size=1024):
        if not self.cap.isOpened():
            print("Errore: webcam non disponibile.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Ridimensiona il frame
            frame_resized = cv2.resize(frame, (320, 240))
            #frame_resized = cv2.flip(frame_resized, 1)
            # Codifica JPEG
            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) # la qualità va fino a 100
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Costruisci il chunk MJPEG
            chunk = (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Simula suddivisione in pacchetti
            for i in range(0, len(chunk), chunk_size):
                yield chunk[i:i+chunk_size]

        self.cap.release()


def webcam_usb():
    return WebcamUSBResponseSimulator()


# tutte queste funzioni vengono richiamate nella funzione computer_vision, flask endpoint:/computer_vision

#################################
#            Tribuna            #
#################################
# variabili blobali task 2
task_2 = False

def tempo_di_pausa():
    print("Ora aspetta")
    timer = threading.Timer(20, ritorno_a_false)
    timer.start()

def ritorno_a_false():
    task_2 = False

def nao_entusiasta():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_entusiasta/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))

def nao_points():
    global task_2
    tempo_di_pausa()


def nao_seat():
    global task_2
    text = "i posti a sedere riservati sono quelli nella prima fila con gli stiker blu"
    nao_animatedSayText(text)
    tempo_di_pausa()
    
def nao_stats():
    global task_2
    tempo_di_pausa()


def nao_coro():  
    global task_2
    text = "vincere"#haseeb non modificare!!!!
    nao_entusiasta()
    nao_SayText(text)
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
    #data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    #url      = "http://127.0.0.1:5011/nao_webcam/" + str(data) 
    #response = requests.get(url, json=data, stream=True)

    ### per recuperare frame tramite webcam collegata al pc ###
    response = webcam_usb()
    
    #recupero variabili
    # Inizializza il dizionario ArUco
    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters()

    # Inizializza il rilevatore ArUco
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # aruco detection

    
    def generate_frames():
        #recupero variabili globali
        global_variabili()

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

                            pixel_points = []
                            real_points = []

                            for i, marker_id in enumerate(marker_ids.flatten()):
                                if marker_id in id_to_coord:
                                    corners = marker_corners[i][0]
                                    cx = int(np.mean(corners[:, 0]))
                                    cy = int(np.mean(corners[:, 1]))
                                    pixel_points.append([cx, cy])
                                    real_points.append(id_to_coord[marker_id])

                            #riconosce i 4 aruco e assegna le coordinate omografice
                            global omografia_pronta
                            if len(pixel_points) == 4 and homography_matrix is None:
                                print("Marker 1-2-3-4 rilevati, provo a calcolare omografia")
                                pixel_np = np.array(pixel_points, dtype=np.float32)
                                real_np = np.array(real_points, dtype=np.float32)
                                homography_matrix, _ = cv2.findHomography(pixel_np, real_np)
                                print("Omografia calcolata:", homography_matrix)
                                omografia_pronta = True  # flag attivo, si può iniziare partita

                            # GESTIONE PARTITA
                            global partita_iniziata, partita_pausa, partita_secondo_tempo, partita_finita
                            if 180 in marker_ids and not partita_iniziata and omografia_pronta:
                                partita_iniziata = True
                                start_time = time.time()

                            # PAUSA PARTITA
                            
                            if 181 in marker_ids and not partita_pausa:
                                partita_pausa = True

                            # INIZIO SECONDO TEMPO
                            if 182 in marker_ids and not partita_secondo_tempo:
                                partita_secondo_tempo = True

                                # Inversione omografia se esiste (cambio campo)
                                if homography_matrix is not None:
                                    print("⚠️ Cambio campo: inverto la matrice omografica per secondo tempo")
                                    homography_matrix = np.linalg.inv(homography_matrix)
                            
                            #FINE PARTITA
                            if 183 in marker_ids and not partita_finita:
                                partita_finita = True

                            # TASK 2

                            #SCORE PARTITA
                            global task_2
                            if 184 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_points()

                            #Statistiche della partita
                            elif 185 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_stats() #time

                            #Posti a sedere
                            elif 186 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_seat()

                            #cori
                            elif 187 in marker_ids.flatten() and not task_2:
                                task_2 = True
                                nao_coro() #festeggiamo

                        #ricodifica e invia il frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                frame_data = parts[-1]

    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_omografia', methods=['GET']) #per risettare omografia, in caso di errore nella calibrazione
def reset_omografia():
    global homography_matrix, omografia_pronta
    homography_matrix = None
    omografia_pronta = False
    return jsonify({"status": "ok", "message": "Omografia resettata"}), 200

@app.route('/computer_vision', methods=['GET'])
def computer_vision():
    global_variabili()
    if MODEL_PLAYERS is None or tracker is None or processor is None or model is None:
        inizializzazione()
    flask_response = webcam_aruco()

    def generate():
        frame_data = b''
        for chunk in flask_response.response:
            frame_data += chunk
            if b'--frame\r\n' in frame_data:
                parts = frame_data.split(b'--frame\r\n')
                for part in parts[:-1]:
                    if b'Content-Type: image/jpeg\r\n\r\n' in part:
                        img_bytes = part.split(b'\r\n\r\n')[-1]
                        try:
                            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                analyzed = analyze_frame(frame)
                                _, buffer = cv2.imencode('.jpg', analyzed)
                                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                        except Exception as e:
                            print("Errore nell'elaborazione del frame:", e)
                frame_data = parts[-1]

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

matplotlib.use('Agg')  # Usa un backend non interattivo

@app.route('/voronoi_diagram', methods=['GET'])
def voronoi_diagram():
    global_variabili()
    if MODEL_PLAYERS is None or tracker is None or processor is None or model is None:
        inizializzazione()
    flask_response = webcam_aruco()

    def generate():
        frame_data = b''
        for chunk in flask_response.response:
            frame_data += chunk
            if b'--frame\r\n' in frame_data:
                parts = frame_data.split(b'--frame\r\n')
                for part in parts[:-1]:
                    if b'Content-Type: image/jpeg\r\n\r\n' in part:
                        img_bytes = part.split(b'\r\n\r\n')[-1]
                        try:
                            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                analyze_frame(frame)  # aggiorna tracked, team, coords

                                if 'tracked' not in globals() or 'coords' not in globals():
                                    continue
                                if len(coords) < 4:
                                    continue

                                teams = tracked.data["team"] if hasattr(tracked, 'data') and "team" in tracked.data else []

                                # Disegna campo e Voronoi
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.set_xlim(0, 30)
                                ax.set_ylim(15, 0)
                                ax.set_facecolor("#a5bc94")  # colore campo

                                points = np.array(coords)
                                vor = Voronoi(points)
                                voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='white')

                                for (x, y), team, pid in zip(coords, teams, tracked.tracker_id):
                                    color = TEAM_A_COLOR if team == TEAM_A_COLOR else TEAM_B_COLOR
                                    circle = Circle((x, y), 0.5, edgecolor='black', facecolor=color, linewidth=1.5)
                                    ax.add_patch(circle)
                                    ax.text(x, y, str(pid), color='white', ha='center', va='center', fontsize=8)

                                ax.axis('off')
                                buf = io.BytesIO()
                                plt.savefig(buf, format='jpg', bbox_inches='tight')
                                buf.seek(0)
                                frame_bytes = buf.read()
                                buf.close()
                                plt.close(fig)

                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                        except Exception as e:
                            print("Errore nel generare il Voronoi:", e)
                frame_data = parts[-1]

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')


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
    model=whisper.load_model("base")
    result = model.transcribe(local_path)
    ORDINE = result['text']
    print(ORDINE)


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

def nao_animatedSayText(text_to_say):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "text_to_say":text_to_say}
    url      = "http://127.0.0.1:5011/nao_animatedSayText/" + str(data) 
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

@app.route('/home2', methods=['GET']) # non definito
@login_required
def home2():
    return render_template('home2.html')

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
    return render_template('partita.html')

@app.route('/registra', methods = ['GET']) 
# tramite questa pagina creai nuovi utenti e salvi nel database tramite funzione api_app_utenti(id):
@login_required
def database():
    return render_template('registra.html')

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
        response = client.audio.speech.create(model="tts-1",voice="alloy",input=text)
        response.stream_to_file(speech_file_path)
        nao_tts_audiofile("speech.mp3")
        
    return redirect('/home')


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
    response = requests.get(url, json=data)
    logger.info(str(response.text))
    battery_info = response.json()
    battery_level = battery_info["battery_level"]
    return jsonify({'battery_level': battery_level}), 200



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
                #{"id_player:n, bpm:98, passi:72, velocità:4m/s"}
                json = request.json
                id_player = json["id_player"]
                bpm = json["password"]
                passi = json["passi"]
                velocità = json["velocita"]
                db_helper.insert_dati(id_player, bpm, passi, velocità)
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
                #{"id_player:n, ammonizone:False, infortunio:True"}
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


@app.route('/api/app/utenti/<id>', methods=['POST'])
def api_app_utenti(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"username:mt, password:0987, nome:marco, cognome:tomazzoli, posizione:laterale"}
                json = request.json
                username = json["username"]
                password = json["password"]
                nome = json["nome"]
                cognome = json["cognome"]
                posizione = json["posizione"]
                db_helper.insert_utente(username, password, nome, cognome, posizione)
                return jsonify({'code': 200, 'message': 'OK',}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            logger.error('No id argument passed')
            return jsonify({'code': 500, 'message': 'No id was passed'}), 500


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
 