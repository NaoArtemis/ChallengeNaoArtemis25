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
#variabili globali
recording = False
recording_thread = None
video_writer = None
MODEL_PLAYERS = None
MODEL_LINES = None
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path_mp4 = os.path.join(script_dir, "recordings", "partita", "partita.mp4")

#inzializzazione modelli
def inizializza_modelli_postpartita():
    """Inizializza YOLOv8 per giocatori e linee."""
    global MODEL_PLAYERS, MODEL_LINES
    model_dir = os.path.join(script_dir, "models")
    MODEL_PLAYERS = YOLO(os.path.join(model_dir, "yolov8n_persone.pt"), verbose=False)
    MODEL_LINES = YOLO(os.path.join(model_dir, "yolo_lines_8n.pt"), verbose=False)

#recupero e salvataggio del video dal nao
def start_video_recording_from_nao():
    global recording, video_writer

    #se vuoi usare nao
    #data = {"nao_ip": nao_ip, "nao_port": nao_port}
    #url = f"http://127.0.0.1:5011/nao_webcam/{str(data)}"
    #response = requests.get(url, stream=True)

    #se vuoi usare webcam
    response = webcam_usb()

    # MJPEG frame boundaries
    boundary = b'--frame\r\n'
    content_type = b'Content-Type: image/jpeg\r\n\r\n'
    frame_data = b''

    # setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # mp4 format
    fps = 20.0
    frame_size = (640, 480)
    video_writer = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))


    recording = True
    for chunk in response.iter_content(chunk_size=1024):
        if not recording:
            break
        frame_data += chunk
        if boundary in frame_data:
            parts = frame_data.split(boundary)
            for part in parts[:-1]:
                if content_type in part:
                    jpg = part.split(content_type)[-1]
                    np_frame = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                    if frame is not None:
                        video_writer.write(frame)
            frame_data = parts[-1]

    video_writer.release()

#formazione vornoi soccer
def genera_video_voronoi():
    """Genera video Voronoi dal database PostgreSQL."""
    output_path = os.path.join(script_dir, "recordings", "voronoi", "voronoi_video.mp4")
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    frame_count = db_helper.get_total_frames()
    for frame_id in range(frame_count):
        rows = db_helper.select_player_positions_by_frame(frame_id)
        if len(rows) < 4:
            continue

        coords = np.array([[row[0], row[1]] for row in rows])
        teams = [row[2] for row in rows]

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlim(0, 30)
        ax.set_ylim(15, 0)
        ax.set_facecolor("#a5bc94")

        vor = Voronoi(coords)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='white')
        for (x, y), team in zip(coords, teams):
            color = 'blue' if team == 'blue' else 'red'
            ax.plot(x, y, 'o', color=color)

        ax.axis('off')
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, (640, 480))
        video_writer.write(img)
        plt.close(fig)

    video_writer.release()


#generazione heatmap
def genera_heatmap_giocatori():
    """Crea heatmap dei giocatori usando db PostgreSQL."""
    os.makedirs("recordings/heat_map", exist_ok=True)
    players = db_helper.select_players()

    for player_id in players:
        rows = db_helper.select_positions_by_player(player_id)
        if not rows:
            continue

        x_coords, y_coords = zip(*rows)
        plt.figure(figsize=(6.4, 4.8))
        ax = sns.kdeplot(x=x_coords, y=y_coords, fill=True, cmap="Reds", bw_adjust=0.5, thresh=0.05)
        ax.set_xlim(0, 30)
        ax.set_ylim(15, 0)
        ax.set_facecolor("#a5bc94")
        plt.axis('off')
        plt.savefig(f"recordings/heat_map/{player_id}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


#analisi della partita(video catturato)
def analizza_partita():
    global db_helper
    inizializza_modelli_postpartita()

    output_path = "recordings/annotato.mp4"
    
    if not os.path.exists(video_path_mp4):
        print(f"Errore: il file {video_path_mp4} non esiste")
        return

    cap = cv2.VideoCapture(video_path_mp4)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video {video_path_mp4}")
        return

    if not cap.isOpened():
        print("Errore: impossibile aprire il video")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = sv.ByteTrack()
    ellipse_annotator = EllipseAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
    label_annotator   = LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_position=sv.Position.BOTTOM_CENTER, text_scale=0.5)

    homography_matrix = None
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ##########################
        # STEP 1: omografia da linee
        ##########################
        if homography_matrix is None:
            line_results = MODEL_LINES(frame, conf=0.3)
            line_detections = sv.Detections.from_ultralytics(line_results[0])

            line_coords = []
            for box in line_detections.xyxy[:4]:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                line_coords.append([cx, cy])

            if len(line_coords) == 4:
                pixel_np = np.array(line_coords, dtype=np.float32)
                real_np = np.array([[0,0], [30,0], [0,15], [30,15]], dtype=np.float32)  # campo futsal 30x15 m
                homography_matrix, _ = cv2.findHomography(pixel_np, real_np)
                print("✅ Omografia calcolata")

        ##########################
        # STEP 2: YOLO su giocatori/palla
        ##########################
        results = MODEL_PLAYERS(frame, conf=0.2)
        detections = sv.Detections.from_ultralytics(results[0])
        ball_detections = detections[detections.class_id == 0]
        player_detections = detections[detections.class_id != 0].with_nms(threshold=0.5, class_agnostic=True)

        if len(player_detections) == 0:
            out.write(frame)
            continue

        ##########################
        # STEP 3: tracking + KMeans
        ##########################
        tracked = tracker.update_with_detections(player_detections)
        colors = []

        for xyxy in tracked.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                colors.append([0, 0, 0])
                continue
            avg_color = np.mean(crop.reshape(-1, 3), axis=0)
            colors.append(avg_color)

        if len(colors) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10).fit(colors)
            teams = ["blue" if label == 0 else "red" for label in kmeans.labels_]
        else:
            # se c'è solo 1 giocatore lo mettiamo in una squadra "unknown"
            teams = ["unknown"] * len(colors)

        tracked.data["team"] = teams



        kmeans = KMeans(n_clusters=2, n_init=10).fit(colors)
        teams = ["blue" if label == 0 else "red" for label in kmeans.labels_]
        tracked.data["team"] = teams

        ##########################
        # STEP 4: Annotazione + salvataggio
        ##########################
        labels = [f"#{tid} ({team})" for tid, team in zip(tracked.tracker_id, teams)]
        annotated = ellipse_annotator.annotate(scene=frame, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

        # Annotazione e salvataggio palla
        if len(ball_detections) > 0:
            ball_box = ball_detections.xyxy[0]
            x_center = float((float(ball_box[0]) + float(ball_box[2])) / 2)
            y_bottom = float(ball_box[3])

            triangle_annotator = sv.TriangleAnnotator(color=sv.Color.RED)
            annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)

            if homography_matrix is not None:
                pt = np.array([x_center, y_bottom, 1])
                transformed = homography_matrix @ pt
                transformed /= transformed[2]
                x_real = float(transformed[0])
                y_real = float(transformed[1])
            else:
                x_real, y_real = x_center, y_bottom

            db_helper.insert_player("ball", 0, x_real, y_real, "none")

        # Salvataggio giocatori
        for i, box in enumerate(tracked.xyxy):
            x_pixel = float((float(box[0]) + float(box[2])) / 2)
            y_pixel = float(box[3])
            pt = np.array([x_pixel, y_pixel, 1])
            if homography_matrix is not None:
                transformed = homography_matrix @ pt
                transformed /= transformed[2]
                x_real = float(transformed[0])
                y_real = float(transformed[1])
            else:
                x_real, y_real = x_pixel, y_pixel

            player_id = f"player_{tracked.tracker_id[i]}"
            db_helper.insert_player(player_id, 0, x_real, y_real, teams[i])

        out.write(annotated)
        frame_index += 1

    cap.release()
    out.release()

    #generazione vornoi
    genera_video_voronoi()
    #generazione heatmap
    genera_heatmap_giocatori()

    shutil.copy(os.path.join(script_dir, "recordings", "voronoi", "voronoi_video.mp4"),
            os.path.join(script_dir, "static", "voronoi_video.mp4"))

    
# per recuperare stream video da webcam
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



#funzioni flask
@app.route('/inizia_partita', methods=['GET'])
def inizia_partita():
    """Endpoint per avviare registrazione video."""
    global recording, recording_thread
    if recording:
        return jsonify({"status": "error", "message": "Registrazione già in corso"}), 400

    recording_thread = threading.Thread(target=start_video_recording_from_nao)
    recording_thread.start()
    return jsonify({"status": "ok", "message": "Sto registrando la partita dal NAO..."}), 200

@app.route('/fine_partita', methods=['GET'])
def fine_partita():
    """Endpoint per fermare registrazione."""
    global recording, recording_thread
    if not recording:
        return jsonify({"status": "error", "message": "Nessuna registrazione attiva"}), 400

    recording = False
    if recording_thread is not None:
        recording_thread.join()

    return jsonify({"status": "ok", "message": "Registrazione terminata. Puoi ora avviare l'analyze."}), 200

@app.route('/analyze', methods=['GET'])
def analyze():
    threading.Thread(target=analizza_partita).start()
    return jsonify({"status": "ok", "message": "Analisi avviata. Attendere qualche minuto..."}), 200

@app.route('/stream_voronoi')
def stream_voronoi():
    """Restituisce il video Voronoi appena generato."""
    path = os.path.join(script_dir, "recordings", "voronoi", "voronoi_video.mp4")
    if os.path.exists(path):
        return send_file(path, mimetype='video/mp4')
    else:
        return "Video non trovato", 404

@app.route('/api/players', methods=['GET'])
def api_players():
    """Restituisce lista dei player_id rilevati."""
    players = db_helper.select_players()
    return jsonify(players)

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
    players = db_helper.select_players()
    return render_template('partita.html', players=players)

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
 