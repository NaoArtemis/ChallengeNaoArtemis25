# -*- coding: utf-8 -*-

'''
python3 64 bit
pip3 install -r requirements.txt
python3 main.py

NOTA BENE!!!  
Avviare prima server Nao: python2 main.py
Il server nao comunica con il Nao attraverso python2.
Questo server si interfaccia con l'utente, il database e AI attraverso python3.
''' 

# Modules
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from hashlib import md5, sha256
from datetime import datetime
from selenium import webdriver
import requests
import time
import os
import cv2
import numpy as np
import utilities
from helpers.config_helper import Config
from helpers.logging_helper import logger
from helpers.speech_recognition_helper import SpeechRecognition
from helpers.db_helper import DB


config_helper  = Config()
db_helper      = DB(config_helper)

nao_ip         = config_helper.nao_ip
nao_port       = config_helper.nao_port
nao_user       = config_helper.nao_user
nao_password   = config_helper.nao_password

nao_ip_2       = config_helper.nao_ip_2
nao_port_2     = config_helper.nao_port_2
nao_user_2     = config_helper.nao_user_2
nao_password_2 = config_helper.nao_password_2

path_drivers   = config_helper.srv_drivers

face_detection = True
face_tracker   = True
local_db_morphcast = []
local_rec          = []
local_db_dialog    = []
local_gioiello_consigliato = []


app  = Flask(__name__)

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


def carenza(where, product_name):
    if (where == "scaffale"):
        return "È terminato " + str(product_name) + " sullo scaffale."
    elif (where == "magazzino"):
        return "Il " + str(product_name) + " è quasi terminato in magazzino."
    else:
        return ""
    

def conferma_ordine(id_ordine, nome, cognome, prezzo_totale):
    # nao1
    return "L'ordine " + str(id_ordine) + " è stato confermato! " + str(nome) + " " + str(cognome) + " devi recarti in cassa e pagare " + str(prezzo_totale) + " euro. Grazie e arrivederci!"

def prepara_ordine(product_name, nome, cognome):
    # nao2
    return "Il cliente " + str(nome) + " " + str(cognome) + " ha ordinato " + str(product_name) + ". Arriverà in cassa per ritirarlo!"


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


def nao_speech_to_text(sec_sleep):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "nao_user":nao_user, "nao_password":nao_password, "sec_sleep":sec_sleep}
    url      = "http://127.0.0.1:5011/nao_get_audio/" + str(data) 
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
    
    logger.info("nao_speech_to_text: " + str(speech_recognition.result))
    return str(speech_recognition.result)


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


def nao_autonomous_life():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_autonomous_life/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                                                     


def nao2_autonomous_life():
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2}
    url      = "http://127.0.0.1:5011/nao_autonomous_life/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                                                     


def nao_wakeup():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_wakeup/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                                                   


def nao2_wakeup():
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2}
    url      = "http://127.0.0.1:5011/nao_wakeup/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))     


def nao_eye_white():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_eye_white/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao2_eye_white():
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2}
    url      = "http://127.0.0.1:5011/nao_eye_white/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_animatedSayText(text_to_say):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "text_to_say":text_to_say}
    url      = "http://127.0.0.1:5011/nao_animatedSayText/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao2_animatedSayText(text_to_say):
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2, "text_to_say":text_to_say}
    url      = "http://127.0.0.1:5011/nao_animatedSayText/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_standInit():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_standInit/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))                      


def nao2_standInit():
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2}
    url      = "http://127.0.0.1:5011/nao_standInit/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))  


def nao_stand():
    data     = {"nao_ip":nao_ip, "nao_port":nao_port}
    url      = "http://127.0.0.1:5011/nao_stand/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao2_stand():
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2}
    url      = "http://127.0.0.1:5011/nao_stand/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_volume_sound(volume_level):
    data     = {"nao_ip":nao_ip, "nao_port":nao_port, "volume_level":volume_level}
    url      = "http://127.0.0.1:5011/nao_volume_sound/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao2_volume_sound(volume_level):
    data     = {"nao_ip":nao_ip_2, "nao_port":nao_port_2, "volume_level":volume_level}
    url      = "http://127.0.0.1:5011/nao_volume_sound/" + str(data) 
    response = requests.get(url, json=data)
    logger.info(str(response.text))


def nao_dialog():
    dialogo = []

    text_to_say = "Come ti chiami?"
    nao_animatedSayText(text_to_say)
    dialogo.append({"nao":text_to_say})
    response = nao_speech_to_text(5)
    dialogo.append({"umano":response})

    text_to_say = "Come stai?"
    nao_animatedSayText(text_to_say)
    dialogo.append({"nao":text_to_say})
    response = nao_speech_to_text(5)
    dialogo.append({"umano":response})

    local_db_dialog.append(dialogo)




# PAGINE WEB
# Per impedire all'utente di tornare indietro dopo aver fatto il logout
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

class User(UserMixin):
    def __init__(self, id = None, username = ''):
        self.id = id

users = {'1': {'id': '1', 'username': 'admin', 'password': '21232f297a57a5a743894a0e4a801fc3'}, #md5(admin)
         '2': {'id': '2', 'username': 'naonexus', 'password': '898d0dc0895b537fc1732a03cba7aff4'}} #md5(naonexus)

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = make_md5(request.form["password"])

        # Verifica credenziali utente
        user = next((u for u in users.values() if u['username'] == username and u['password'] == password), None)
        if user:
            user_obj = User(user['id'])
            login_user(user_obj)
            return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route("/logout", methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route("/dashboard", methods=['GET', 'POST'])

def dashboard():
    if request.method == "POST":
        pagina = request.form.get("pagina")
        if pagina:
            return redirect(url_for(pagina))
    oggetto = db_helper.get_oggetto()
    return render_template("dashboard.html", oggetto=oggetto)


@app.route('/analisi_morphcast', methods=['GET'])
def analisi_morphcast():
    return render_template('analisi_morphcast.html')

'''
@app.route('/webcam', methods=['GET'])
def webcam():
    return Response(nao_generate_frames(face_detection), mimetype='multipart/x-mixed-replace; boundary=frame')
'''

@app.route("/abbinamenti", methods=['GET'])
@login_required
def abbinamenti():
    oggetti = db_helper.get_oggetto()
    abbinamenti = db_helper.get_abbinamento()
    lista_oggetti = []
    for abb in abbinamenti:
        id_oggetto1 = db_helper.get_id_oggetto(abb['id_oggetto1'])
        id_oggetto2 = db_helper.get_id_oggetto(abb['id_oggetto2'])
        lista_oggetti.append({'id':abb['id'],'oggetto1':id_oggetto1,'oggetto2':id_oggetto2})
    return render_template("abbinamenti.html", oggetti=oggetti, lista_oggetti=lista_oggetti)


@app.route("/gestione_prodotto/<id>", methods=['GET'])
@login_required
def gestione_prodotto(id):
    if (id != None and id != ''):
        try:
            prodotto = db_helper.get_id_oggetto(id)
            return render_template('gestione_prodotto.html', prodotto=prodotto)
        except Exception as e:
            logger.error(str(e))
            return redirect("/dashboard", code=500)
    return redirect("/dashboard", code=500)


@app.route("/nuovo_prodotto", methods=['GET'])
@login_required
def nuovo_prodotto():
    prodotto = db_helper.get_id_oggetto(0)
    return render_template('gestione_prodotto.html', prodotto=prodotto)


@app.route("/prodotti", methods=['GET'])
@login_required
def prodotti():
    ordine_oggetto = db_helper.get_ordine_oggetto()
    oggetti = db_helper.get_oggetto()
    ordini = db_helper.get_ordine()

    vendite = {}
    for oggetto in oggetti:
        id_oggetto  = oggetto['id']
        titolo      = oggetto['titolo']
        vendite[id_oggetto] =  { 'id_oggetto'    : id_oggetto,
                                 'titolo'        : titolo,
                                 'data_acquisto' : []
                                }
        for item in ordine_oggetto:
            if (id_oggetto == item['id_oggetto']):
                id_ordine  = item['id_ordine']
                for ordine in ordini:
                    if(id_ordine == ordine['id']):
                        data_acquisto = ordine['data_acquisto']
                        data_acquisto = data_acquisto.strftime('%d/%m/%Y')
                        vendite[id_oggetto]['data_acquisto'].append(data_acquisto)
    
    return render_template("prodotti.html", vendite=vendite)


@app.route("/vendite")
@login_required
def vendite():
    ordine_oggetto_cliente = db_helper.get_ordine_oggetto_cliente()
    vendite = {}
    for item in ordine_oggetto_cliente:
        oggetto = item['oggetto']
        data_acquisto = item['ordine']['data_acquisto']
        data_acquisto = data_acquisto.strftime('%d/%m/%Y')
        if(data_acquisto not in vendite):
            vendite[data_acquisto] = []
        vendite[data_acquisto].append(oggetto)
    data_prezzo = {}
    for k, v in vendite.items():
        if (k not in data_prezzo):
            data_prezzo[k] = 0
        for item in v:
            data_prezzo[k] += float(item['prezzo'])

    return render_template("vendite.html", data_prezzo=data_prezzo)


@app.route("/utenti")
@login_required
def utenti():
    cliente = db_helper.get_cliente()
    count_cliente = db_helper.get_count_cliente()
    return render_template("utenti.html", cliente=cliente, count_cliente=count_cliente)


@app.route("/gestione_utente/<id>", methods=['GET'])
@login_required
def gestione_utente(id):
    if (id != None and id != ''):
        try:
            cliente = db_helper.get_id_cliente(id)
            return render_template('gestione_utente.html', cliente=cliente)
        except Exception as e:
            logger.error(str(e))
            return redirect("/utenti", code=500)
    return redirect("/utenti", code=500)


@app.route("/nuovo_utente", methods=['GET'])
@login_required
def nuovo_utente():
    cliente = db_helper.get_id_cliente(0)
    return render_template('gestione_utente.html', cliente=cliente)
        

@app.route("/carrelli", methods=['GET'])
@login_required
def carrelli():
    carrello = db_helper.get_carrello()
    carrello_oggetto = db_helper.get_carrello_oggetto()
    result = []
    for item in carrello:
        key   = 'carrello'
        value = {'id':item['id']}
        oggetto = []
        for index in carrello_oggetto:
            if(item['id'] == index['carrello']['id']):
                oggetto.append(index['oggetto'])

            if(item['id_cliente'] == index['cliente']['id']):
                if('cliente' not in value):
                    value['cliente'] = index['cliente']

        value['oggetto'] = oggetto
        if (len(oggetto) > 0):
            result.append({key:value})
    carrello_oggetto = result
    count_carrello = 0
    for item in carrello_oggetto:
        totale_carrello = 0
        if(len(item['carrello']['oggetto']) != 0):
            count_carrello += 1
        for index in item['carrello']['oggetto']:
            totale_carrello += index['prezzo']
        item['carrello']['totale'] = totale_carrello
    return render_template("carrelli.html", carrello_oggetto=carrello_oggetto, count_carrello=count_carrello)


@app.route("/scaffale", methods=['GET'])
@login_required
def scaffale():
    oggetto = db_helper.get_oggetto()
    count_scaffale = 0
    for item in oggetto:
        count_scaffale += item['qta_scaffale']
    return render_template("scaffale.html", oggetto=oggetto, count_scaffale=count_scaffale)


@app.route("/magazzino", methods=['GET'])
@login_required
def magazzino():
    oggetto = db_helper.get_oggetto()
    count_magazzino = 0
    for item in oggetto:
        count_magazzino += item['qta_magazzino']
    return render_template("magazzino.html", oggetto=oggetto, count_magazzino=count_magazzino)


@app.route("/emozioni", methods=['GET'])
@login_required
def emozioni():
    data = db_helper.get_emozioni()
    return render_template("emozioni.html", data=data)


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


@app.route('/api/morphcast', methods=['GET', 'POST'])
def api_morphcast():
    if request.method == 'GET':
        try:
            # estraggo l'ultimo valore dalla lista locale
            if len(local_db_morphcast) > 1:
                dati = local_db_morphcast[-1]
            else:
                dati = []
            return jsonify({'code': 200, 'message': 'OK', 'data': dati}), 200
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500
    elif request.method == 'POST':
        try:
            dati = request.get_json()   

            # salvo i dati in una lista locale                                
            local_db_morphcast.append(dati)

            return jsonify({'code': 201, 'message': 'OK', 'data': dati}), 201                  
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500


@app.route('/api/utente/<id>', methods=['POST', 'DELETE'])
def api_utente(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                username = request.form["username"]
                nome     = request.form["nome"]
                cognome  = request.form["cognome"]
                password = make_md5(request.form["password"])
                data     = db_helper.set_id_cliente(id,username,nome,cognome,password)
                return redirect("/utenti")  #return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_id_cliente(id,username,nome,cognome,password)}), 200
            except Exception as e:
                logger.error(str(e))
                return redirect("/utenti")  #return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'DELETE':
            try:
                data = db_helper.delete_carrello(id)
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.delete_cliente(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return redirect("/utenti")          #return jsonify({'code': 500, 'message': 'No id was passed'}), 500


@app.route('/api/nuovo_utente', methods=['POST'])
def api_nuovo_utente():
    if request.method == 'POST':
        try:
            username = request.form["username"]
            nome     = request.form["nome"]
            cognome  = request.form["cognome"]
            password = make_md5(request.form["password"])
            data, id = db_helper.set_cliente(username,nome,cognome,password)
            data     = db_helper.set_carrello(id)
            return redirect("/utenti")      #return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_cliente(username,nome,cognome,password)}), 200
        except Exception as e:
            logger.error(str(e))
            return redirect("/utenti")
    return redirect("/utenti")


@app.route('/api/catalogo', methods=['GET'])
def api_catalogo():
    if request.method == 'GET':
        try:
            return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_oggetto()}), 200
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500


@app.route('/api/prodotto/<id>', methods=['GET', 'POST', 'DELETE'])
def api_prodotto(id):
    if (id != None and id != ''):
        if request.method == 'GET':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_oggetto(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'POST':
            try:
                titolo              = request.form["titolo"]
                categoria           = request.form["categoria"]
                prezzo              = request.form["prezzo"]
                descrizione         = request.form["descrizione"]
                foto                = request.form["foto"]
                qta_magazzino       = request.form["qta_magazzino"]
                qta_scaffale        = request.form["qta_scaffale"]
                sconto              = request.form["sconto"]
                eta_consigliata     = request.form["eta_consigliata"]
                sesso_consigliato   = request.form["sesso_consigliato"]
                data = db_helper.set_id_oggetto(id,titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato)
                return redirect("/dashboard")  #return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_id_oggetto(id,titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato)}), 200
            except Exception as e:
                logger.error(str(e))
                return redirect("/dashboard")  #return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'DELETE':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.delete_oggetto(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return redirect("/dashboard")          #return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    

@app.route('/api/descrizione_prodotto/<id>', methods=['GET'])
def api_descrizione_prodotto(id):
    if (id != None and id != ''):
        if request.method == 'GET':
            try:
                #/api/descrizione_prodotto/id
                oggetto = db_helper.get_id_oggetto(id)
                testo = "Il prodotto " + str(oggetto['titolo']) + " è " + str(oggetto['descrizione'].encode('utf-8') + " e costa " + str(oggetto['prezzo'])) + " euro."
                nao_animatedSayText(testo)
                return jsonify({'code': 200, 'message': 'OK', 'data': oggetto}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500


@app.route('/api/nuovo_prodotto', methods=['POST'])
def api_nuovo_prodotto():
    if request.method == 'POST':
        try:
            titolo              = request.form["titolo"]
            categoria           = request.form["categoria"]
            prezzo              = request.form["prezzo"]
            descrizione         = request.form["descrizione"]
            foto                = request.form["foto"]
            qta_magazzino       = request.form["qta_magazzino"]
            qta_scaffale        = request.form["qta_scaffale"]
            sconto              = request.form["sconto"]
            eta_consigliata     = request.form["eta_consigliata"]
            sesso_consigliato   = request.form["sesso_consigliato"]
            data = db_helper.set_oggetto(titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato)
            return redirect("/dashboard")      #return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_oggetto(titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato)}), 200
        except Exception as e:
            logger.error(str(e))
            return redirect("/dashboard")
    return redirect("/dashboard")


@app.route('/api/qta_scaffale/<id>', methods=['PLUS', 'MINUS'])
def api_qta_scaffale(id):
    if (id != None and id != ''):
        if request.method == 'PLUS':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_qta_scaffale_plus(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'MINUS':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_qta_scaffale_minus(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500


@app.route('/api/qta_magazzino/<id>', methods=['PLUS', 'MINUS'])
def api_qta_magazzino(id):
    if (id != None and id != ''):
        if request.method == 'PLUS':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_qta_magazzino_plus(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'MINUS':
            try:
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.set_qta_magazzino_minus(id)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500


@app.route('/api/abbinamenti', methods=['POST'])
def api_abbinamenti():
    if request.method == 'POST':
        try:
            id_oggetto1 = request.form["productSelect1"]
            id_oggetto2 = request.form["productSelect2"]
            data = db_helper.set_abbinamento(id_oggetto1, id_oggetto2)
            return redirect("/abbinamenti")
        except Exception as e:
            logger.error(str(e))
            return redirect("/abbinamenti")
    return redirect("/abbinamenti")


@app.route('/api/delete_abbinamenti/<id>', methods=['DELETE'])
def api_delete_abbinamenti(id):
    if request.method == 'DELETE':
        try:
            data = db_helper.delete_abbinamento(id)
            return redirect("/abbinamenti")
        except Exception as e:
            logger.error(str(e))
            return redirect("/abbinamenti")
    return redirect("/abbinamenti")


@app.route('/api/scaffale', methods=['GET'])
def api_scaffale():
    oggetto = db_helper.get_oggetto()
    count_scaffale = 0
    for item in oggetto:
        count_scaffale = item['qta_scaffale']
        
        #inviare al nao2 oggetti da ripopolare
        if (count_scaffale < 1):
            text_to_say = carenza("scaffale", item['titolo'])
            nao2_animatedSayText(text_to_say)
            print(text_to_say)
    
    return redirect("/scaffale")


@app.route('/api/magazzino', methods=['GET'])
def api_magazzino():
    oggetto = db_helper.get_oggetto()
    count_magazzino = 0
    for item in oggetto:
        count_magazzino = item['qta_magazzino']

        #inviare al nao2 oggetti da ripopolare
        if (count_magazzino < 5):
            text_to_say = carenza("magazzino", item['titolo'])
            nao2_animatedSayText(text_to_say)
            print(text_to_say)
    text_to_say = "Siamo stati il più ecosostenibile possibile! L'ordine tiene conto anche di questo!"
    nao2_animatedSayText(text_to_say)
    print(text_to_say)
    
    return redirect("/magazzino")


@app.route('/api/carrello/<id>', methods=['GET', 'POST', 'DELETE'])
def api_carrello(id):
    if (id != None and id != ''):
        if request.method == 'GET':
            try:
                #/api/carrello/{"id_cliente":value}
                json = eval(id)
                id_cliente = json['id_cliente']
                id_carrello = db_helper.get_id_carrello(id_cliente)
                data = db_helper.get_id_carrello_oggetto(id_carrello['id'])
                if data:
                    return jsonify({'code': 200, 'message': 'OK', 'data': data}), 200
                else:
                    return jsonify({'code': 200, 'message': 'OK', 'data': 'no_data'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'POST':
            try:
                #{"id_cliente":value, "id_oggetto":value}
                json = request.json
                id_cliente  = json['id_cliente']
                id_oggetto  = json['id_oggetto']
                id_carrello = db_helper.get_id_carrello(id_cliente)
                msg = db_helper.set_carrellooggetto(id_carrello['id'], id_oggetto)
                return jsonify({'code': 200, 'message': 'OK', 'data': 'INSERT'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'DELETE':
            try:
                #{"id_cliente":value, "id_oggetto":value}
                json = eval(id)
                id_cliente  = json['id_cliente']
                id_oggetto  = json['id_oggetto']
                id_carrello = db_helper.get_id_carrello(id_cliente)
                msg = db_helper.delete_carrellooggetto(id_carrello['id'],id_oggetto)
                return jsonify({'code': 200, 'message': 'OK', 'data': 'DELETE'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500


@app.route('/api/ordine/<id>', methods=['GET', 'POST'])
def api_ordine(id):
    if (id != None and id != ''):
        if request.method == 'GET':
            try:
                #{"id_cliente":value}
                json = request.json
                id_cliente = json['id_cliente']
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_ordine(id_cliente)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'POST':
            try:
                #{"id_cliente":value, "modalita_pagamento":value}
                json = request.json
                id_cliente         = json['id_cliente']
                modalita_pagamento = json['modalita_pagamento']
                cliente            = db_helper.get_id_cliente(id_cliente)
                id_carrello        = db_helper.get_id_carrello(id_cliente)
                id_carrello        = id_carrello['id']
                carrello           = db_helper.get_id_carrello_oggetto(id_carrello)
                prezzo_totale = 0
                for item in carrello:
                    prezzo_totale += float(item['oggetto']['prezzo'])
                data_acquisto = datetime.now().date()
                ora_acquisto  = datetime.now().strftime("%H:%M:%S")
                id_ordine = db_helper.set_ordine(id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento)
                for item in carrello:
                    id_oggetto = item['oggetto']['id']
                    id_ordineoggetto = db_helper.set_ordineoggetto(id_ordine, id_oggetto)

                # nao1 fa il riepilogo dell'ordine appena confermato
                text_to_say = conferma_ordine(id_ordine, cliente['nome'], cliente['cognome'], prezzo_totale)
                nao_animatedSayText(text_to_say)
                print(text_to_say)
                
                # eseguito l'ordine elimino il contenuto del carrello
                for item in carrello:
                    id_oggetto = item['oggetto']['id']
                    data = db_helper.delete_carrellooggetto(id_carrello, id_oggetto)
                    
                    # il nao2 avvisa il commesso di preparare l'ordine e toglie l'oggetto dallo scaffale
                    text_to_say = prepara_ordine(item['oggetto']['titolo'], cliente['nome'], cliente['cognome'])
                    nao2_animatedSayText(text_to_say)
                    print(text_to_say)

                    data = db_helper.set_qta_scaffale_minus(id_oggetto)
                    text_to_say = carenza("scaffale", item['oggetto']['titolo'])
                    nao2_animatedSayText(text_to_say)
                    print(text_to_say)
                
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_ordine(id_cliente)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    

@app.route('/api/ordine_mobile/<id>', methods=['GET', 'POST'])
def api_ordine_mobile(id):
    if (id != None and id != ''):
        if request.method == 'GET':
            try:
                #{"id_cliente":value}
                json = request.json
                id_cliente  = json['id_cliente']
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_ordine(id_cliente)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        elif request.method == 'POST':
            try:
                #{"id_cliente":value, "modalita_pagamento":value}
                json = eval(id)
                id_cliente         = json['id_cliente']
                modalita_pagamento = json['modalita_pagamento']
                cliente            = db_helper.get_id_cliente(id_cliente)
                id_carrello        = db_helper.get_id_carrello(id_cliente)
                id_carrello        = id_carrello['id']
                carrello           = db_helper.get_id_carrello_oggetto(id_carrello)
                prezzo_totale = 0
                for item in carrello:
                    prezzo_totale += float(item['oggetto']['prezzo'])
                data_acquisto = datetime.now().date()
                ora_acquisto  = datetime.now().strftime("%H:%M:%S")
                id_ordine = db_helper.set_ordine(id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento)
                for item in carrello:
                    id_oggetto = item['oggetto']['id']
                    id_ordineoggetto = db_helper.set_ordineoggetto(id_ordine, id_oggetto)
                
                # eseguito l'ordine elimino il contenuto del carrello
                for item in carrello:
                    id_oggetto = item['oggetto']['id']
                    data = db_helper.delete_carrellooggetto(id_carrello, id_oggetto)
                    
                    # il nao2 avvisa il commesso di preparare l'ordine e toglie l'oggetto dallo scaffale
                    text_to_say = prepara_ordine(item['oggetto']['titolo'], cliente['nome'], cliente['cognome'])
                    nao2_animatedSayText(text_to_say)
                    print(text_to_say)

                    data = db_helper.set_qta_scaffale_minus(id_oggetto)
                    text_to_say = carenza("scaffale", item['oggetto']['titolo'])
                    nao2_animatedSayText(text_to_say)
                    print(text_to_say)
                
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_ordine(id_cliente)}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    

@app.route('/api/utente/app/<id>', methods=['POST'])
def api_utente_app(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"username":value, "password":value}
                json = request.json
                username = json["username"]
                password = make_md5(json["password"])
                data = db_helper.get_account_cliente(username, password)
                return jsonify({'code': 200, 'message': 'OK', 'data': data}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    

@app.route('/api/utente/dialogo/<id>', methods=['POST'])
def api_utente_dialogo(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                print("POST /api/utente/dialogo/", id)
                #{"id_cliente":value, "nome":value, "cognome":value}
                json = request.json
                id_cliente  = json["id_cliente"]
                nome        = json["nome"]
                cognome     = json["cognome"]
                id_carrello = db_helper.get_id_carrello(id_cliente)

                import dialogo_decision_tree

                # inizio servizio browser
                services_start_morphcast()

                # preparazione nao con face tracker
                '''
                nao_volume_sound(80)
                nao_autonomous_life("disabled")
                nao_eye_white()
                nao_wakeup()
                nao_stand()
                if face_tracker:
                    nao_face_tracker()
                else:
                    nao_stop_face_tracker()
                '''

                # dialogo con decision tree
                backup_nao = True
                dt = dialogo_decision_tree.Dialogo(db_helper, id_cliente, nome, cognome, id_carrello, backup_nao)

                # fine servizio browser
                services_stop_morphcast()

                # salvo il dialogo
                local_db_dialog.append(dt.dialog)

                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_carrello_oggetto(id_carrello['id'])}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    

@app.route('/api/post_gioiello_consigliato/<id>', methods=['POST'])
def api_post_gioiello_consigliato(id):
    if (id != None and id != ''):
        if request.method == 'POST':
            try:
                #{"id_oggetto": value}
                json = request.json
                id_oggetto = json["id_oggetto"]
                local_gioiello_consigliato.append(id_oggetto)
                return jsonify({'code': 200, 'message': 'OK', 'data': id_oggetto}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
    else:
        logger.error('No id argument passed')
        return jsonify({'code': 500, 'message': 'No id was passed'}), 500
    
            
@app.route('/api/get_gioiello_consigliato', methods=['GET'])
def api_get_gioiello_consigliato():
    if request.method == 'GET':
        try:
            if len(local_gioiello_consigliato) > 0:
                id_oggetto = local_gioiello_consigliato[0]
                local_gioiello_consigliato.remove(id_oggetto)
                return jsonify({'code': 200, 'message': 'OK', 'data': db_helper.get_id_oggetto(id_oggetto)}), 200
            else:
                return jsonify({'code': 200, 'message': 'OK', 'data': None}), 200
        except Exception as e:
            logger.error(str(e))
            return jsonify({'code': 500, 'message': str(e)}), 500
        

# SERVICES
@app.route('/services', methods=['GET'])
def services():
    return render_template('services.html')

# Servizio START morphcast
@app.route('/services/start_morphcast', methods=['GET'])
def services_start_morphcast():
    # Aprire il link nel browser predefinito
    try:
        global driver
        driver = webdriver.Chrome(path_drivers)
        driver.get("http://127.0.0.1:5010/analisi_morphcast")
        return jsonify({'code': 200, 'message': 'START morphcast'}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({'code': 500, 'message': str(e)}), 500


# Servizio STOP morphcast
@app.route('/services/stop_morphcast', methods=['GET'])
def services_stop_morphcast():
    # Chiudere il link nel browser predefinito
    try:
        driver.close()
        return jsonify({'code': 200, 'message': 'STOP morphcast'}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({'code': 500, 'message': str(e)}), 500


# Servizio START STOP dialogo
@app.route('/services/start_dialogo', methods=['GET'])
def services_start_dialogo():
    try:
        nao_dialog()
        return jsonify({'code': 200, 'message': 'dialogo completato', 'data': local_db_dialog}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({'code': 500, 'message': str(e)}), 500


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
    nao_animatedSayText("Ciao sono Peara, benvenuti al mio Riteil ti aiuterò con la scelta dei prodotti")
    nao_stand()
    if face_tracker:
        nao_face_tracker()
    else:
        nao_stop_face_tracker()

    nao2_volume_sound(80)
    nao2_autonomous_life()
    nao2_eye_white()
    nao2_wakeup()
    nao2_animatedSayText("Ciao sono Lesso, benvenuti al mio Riteil ti aiuterò nella gestione dei prodotti")
    nao2_stand()


if __name__ == "__main__":
    startTime  = time.time()
        
    #nao_start()

    app.secret_key = os.urandom(12)
    app.run(host=config_helper.srv_host, port=config_helper.srv_port, debug=config_helper.srv_debug)