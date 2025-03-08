# -*- coding: utf-8 -*-

'''
python2 64 bit
pip2 install -r requirements.txt
python2 main.py
''' 

# Modules
from naoqi import ALProxy
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from datetime import datetime
import cv2
import time
import numpy as np
import random
import os
import json
import math


import utilities
import config_helper
from logging_helper import logger
from speech_recognition_helper import SpeechRecognition


config_helper  = config_helper.Config()

nao_ip         = config_helper.nao_ip
nao_port       = config_helper.nao_port
nao_user       = config_helper.nao_user
nao_password   = config_helper.nao_password


face_detection = True
face_tracker   = True
local_db_morphcast = []
local_rec          = []


app  = Flask(__name__)


def nao_face_detection(image_data):
    face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     # Create a face cascade for detecting faces
    gray            = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)                                             # Convert the image to grayscale for face detection
    faces           = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))   # Detect faces in the image

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_data, (x, y), (x+w, y+h), (255, 0, 0), 2)


def nao_face_tracker():
    tracker_proxy = ALProxy("ALTracker", nao_ip, nao_port)
    targetName = "Face"
    faceWidth  = 0.1
    tracker_proxy.setMode("Head")
    tracker_proxy.registerTarget(targetName, faceWidth)
    tracker_proxy.track(targetName)


def nao_stop_face_tracker():
    tracker_proxy = ALProxy("ALTracker", nao_ip, nao_port)
    tracker_proxy.stopTracker()
    tracker_proxy.unregisterAllTargets()


def nao_generate_frames(face_detection):
    video_proxy = ALProxy("ALVideoDevice", nao_ip, nao_port)            # NAO webcam

    # Set the camera parameters
    name_id          = "video_image_" + str(random.randint(0,100))      # The same Name could be used only six time
    camera_id        = 1                                                # Use the top camera (1 for bottom camera)
    resolution       = 1                                                # Image of 320*240px
    color_space      = 13                                               # RGB
    camera_fps       = 30                                               # fps
    brightness_value = 55                                               # default of 55
    video_proxy.setParameter(camera_id, 0, brightness_value)            # brightness

    # Subscribe to the video feed
    video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, color_space, camera_fps)
    try:
        while True:
            image          = video_proxy.getImageRemote(video_client)   # Capture a frame from the camera
            
            image_width    = image[0]
            image_height   = image[1]
            image_channels = 3
            image_data     = np.frombuffer(image[6], dtype=np.uint8).reshape((image_height, image_width, image_channels))
            
            if face_detection:
                nao_face_detection(image_data)

            desired_width  = 640
            desired_height = 480
            resized_image  = cv2.resize(image_data, (desired_width, desired_height))

            ret, buffer    = cv2.imencode('.jpg', resized_image)            
            frame          = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logger.error(str(e))
    finally:
        video_proxy.unsubscribe(video_client)                               # Unsubscribe from the video feed and release resources


def nao_autonomous_life(state="disabled"):
    life_proxy = ALProxy("ALAutonomousLife", nao_ip, nao_port)              # Creare un proxy per il servizio Autonomous Life
    life_proxy.setState(state)                                              # Ferma il comportamento autonomo di base
    life_proxy = None                                                       # Chiudi eventuali connessioni


def nao_wakeup():
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)                    # Creare un proxy per il servizio Motion
    motion_proxy.wakeUp()                                                   # Attivare il robot dallo stato di "sleep"
    motion_proxy = None                                                     # Chiudi eventuali connessioni


def nao_eye_white():
    leds_proxy = ALProxy("ALLeds", nao_ip, nao_port)                        # Creare un proxy per il servizio ALLeds
    eye_group_name = "FaceLeds"                                             # Specificare il nome del gruppo LED degli occhi
    eye_color = [255, 255, 255]                                             # RGB values for white
    value_color = 256*256*eye_color[0] + 256*eye_color[1] + eye_color[2]
    leds_proxy.fadeRGB(eye_group_name, value_color, 0.5)                    # 0.5 is the duration in seconds
    leds_proxy = None                                                       # Chiudi la connessione


def nao_animatedSayText(text_to_say):
    tts_proxy = ALProxy("ALAnimatedSpeech", nao_ip, nao_port)               # Connect to the text-to-speech module             
    animated_speech_config = {"bodyLanguageMode": "contextual"}             # Set the parameter for animated speech
    tts_proxy.say(text_to_say, animated_speech_config)                      # Say the text with animated speech                                        
    tts_proxy = None      


def nao_standInit():
    posture_proxy = ALProxy("ALRobotPosture", nao_ip, nao_port)             # Connect to the ALRobotPosture module             
    posture_proxy.goToPosture("StandInit", 0.8)                             # Make the robot stand up
    posture_proxy = None                                                  
   

def nao_stand():
    posture_proxy = ALProxy("ALRobotPosture", nao_ip, nao_port)             # Connect to the ALRobotPosture module             
    posture_proxy.goToPosture("Stand", 0.8)                                 # Make the robot stand up
    posture_proxy = None  


def nao_volume_sound(volume_level):
    audio_proxy = ALProxy("ALAudioDevice", nao_ip, nao_port)
    audio_proxy.setOutputVolume(volume_level)
    audio_proxy = None


def nao_move_toward(x, y, theta, sec):
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    frequency = 1.0
    motion_proxy.moveToward(x, y, theta, [["Frequency", frequency]])
    time.sleep(sec)                                                         # Attendere qualche secondo
    motion_proxy.stopMove()                                                 # Fermare il movimento
    motion_proxy = None


def nao_move_to(x, y, theta):
    theta = math.radians(theta)
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.moveTo(x, y, theta)
    motion_proxy.stopMove()                                                 # Fermare il movimento
    motion_proxy = None


def nao_move_head(theta, head_pitch_angle):
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.setAngles("HeadYaw", theta, 0.5)                           # Cambiare l'angolo e la velocità come desiderato
    motion_proxy.setAngles("HeadPitch", head_pitch_angle, 0.5)
    motion_proxy = None


def nao_move_stop():
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.stopMove()                                                 # Fermare il movimento
    motion_proxy = None


def nao_move_left():
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.move(1.0, 0.0, 10,  [ 
                                        ["MaxStepX", 0.08],         # step of 2 cm in front
                                        #["MaxStepY", 0.16],         # default value
                                        #["MaxStepTheta", 0.4],      # default value
                                        ["MaxStepFrequency", 1.0],  # low frequency
                                        ["StepHeight", 0.01],       # step height of 1 cm
                                        ["TorsoWx", 0.0],           # default value
                                        #["TorsoWy", 0.1]            # torso bend 0.1 rad in front
                                    ])
    motion_proxy.move(1.0, 0.0, 0,  [ 
                                        ["MaxStepX", 0.08],         # step of 2 cm in front
                                        #["MaxStepY", 0.16],         # default value
                                        #["MaxStepTheta", 0.4],      # default value
                                        ["MaxStepFrequency", 1.0],  # low frequency
                                        ["StepHeight", 0.01],       # step height of 1 cm
                                        ["TorsoWx", 0.0],           # default value
                                        #["TorsoWy", 0.1]            # torso bend 0.1 rad in front
                                    ])            

def nao_move_right():
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.move(1.0, 0.0, -10,  [ 
                                        ["MaxStepX", 0.08],         # step of 2 cm in front
                                        #["MaxStepY", 0.16],         # default value
                                        #["MaxStepTheta", 0.4],      # default value
                                        ["MaxStepFrequency", 1.0],  # low frequency
                                        ["StepHeight", 0.01],       # step height of 1 cm
                                        ["TorsoWx", 0.0],           # default value
                                        #["TorsoWy", 0.1]            # torso bend 0.1 rad in front
                                    ])  
    motion_proxy.move(1.0, 0.0, 0.0,  [ 
                                        ["MaxStepX", 0.08],         # step of 2 cm in front
                                        #["MaxStepY", 0.16],         # default value
                                        #["MaxStepTheta", 0.4],      # default value
                                        ["MaxStepFrequency", 1.0],  # low frequency
                                        ["StepHeight", 0.01],       # step height of 1 cm
                                        ["TorsoWx", 0.0],           # default value
                                        #["TorsoWy", 0.1]            # torso bend 0.1 rad in front
                                    ])   
          

def nao_move_start2():
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    #motion_proxy.move(1.0, 0.0, 0.0)
    
    motion_proxy.move(1.0, 0.0, 0.0,  [ 
                                        ["MaxStepX", 0.08],         # step of 2 cm in front
                                        #["MaxStepY", 0.16],         # default value
                                        #["MaxStepTheta", 0.4],      # default value
                                        ["MaxStepFrequency", 1.0],  # low frequency
                                        ["StepHeight", 0.01],       # step height of 1 cm
                                        ["TorsoWx", 0.0],           # default value
                                        #["TorsoWy", 0.1]            # torso bend 0.1 rad in front
                                    ])   


def nao_move_start():    
    #nao_standInit()         #si prepara alla partenza
    #nao_move_head(0, 0.5)   #ruota testa verso basso

    # Imposta la velocita e l'accelerazione del movimento
    motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)
    motion_proxy.move(1.0, 0.0, 0.0)     

    # Impostazione della fotocamera inferiore
    camera_proxy = ALProxy("ALVideoDevice", nao_ip, nao_port)
    camera_proxy.setActiveCamera(1)

    # Create a proxy to ALLandMarkDetection
    try:
        landMarkProxy = ALProxy("ALLandMarkDetection", nao_ip, nao_port)
    except Exception as e:
        print("Error when creating landmark detection proxy:")
        print(str(e))
        exit(1)
    
    # Subscribe to the ALLandMarkDetection proxy. This means that the module will write in ALMemory with the given period below
    period = 100
    landMarkProxy.subscribe("Test_LandMark", period, 0.0)

    # ALMemory variable where the ALLandMarkdetection module outputs its results
    memValue = "LandmarkDetected"
    
    # Create a proxy to ALMemory
    try:
        memoryProxy = ALProxy("ALMemory", nao_ip, nao_port)
    except Exception as e:
        print("Error when creating memory proxy:")
        print(str(e))
        exit(1)
    
    print("Creating landmark detection proxy")
    
    # A simple loop that reads the memValue and checks whether landmarks are detected.                
    while True:
        time.sleep(0.1)
        val = memoryProxy.getData(memValue, 0)
    
        print("")
        print("*****")
        print("")
    
        # Check whether we got a valid output: a list with two fields.
        if(val and isinstance(val, list)):
            # We detected naomarks! For each mark, we can read its shape info and ID.
    
            # First Field = TimeStamp.
            timeStamp = val[0]
    
            # Second Field = array of Mark_Info's.
            markInfoArray = val[1]
    
            try:
                # Browse the markInfoArray to get info on each detected mark.
                for markInfo in markInfoArray:
    
                    # First Field = Shape info.
                    markShapeInfo = markInfo[0]
    
                    # Second Field = Extra info (ie, mark ID).
                    markExtraInfo = markInfo[1]
                    markID = int(markExtraInfo[0])

                    '''
                    alpha and beta represent the location of the NaoMark’s center in terms of camera angles in radian.
                    sizeX and sizeY are the mark’s size in camera angles.
                    the heading angle describes how the Naomark is oriented about the vertical axis with regards to the robot’s head.
                    '''
                    alpha   = markShapeInfo[1]
                    beta    = markShapeInfo[2]
                    sixeX   = markShapeInfo[3]
                    sizeY   = markShapeInfo[4]
                    heading = markShapeInfo[5]
    
                    # Print Mark information.
                    print("mark  ID: %d" % (markID))
                    print("  alpha %.3f - beta %.3f" % (alpha, beta))
                    print("  width %.3f - height %.3f" % (sixeX, sizeY))
                    print("  heading %.3f" % (heading))

                    motion_proxy.move(1.0, 0.0, alpha)

            except Exception as e:
                print("Naomarks detected, but it seems getData is invalid. ALValue =")
                print(val)
                print("Error msg %s" % (str(e)))
        else:
            print("Error with getData. ALValue = %s" % (str(val)))
    
    # Unsubscribe the module.
    landMarkProxy.unsubscribe("Test_LandMark")
    print("Test terminated successfully." )


# PAGINE WEB
@app.route("/", methods=['GET'])
def login():
    return render_template('hackathon.html')


@app.route('/api', methods=['GET'])
def api():
    return render_template('api.html')


@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({'code': 200, 'status': 'online', 'elapsed time': utilities.getElapsedTime(startTime)}), 200


@app.route('/webcam', methods=['GET'])
def webcam():
    return Response(nao_generate_frames(face_detection), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stand', methods=['GET'])
def stand():
    nao_stand()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200


@app.route('/standInit', methods=['GET'])
def standInit():
    nao_standInit()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200


@app.route('/start', methods=['GET'])
def start():
    nao_move_start2()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200


@app.route('/stop', methods=['GET'])
def stop():
    nao_move_stop()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200

@app.route('/left', methods=['GET'])
def left():
    nao_move_left()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200

@app.route('/right', methods=['GET'])
def right():
    nao_move_right()
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200


@app.route('/down', methods=['GET'])
def down():
    nao_move_head(0, 1)
    return jsonify({'code': 200, 'status': 'online', 'data': 'ok'}), 200

'''
CODICI JSON
200 messaggio inviato
201 messaggio ricevuto
500 errore
'''


if __name__ == "__main__":
    startTime     = time.time()

    nao  = True
    
    if nao:
        nao_volume_sound(40)
        nao_autonomous_life("disabled")
        nao_eye_white()
        nao_wakeup()
        nao_stand()
        nao_standInit()
        nao_move_head(0, 1)

        #nao_move_toward(1, 0, 0, 10)
        #nao_move_to(1, 0, 0)               #avanti
        #nao_move_to(-1, 0, 0)              #indietro
        #nao_move_to(0, 1, 0)               #sinistra
        #nao_move_to(0, -1, 0)              #destra
        #nao_move_to(0, 0, 90)              #rotazione verso sinistra
        #nao_move_to(0, 0, -90)             #rotazione verso destra
        #nao_move_head(1, 0)                #ruota testa verso sinistra
        #nao_move_head(-1, 0)               #ruota testa verso destra
        #nao_move_head(0, 1)                #ruota testa verso basso
        #nao_move_head(0, -1)               #ruota testa verso alto

    app.secret_key = os.urandom(12)
    app.run(host=config_helper.srv_host, port=config_helper.srv_port, debug=config_helper.srv_debug)