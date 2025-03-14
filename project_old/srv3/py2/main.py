# -*- coding: utf-8 -*-

'''
python2 64 bit
pip2 install -r requirements.txt
python2 main.py
''' 

# Modules
from naoqi import ALProxy
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file
import cv2
import time
import numpy as np
import random
import paramiko
import utilities
from logging_helper import logger

app  = Flask(__name__)




#################################
# FUNZIONI FLASK SERVER Python2 #
#################################
def nao_get_image(nao_ip, nao_port):
    video_proxy = ALProxy("ALVideoDevice", nao_ip, nao_port)            # NAO webcam

    # Set the camera parameters
    name_id          = "video_image_" + str(random.randint(0,100))      # The same Name could be used only six time
    camera_id        = 0                                                # Use the top camera (1 for bottom camera)
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
        video_proxy.unsubscribe(video_client)

        
@app.route('/nao_webcam/<params>', methods=['GET'])
def nao_webcam(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json           = eval(params)
                nao_ip         = json['nao_ip']
                nao_port       = json['nao_port']

                return Response(nao_get_image(nao_ip, nao_port), mimetype='multipart/x-mixed-replace; boundary=frame')
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500 
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500


@app.route('/nao_get_audio/<params>', methods=['GET'])
def nao_get_audio(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value, "nao_user":value, "nao_password":value, "sec_sleep":value}
                json         = eval(params)
                nao_ip       = json['nao_ip']
                nao_port     = json['nao_port']
                nao_user     = json['nao_user']
                nao_password = json['nao_password']
                sec_sleep    = json['sec_sleep']

                audio_device_proxy = ALProxy("ALAudioRecorder", nao_ip, nao_port)
                remote_path = "/data/home/nao/recordings/microphones/microphone_audio.wav" # sul nao
                sample_rate = 16000
                
                # Registra l'audio dal microfono del NAO per 'sec_sleep' secondi
                audio_data = audio_device_proxy.startMicrophonesRecording(remote_path, "wav", sample_rate, [0, 0, 1, 0])
                time.sleep(sec_sleep)
                audio_device_proxy.stopMicrophonesRecording()

                # Connessione SSH al Nao
                try:
                    transport = paramiko.Transport((nao_ip, 22))                 
                    transport.connect(username=nao_user, password=nao_password)  
                    scp = paramiko.SFTPClient.from_transport(transport)          
                    local_path  = "recordings/microphone_audio.wav"
                    scp.get(remote_path, local_path)
                except Exception as e:
                    logger.error(str(e))
                finally:
                    scp.close()         
                    transport.close()   

                audio_device_proxy = None
                return send_file(local_path, as_attachment=True)
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500  
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500


@app.route('/nao_face_tracker/<params>', methods=['GET'])  
def nao_face_tracker(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                tracker_proxy = ALProxy("ALTracker", nao_ip, nao_port)
                targetName = "Face"
                faceWidth  = 0.1
                tracker_proxy.setMode("Head")
                tracker_proxy.registerTarget(targetName, faceWidth)
                tracker_proxy.track(targetName)
                return jsonify({'code': 200, 'function': 'nao_face_tracker(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500  
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500


@app.route('/nao_stop_face_tracker/<params>', methods=['GET'])  
def nao_stop_face_tracker(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                tracker_proxy = ALProxy("ALTracker", nao_ip, nao_port)
                tracker_proxy.stopTracker()
                tracker_proxy.unregisterAllTargets()
                return jsonify({'code': 200, 'function': 'nao_stop_face_tracker(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500 
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500 


@app.route('/nao_autonomous_life/<params>', methods=['GET'])  
def nao_autonomous_life(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']
                state    = "disabled"

                life_proxy = ALProxy("ALAutonomousLife", nao_ip, nao_port)              
                life_proxy.setState(state)                                              
                life_proxy = None                                                       
                return jsonify({'code': 200, 'function': 'nao_autonomous_life(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500  
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500
            

@app.route('/nao_wakeup/<params>', methods=['GET'])  
def nao_wakeup(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                motion_proxy = ALProxy("ALMotion", nao_ip, nao_port)                    
                motion_proxy.wakeUp()                                                   
                motion_proxy = None   
                return jsonify({'code': 200, 'function': 'nao_wakeup(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500   
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500                                                


@app.route('/nao_eye_white/<params>', methods=['GET'])  
def nao_eye_white(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                leds_proxy = ALProxy("ALLeds", nao_ip, nao_port)                        
                eye_group_name = "FaceLeds"                                             
                eye_color = [255, 255, 255]                                             # RGB values for white
                value_color = 256*256*eye_color[0] + 256*eye_color[1] + eye_color[2]
                leds_proxy.fadeRGB(eye_group_name, value_color, 0.5)                    # 0.5 is the duration in seconds
                leds_proxy = None
                return jsonify({'code': 200, 'function': 'nao_eye_white(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500  
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500                                                        


@app.route('/nao_animatedSayText/<params>', methods=['GET'])  
def nao_animatedSayText(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value, "text_to_say":value}
                json        = eval(params)
                nao_ip      = json['nao_ip']
                nao_port    = json['nao_port']
                text_to_say = json['text_to_say']

                tts_proxy = ALProxy("ALAnimatedSpeech", nao_ip, nao_port)                          
                animated_speech_config = {"bodyLanguageMode": "contextual"}             
                tts_proxy.say(text_to_say, animated_speech_config)                                                             
                tts_proxy = None
                return jsonify({'code': 200, 'function': 'nao_animatedSayText(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500   
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500  


@app.route('/nao_standInit/<params>', methods=['GET'])  
def nao_standInit(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                posture_proxy = ALProxy("ALRobotPosture", nao_ip, nao_port)                        
                posture_proxy.goToPosture("StandInit", 0.8)                             
                posture_proxy = None
                return jsonify({'code': 200, 'function': 'nao_standInit(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500                                                 


@app.route('/nao_stand/<params>', methods=['GET'])            
def nao_stand(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value}
                json     = eval(params)
                nao_ip   = json['nao_ip']
                nao_port = json['nao_port']

                posture_proxy = ALProxy("ALRobotPosture", nao_ip, nao_port)                        
                posture_proxy.goToPosture("Stand", 0.8)                                 
                posture_proxy = None  
                return jsonify({'code': 200, 'function': 'nao_stand(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500


@app.route('/nao_volume_sound/<params>', methods=['GET'])    
def nao_volume_sound(params):
    if (params != None and params != ''):
        if request.method == 'GET':
            try:
                #{"nao_ip":value, "nao_port":value, "volume_level":value}
                json         = eval(params)
                nao_ip       = json['nao_ip']
                nao_port     = json['nao_port']
                volume_level = json['volume_level']

                audio_proxy = ALProxy("ALAudioDevice", nao_ip, nao_port)
                audio_proxy.setOutputVolume(volume_level)
                audio_proxy = None
                return jsonify({'code': 200, 'function': 'nao_volume_sound(ip:' + str(nao_ip) + ' port:' + str(nao_port) + ')', 'status':'OK'}), 200
            except Exception as e:
                logger.error(str(e))
                return jsonify({'code': 500, 'message': str(e)}), 500
        else:
            return jsonify({'code': 500, 'message': 'methods error'}), 500
    else:
        return jsonify({'code': 500, 'message': 'params error'}), 500




# API
@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({'code': 200, 'status': 'online', 'elapsed time': utilities.getElapsedTime(startTime)}), 200


# INDEX
@app.route('/', methods=['GET'])
def services():
    return render_template('index.html')




'''
CODICI JSON
200 messaggio inviato
201 messaggio ricevuto
500 errore
'''


if __name__ == "__main__":
    startTime = time.time()
    app.run(host="0.0.0.0", port=5011, debug=False)