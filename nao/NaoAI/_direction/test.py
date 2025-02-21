from naoqi import ALProxy
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Carica il modello addestrato
model = tf.keras.models.load_model('nao_movement_model.h5')

# Crea proxy per accedere ai sensori e controllare il movimento
robot_ip = "xxx.xxx.xxx.xxx"  # Sostituisci con l'IP del robot Nao
port = 9559

memory_proxy = ALProxy("ALMemory", robot_ip, port)
motion_proxy = ALProxy("ALMotion", robot_ip, port)

# Standardizzatore che dovrai usare (quello con cui hai scalato i dati in fase di training)
scaler = StandardScaler()  # Sostituiscilo con lo stesso scaler che hai usato in fase di training

# Funzione per ottenere i dati sensoriali in tempo reale
def get_sensor_data():
    gyro_x = memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value")
    gyro_y = memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value")
    acc_x = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
    acc_y = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
    acc_z = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
    return np.array([gyro_x, gyro_y, acc_x, acc_y, acc_z]).reshape(1, -1)  # Reshape per adattare al modello

# Funzione per camminare in base alla predizione
def walk_nao(x_speed, y_speed, theta_speed):
    motion_proxy.moveToward(x_speed, y_speed, theta_speed)

# Loop principale per raccogliere dati in tempo reale e predire i movimenti
while True:
    # Raccogli i dati sensoriali
    sensor_data = get_sensor_data()
    
    # Scala i dati con lo scaler usato nel training
    sensor_data_scaled = scaler.transform(sensor_data)
    
    # Usa il modello per predire i movimenti
    predicted_speeds = model.predict(sensor_data_scaled)

    # Estrai le velocità predette
    x_speed, y_speed, theta_speed = predicted_speeds[0]

    # Applica i movimenti predetti al robot
    walk_nao(x_speed, y_speed, theta_speed)

    # Attendi prima della prossima iterazione
    time.sleep(0.1)
