from naoqi import ALProxy
import time
import csv

# Imposta l'IP del robot Nao
robot_ip = "192.168.0.138"
port     = 9559

# Crea proxy per la memoria e il controllo del movimento
memory_proxy = ALProxy("ALMemory", robot_ip, port)
motion_proxy = ALProxy("ALMotion", robot_ip, port)

# Funzione per ottenere i dati dai sensori (giroscopio e accelerometro)
def get_sensor_data():
    gyro_x = memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value")
    gyro_y = memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value")
    acc_x  = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
    acc_y  = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
    acc_z  = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
    return [gyro_x, gyro_y, acc_x, acc_y, acc_z]

# Avvia la camminata di Nao
def walk_nao(x_speed, y_speed, theta_speed):
    motion_proxy.move(x_speed, y_speed, theta_speed)

# Inizia la raccolta dei dati
def collect_data(duration, filename):
    start_time = time.time()
    data = []

    while time.time() - start_time < duration:
        # Raccogli i dati sensoriali
        sensors = get_sensor_data()

        # Ottieni le correzioni manuali applicate durante la raccolta dati
        # Puoi inserire queste correzioni manualmente o usare un sistema per applicarle dinamicamente
        '''
        x_speed = float(input("Inserisci x_speed (ad esempio 0.5): "))
        y_speed = float(input("Inserisci y_speed (ad esempio 0.0): "))
        theta_speed = float(input("Inserisci theta_speed (ad esempio 0.0): "))
        '''
        x_speed = 1.0
        y_speed = 0.0
        theta_speed = 0.0

        # Salva i dati sensoriali e le correzioni nel dataset
        data.append(sensors + [x_speed, y_speed, theta_speed])

        # Applica le correzioni al movimento di Nao
        walk_nao(x_speed, y_speed, theta_speed)

        # Attendi prima della prossima iterazione
        time.sleep(0.1)

    motion_proxy.stopMove()  


    # Salva i dati in un file CSV
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['gyro_x', 'gyro_y', 'acc_x', 'acc_y', 'acc_z', 'x_speed', 'y_speed', 'theta_speed'])
        writer.writerows(data)




# Raccogli i dati per un certo periodo di tempo
secondi = 2     # Raccoglie dati per x secondi
collect_data(secondi, 'nao_training_data.csv')  
