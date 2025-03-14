# python3

import cv2
import cv2.aruco as aruco

# Inizializza il dispositivo di acquisizione video (la webcam)
cap = cv2.VideoCapture(1)  # 0 indica il dispositivo di default (la prima webcam disponibile)

# Verifica se la webcam è stata aperta correttamente
if not cap.isOpened():
    print("Impossibile aprire il dispositivo di acquisizione video.")
    exit()

# Ottieni le dimensioni del frame dalla webcam
width    = int(cap.get(3))
height   = int(cap.get(4))
center_x = width  // 2
center_y = height // 2

# Inizializza il dizionario ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# Inizializza il rilevatore ArUco
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Dimensione reale del marker in metri (ad esempio: 0.05 per 5 cm)
marker_size = 0.02

# Focale della camera in pixel (esempio, deve essere calibrata per la tua camera specifica)
focal_length = 800

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    # Verifica se la lettura del frame è avvenuta con successo
    if not ret:
        print("Errore durante la lettura del frame.")
        break

    # Conversione del frame in scala di grigi per migliorare il rilevamento dei marker
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i marker ArUco nel frame
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

    # Se vengono rilevati marker, determina la posizione e la distanza
    if marker_ids is not None:
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
            if abs(deviation_x) < 10:  # Tolleranza per essere considerato "dritto"
                direction = "Dritto"
            elif deviation_x > 0:
                direction = "Storto a destra"
            else:
                direction = "Storto a sinistra"

            print(f"Distanza dal marker: {distance:.2f} metri, Posizione marker: ({marker_center_x}, {marker_center_y}), Direzione: {direction}")

            # Disegna i marker rilevati sul frame
            aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

            # Disegna una linea tra il centro del frame e il centro del marker
            cv2.line(frame, (center_x, center_y), (marker_center_x, marker_center_y), (0, 255, 0), 2)

    # Disegna il centro del frame
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # Mostra il frame in una finestra
    cv2.imshow('Webcam', frame)

    # Attendi il tasto 'q' per interrompere il loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia il dispositivo di acquisizione e chiudi la finestra
cap.release()
cv2.destroyAllWindows()
