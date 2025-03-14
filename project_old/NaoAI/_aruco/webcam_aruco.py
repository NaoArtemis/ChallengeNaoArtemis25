# python3

import cv2
import cv2.aruco as aruco

# Inizializza il dispositivo di acquisizione video (la webcam)
cap = cv2.VideoCapture(1)  # 0 indica il dispositivo di default (la prima webcam disponibile)

# Verifica se il dispositivo di acquisizione e' stato aperto correttamente
if not cap.isOpened():
    print("Impossibile aprire il dispositivo di acquisizione video.")
    exit()

# Ottieni le dimensioni del frame dalla webcam
width  = int(cap.get(3))
height = int(cap.get(4))

# Inizializza il dizionario ArUco
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# Inizializza il rilevatore ArUco
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    # Verifica se la lettura del frame e' avvenuta con successo
    if not ret:
        print("Errore durante la lettura del frame.")
        break

    # Specchia il frame orizzontalmente
    #frame = cv2.flip(frame, 1)

    # Conversione del frame in scala di grigi per migliorare il rilevamento dei marker
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i marker ArUco nel frame usando ArucoDetector
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

    # Se vengono rilevati marker, disegnali sul frame
    if marker_ids is not None:
        # Disegna i marker rilevati sul frame
        aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    # Mostra il frame in una finestra
    cv2.imshow('Webcam', frame)

    # Attendi il tasto 'q' per interrompere il loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia il dispositivo di acquisizione e chiudi la finestra
cap.release()
cv2.destroyAllWindows()
