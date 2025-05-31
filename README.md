
# NAO coach & care
## Contents
* [NAO Challenge 2025](#nao-challenge-2025)
    * [Project](#project)
        * [Coding](#coding)
            * [App](#app)
            * [Server](#server)
            * [Database](#database)
            * [Sequence Diagram](#sequence-diagram)
        * [Social](#social)
            * [Logos](#logos)
            * [Merch](#merch)
            * [Website](#website)
* [Authors](#authors)

## NAO Challenge 2025

Every year, the theme of the NaoChallenge changes, but its goal remains the same: using robotics to tackle real-world challenges. This year, the NaoArtemis team has focused on sport, aiming to support both gameplay and the fan experience.

### Project

For the NAO Challenge 2025, the NaoArtemis team has developed a project that brings innovation and inclusion into the world of sport. The system is divided into two core components. On the field, NAO acts as an assistant coach, supported by a web app that provides tactical insights such as player tracking, Voronoi analysis, and performance monitoring using computer vision and biometric data. In the stands, NAO promotes accessibility and participation for fans with autism or communication challenges by using ArUco symbols and Augmentative and Alternative Communication (AAC) to interact through personalized voice messages. The project aims to merge robotics, sport, and social values to create a more inclusive and technologically enhanced game experience.

- [x] Computer Vision
- [x] AI-powered Coaching
- [x] Fan Engagement 
- [X] Accessibility
- [X] Integration

##### REQUIREMENTS:
> [!IMPORTANT]
> - opencv-python==4.2.0.32 <br>
> - opencv-python-headless==4.2.0.32 <br>
> - SpeechRecognition==3.8.1 <br>
> flask <br>
> flask_login <br>
> numpy <br>
> requests <br>
> PyYAML <br>
> yieldfrom <br>
> selenium <br>
> paramiko <br>
> psycopg2 <br>
> scikit-learn <br>
> matplotlib <br>
> scipy <br>
> supervision <br>
> ultralytics <br>
>openai <br>
> whisper <br>

## Coding

### Database:

The folder [database](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/coding/database) contains the complete database with defined tables and attributes.

- This SQL script creates the `dati` table, which stores **biometric and movement data** collected during matches. Each entry is linked to a player from the `utenti` table. These data are then used for **post-match analysis**, allowing the team to evaluate player performance.

```sql
    CREATE TABLE dati (
    id SERIAL PRIMARY KEY,
    id_player INT ,
    bpm INTEGER,
    passi INTEGER,
    velocità INTEGER,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE CASCADE
);
```

### App:

The folder [App/NaoArtemis](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/coding/app) contains the source code of the app named NC&C and Joystick.  

The app **NC&C** allows players and coaches to **interface with the system** and view **biometric and performance data** collected during the match. Users can log in to access personalized statistics such as heart rate, steps, speed, and match participation.

A simplified control **Joystick** interface that allows us to **communicate directly with the NAO robot**. It includes directional controls and basic command buttons, enabling us to manually move the Nao.

### Server:

The repository in the [Server](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/coding/server) directory serves as the core of the NaoArtemis 2025 project. The folder contains two separate servers:
- A main server which runs on **Python 3 server**.
    - The core script orchestrates the analysis of match recordings by loading pretrained AI models such as YOLO for player detection and KMeans for role clustering. The video is processed frame by frame, with annotated results stored, and a Voronoi diagram is generated to represent the spatial dynamics and positioning of the entire squad. The code initializes key components, including annotators, model paths, and tracking parameters, while also exposing HTTP endpoints via Flask to trigger the analysis externally. Biometric data—such as heart rate, step count, and speed—can be integrated into the analytical output, enhancing performance evaluation. In addition to match analysis, the server manages requests related to audience interaction: it decodes ArUco symbols shown by users in the stands and triggers personalized vocal responses, enabling NAO to communicate through AAC (Augmentative and Alternative Communication). This functionality is specifically designed to support the participation of individuals with autism or communication difficulties. The structure also includes a PostgreSQL-integrated database helper, logging utilities, and configuration loaders, allowing for scalable and modular expansion. In summary, this repository excels not only in database operations, meticulous log management, and efficient HTTP communication, but also leverages Flask for powerful web development capabilities and inclusive interaction strategies.



    ```python
    from logging_helper import logger
    from datetime import datetime
    from decimal import Decimal
    class DB:
        def __init__(self):
            import config_helper
            config_helper = config_helper.Config()

            try:
                self.connection = psycopg2.connect(host=config_helper.db_host, 
                                                database=config_helper.db_name,
                                                user=config_helper.db_user, 
                                                password=config_helper.db_password)
    ```

    - This block of code is part of the main computer vision pipeline and is executed for each frame of the match video. The system first uses a YOLOv8 model (`MODEL_PLAYERS`) to detect players in the frame. The system calculates the average color of the player's region and applies a KMeans clustering algorithm (with two clusters) to group players by team, assigning them labels such as "blue" or "red". Once the players are tracked and classified, the frame is annotated with ellipses and custom labels (including tracker ID and team) using Supervision annotators. The result is a structured list of player coordinates and team labels (`frame_data`), which is used for further tactical analysis such as Voronoi diagram generation. The function then returns the annotated frame to be displayed or saved for later use. This logic connects object detection, tracking, clustering, and spatial transformation into a single, modular pipeline.

    ```python
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
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2       
                center = np.array([center_x, center_y, 1])
                if homography_matrix is not None:
                    pt = homography_matrix @ center
                    pt /= pt[2]
                else:
                    pt = center
                frame_data.append((float(pt[0]), float(pt[1]), teams[i]))
            frame = annotated
    ```

    -  In Task 2, the robot interacts with the audience using **CAA (Augmentative and Alternative Communication)** cards that include an **ArUco marker** in the upper part. When a spectator shows the card to the camera, the system detects the marker ID and maps it to a corresponding function. This function triggers a vocal or physical response from NAO, such as announcing the score or celebrating a goal. The logic below shows how a specific action is executed when a known marker is recognized:

    ```python
        import cv2
        import cv2.aruco as aruco

        # Initialize ArUco dictionary and detector parameters
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)

        # Detect ArUco markers in the frame
        marker_corners, marker_ids, _ = detector.detectMarkers(frame)

        # Map detected marker IDs to specific NAO actions
        if marker_ids is not None:
            ids = marker_ids.flatten()

            if 184 in ids and not task_2:
                task_2 = True
                nao_points()         # Announces the current score

            elif 185 in ids and not task_2:
                task_2 = True
                nao_time_match()     # Provides match statistics

            elif 186 in ids and not task_2:
                task_2 = True
                nao_seat()           # informs about available seating

            elif 187 in ids and not task_2:
                task_2 = True
                nao_cori()           # Celebrates and cheers with fan
    ```

- The other server (in **Python 2**) interfaces directly with the **NAO robot**.

    - The secondary server, written in Python 2, is dedicated to controlling the NAO robot using Aldebaran’s proprietary libraries, which are only compatible with this version of Python. It handles all the low-level functionalities related to speech, movement, and interaction. The server receives HTTP requests from the Python 3 backend and executes vocal messages or predefined behaviors on the robot. It includes modules for speech synthesis, head and body motion control, and audio playback. The server ensures stable and responsive communication with the robot without interfering with the main AI and data processing tasks.

    - This code is a function to retrieve **live video from the NAO robot’s top camera** using the `ALVideoDevice` module from Aldebaran’s API. This stream is used, for example, in ArUco symbol detection (Task 2) or in general visual interaction.

    ```python
        def nao_get_image(nao_ip, nao_port):
        video_proxy = ALProxy("ALVideoDevice", nao_ip, nao_port)

        name_id    = "video_image_" + str(random.randint(0, 100))  # unique client name
        camera_id  = 0                                              # top camera
        resolution = 1                                              # 320x240
        color_space = 13                                            # RGB
        camera_fps  = 10
        video_proxy.setParameter(camera_id, 0, 55)                  # brightness

        video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, color_space, camera_fps)

        try:
            while True:
                image = video_proxy.getImageRemote(video_client)
                image_data = np.frombuffer(image[6], dtype=np.uint8).reshape((image[1], image[0], 3))

                resized = cv2.resize(image_data, (640, 480))
                ret, buffer = cv2.imencode('.jpg', resized)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(str(e))
        finally:
            video_proxy.unsubscribe(video_client)
    ```

### Sequence Diagram:

This folder [sequence_diagram](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/coding/sequence_diagram) contains the sequence diagrams representing the key interactions and workflows of the NaoArtemis 2025 project.

The system is structured around two main tasks, each represented by a dedicated diagram:

- **Task 1 – Assistant Coach Functionality**:  
  In this flow, the **coach** uses the web application (WEB-APP) to request tactical support during a match. The request is sent to the AI server (SERVER 3), which processes the current game situation using data such as biometric indicators (heart rate, step count, speed). The server analyzes the distribution of players on the field and their physical condition, then provides suggestions to the coach regarding possible **substitutions** or **tactical adjustments**. These may include replacing tired players, switching roles, or reorganizing the team's spatial balance. The web app displays this feedback in real time, supporting the coach in making informed and strategic decisions during the game.

- **Task 2 – Inclusive Fan Interaction**:  
  This flow involves an **audience member** or guest interacting with NAO in the stands. The user shows an ArUco symbol, which is detected by the robot's camera and processed by the Python 2 server (SERVER 2). The system identifies the symbol and triggers a vocal response using AAC (Augmentative and Alternative Communication), allowing NAO to provide messages of encouragement, game commentary, or information. This task is designed to enhance accessibility, especially for individuals with autism or communication difficulties.  
  <div align="center">
    <img src="https://github.com/NaoArtemis/ChallengeNaoArtemis25/blob/main/coding/sequence_diagram/task_1.svg" width="700" height="400" />
    <br>
    <img src="https://github.com/NaoArtemis/ChallengeNaoArtemis25/blob/main/coding/sequence_diagram/task_2.svg"  width="350" height="300" >
 </div>

These diagrams illustrate how NaoArtemis manages both game-related analysis and inclusive interaction, coordinating data and actions across multiple components: the coach, guest, web app, two servers, database, and the robot.

## Social

### logos 

This folder [logos](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/social/loghi) contains the logos of the project

<div align="center">
<img src="https://github.com/NaoArtemis/ChallengeNaoArtemis25/blob/main/social/loghi/logo_v1.png" width="600" height="350"/>
</div>

### Merch

The folder [merch](https://github.com/NaoArtemis/ChallengeNaoArtemis25/tree/main/social/tshirt_naoartemis) contains the images specifically curated for the production of the team's merchandise line, ensuring high-quality designs and branding consistency across all products.

### Website

## Authors

Suggest us new ideas at:

* social@gmail.com (NAOARTEMIS)


## Social

* [YouTube](https://www.youtube.com/@NaoArtemis)
* [Instagram](https://www.instagram.com/naoartemis/)
* [TikTok](https://www.tiktok.com/@naoartemis)
* [LinkedIn]()

## License

[GNU](https://www.gnu.org/licenses/gpl-3.0.html)