from flask import Flask, request, jsonify
import json
import os


# utilizzata per ricevere dati da gps e salvarli nel file .json

app = Flask(__name__)
DATA_FILE = "gps_data.json"

# caricare dati
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    return []

# salvare dati 
def save_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

@app.route('/send_gps', methods=['POST'])
def receive_gps():
    data = request.json
    gps_data = load_data()
    gps_data.append(data)
    save_data(gps_data)
    return jsonify({"message": "Dati GPS salvati con successo!"}), 200

@app.route('/get_gps', methods=['GET'])
def get_gps():
    return jsonify(load_data())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)