import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carica il dataset dal file CSV
data = pd.read_csv('nao_training_data.csv')

# Separazione delle feature (input) e dei target (output)
X = data[['gyro_x', 'gyro_y', 'acc_x', 'acc_y', 'acc_z']]  # Input sensoriali
y = data[['x_speed', 'y_speed', 'theta_speed']]  # Parametri di movimento

# Dividi il dataset in dati di train e test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dei dati (molto utile per reti neurali)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Costruzione del modello della rete neurale
model = Sequential()

# Aggiungi un livello denso (fully connected layer) con 32 neuroni e funzione di attivazione ReLU
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Aggiungi un altro livello denso con 64 neuroni
model.add(Dense(64, activation='relu'))

# Aggiungi un livello di output con 3 neuroni (x_speed, y_speed, theta_speed) e attivazione lineare
model.add(Dense(3, activation='linear'))

# Compila il modello, usando l'ottimizzatore Adam e la loss funzione MSE (Mean Squared Error)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Addestra il modello con i dati di train
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Valutazione del modello sui dati di test
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Utilizza il modello per predire i movimenti in base ai dati sensoriali
predictions = model.predict(X_test_scaled)

# Visualizza le prime 5 predizioni e i corrispondenti valori reali
print("Predictions: ", predictions[:5])
print("Actual values: ", y_test[:5].values)

# Salva il modello addestrato in un file HDF5
model.save('nao_movement_model.h5')
