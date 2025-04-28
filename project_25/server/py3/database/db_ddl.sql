CREATE TABLE  player_positions (
    id SERIAL PRIMARY KEY,
    player_id TEXT,
    time_sec REAL,
    x_pos REAL,
    y_pos REAL,
    team TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE utenti(
    id SERIAL PRIMARY KEY,
    username VARCHAR NOT NULL,
    password VARCHAR NOT NULL,
    nome VARCHAR,
    cognome VARCHAR
)

CREATE TABLE dati(
    id SERIAL PRIMARY KEY,
    id_player VARCHAR,
    bpm INTEGER,
    passi INTEGER,
    velocità INTEGER,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
)