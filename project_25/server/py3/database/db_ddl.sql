CREATE TABLE player_positions (
    player_id INT,
    timestamp DATETIME,
    x_coordinate FLOAT,
    y_coordinate FLOAT
);

CREATE TABLE utenti(
    id SERIAL PRIMARY KEY,
    username VARCHAR NOT NULL,
    password VARCHAR NOT NULL,
    nome VARCHAR,
    cognome VARCHAR
)

CREATE TABLE dati(
    id_player VARCHAR PRIMARY KEY,
    bpm INTEGER,
    passi INTEGER,
    velocità INTEGER,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
)

CREATE TABLE convocazioni(
    id_player VARCHAR PRIMARY KEY,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL,
    convoazione BOOLEAN,
    infortunio BOOLEAN,
    ammonizione BOOLEAN,
)