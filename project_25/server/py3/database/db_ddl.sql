CREATE TABLE player_positions (
    player_id INT,
    time_sec FLOAT,
    x_pos FLOAT,
    y_pos FLOAT,
    team VARCHAR
);

CREATE TABLE utenti (
    id SERIAL PRIMARY KEY,
    username VARCHAR NOT NULL,
    password VARCHAR NOT NULL,
    nome VARCHAR,
    cognome VARCHAR
);

CREATE TABLE dati (
    id_player INT PRIMARY KEY,
    bpm INTEGER,
    passi INTEGER,
    velocità INTEGER,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE convocazioni (
    id_player INT PRIMARY KEY,
    convocazione BOOLEAN,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE disponibilita (
    id_player INT PRIMARY KEY,
    infortunio BOOLEAN,
    ammonizione BOOLEAN,
    FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
);
