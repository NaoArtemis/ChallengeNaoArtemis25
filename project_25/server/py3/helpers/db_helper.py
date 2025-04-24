import psycopg2

from datetime import datetime
from decimal import Decimal

import helpers.config_helper as config_helper
from helpers.logging_helper import logger

class DB:
    def __init__(self, config_helper: config_helper.Config):
        try:
            self.connection = psycopg2.connect(host=config_helper.db_host, 
                                               database=config_helper.db_name,
                                               user=config_helper.db_user, 
                                               password=config_helper.db_password)
        except Exception as e:
            logger.error(str(e))

#tables
    def create_tables(self):
        with self.connection:
            with self.connection.cursor() as cur:
                # Tabella player_positions
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS player_positions (
                        id SERIAL PRIMARY KEY,
                        player_id TEXT,
                        time_sec REAL,
                        x_pos REAL,
                        y_pos REAL,
                        team TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabella utenti
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS utenti (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR NOT NULL,
                        password VARCHAR NOT NULL,
                        nome VARCHAR,
                        cognome VARCHAR
                    )
                ''')

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS dati (
                        id SERIAL PRIMARY KEY,
                        id_player VARCHAR,
                        bpm INTEGER,
                        passi INTEGER,
                        velocità INTEGER,
                        FOREIGN KEY(id_player) REFERENCES utenti(id) ON UPDATE CASCADE ON DELETE SET NULL
                    )
                ''')


#insert

    def insert_player(self, player_id, time_sec, x_pos, y_pos, team=None):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO player_positions 
                    (player_id, time_sec, x_pos, y_pos, team) 
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (player_id, time_sec, x_pos, y_pos, team)
                )

    def insert_utente(self, username, password, nome, cognome):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO utenti(username, password, nome, cognome)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    ''',
                    (username, password, nome, cognome)
                )
                nuovo_id = cur.fetchone()[0]
                return nuovo_id


#select

    def select_utenti(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('SELECT * FROM utenti')
                if cur.rowcount == 0:
                    return []
                
                lista = []
                for tupla in cur:
                    lista.append({
                        'id': tupla[0], 
                        'username': tupla[1], 
                        'password': tupla[2], 
                        'nome': tupla[3], 
                        'cognome': tupla[4]
                    })
                return lista
            
    def select_account_player(self, username, password):
            with self.connection:
                with self.connection.cursor() as cur:
                    cur.execute('''
                                SELECT * 
                                FROM utenti 
                                WHERE username::text = %s AND password::text = %s;
                                ''', (str(username),str(password)))

                    if (cur.rowcount == 0):
                        return 0
                    else:
                        for tupla in cur:
                            return  {'id': tupla[0], 'nome': tupla[3], 'cognome': tupla[4]} #in [2]username [3]password    