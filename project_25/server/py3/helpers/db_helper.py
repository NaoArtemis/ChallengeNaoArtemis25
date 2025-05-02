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
    
        #si torvano nel db_ddl.sql (data definition language)
                    
    #insert


    def insert_player(self, player_id, time_sec, x_pos, y_pos, team=None):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO player_positions 
                    (player_id, time_sec, x_pos, y_pos, team) 
                    VALUES (%s, %s, %s, %s, %s)
                    ''',
                    (player_id, time_sec, x_pos, y_pos, team)
                )

    def insert_utente(self, username, password, nome, cognome, posizione):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO utenti(username, password, nome, cognome, posizione)
                    VALUES (%s, %s, %s, %s, %s)
                    ''',
                    (username, password, nome, cognome, posizione)
                )

    def insert_dati(self, id_player, bpm, passi, velocità):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO dati(id_player, bpm, passi, velocità)
                    VALUES (%s, %s, %s, %s)
                    ''',
                    (id_player, bpm, passi, velocità)
                )

    def insert_convocazioni(self, id_player ,convocazione):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO convocazioni(convocazione)
                    VALUES (%s,%s)
                    ''',
                    (id_player, convocazione)  
                )

    def insert_disponibilità(self, id_player ,infortunio,ammonizioni):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO convocazioni(convocazione)
                    VALUES (%s,%s,%s)
                    ''',
                    (id_player, infortunio, ammonizioni)  
                )

    # select

    def select_utenti(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('SELECT * FROM utenti')
                if cur.rowcount == 0:
                    return []
                
                lista = []
                for tupla in cur.fetchall():
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
                cur.execute(
                    '''
                    SELECT * 
                    FROM utenti 
                    WHERE username = %s AND password = %s;
                    ''',
                    (username, password)
                )
                tupla = cur.fetchone()
                if tupla is None:
                    return 0
                else:
                    return {
                        'id': tupla[0],
                        'nome': tupla[3],
                        'cognome': tupla[4]
                    }
                
    def select_convocazioni(self, id_player):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''SELECT * FROM convocazioni
                        id_player = %s   
                    ''',
                    (id_player)
                )
                tupla = cur.fetchone()
                if tupla is None:
                    return 0
                else: 
                    return {
                        'infortunio' : tupla[3],
                        'ammonizione': tupla[4]
                    }

