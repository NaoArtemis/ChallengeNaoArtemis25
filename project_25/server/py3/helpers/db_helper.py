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

    def insert_convocazioni(self, id_player, convocazione):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO convocazioni(id_player, convocazione)
                    VALUES (%s, %s)
                    ON CONFLICT (id_player) DO UPDATE
                    SET convocazione = EXCLUDED.convocazione
                    ''',
                    (id_player, convocazione)  
                )

    def insert_disponibilita(self, id_player, infortunio, ammonizione):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO disponibilita(id_player, infortunio, ammonizione)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id_player) DO UPDATE
                    SET infortunio = EXCLUDED.infortunio,
                        ammonizione = EXCLUDED.ammonizione
                    ''',
                    (id_player, infortunio, ammonizione)  
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
                    return "0"
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

    def select_players(self): # per recuperare i giocatori dal db
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute(
                    '''
                    SELECT DISTINCT player_id 
                    FROM player_positions 
                    WHERE player_id != 0
                    '''
                )
                players = cur.fetchall()
                return [p[0] for p in players]

    def get_total_frames(self):
        """
        Conta il numero di frame univoci in player_positions.
        """
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(DISTINCT id_frame) FROM player_positions
                """)
                result = cur.fetchone()
                return result[0] if result else 0

    def select_player_positions_by_frame(self, frame_id):
        """
        Restituisce tutte le posizioni dei player in un frame (x, y, team).
        """
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute("""
                    SELECT x_pos, y_pos, team FROM player_positions
                    WHERE id_frame = %s
                """, (frame_id,))
                return cur.fetchall()

    def select_positions_by_player(self, player_id):
        """
        Restituisce tutte le posizioni (x, y) per un singolo player.
        """
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute("""
                    SELECT x_pos, y_pos FROM player_positions
                    WHERE player_id = %s
                """, (player_id,))
                return cur.fetchall()