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

    def get_cliente(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Cliente
                            ORDER BY id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({
                                'id'      : tupla[0], 
                                'username': tupla[1], 
                                'nome'    : tupla[2], 
                                'cognome' : tupla[3], 
                                'password': tupla[4]
                            })
                return data

    def get_id_cliente(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Cliente
                            WHERE id::text = %s;
                            ''', (str(id),))

                if (cur.rowcount == 0):
                    return {}

                for tupla in cur:
                    return  {
                                'id'      : tupla[0], 
                                'username': tupla[1], 
                                'nome'    : tupla[2], 
                                'cognome' : tupla[3], 
                                'password': tupla[4]
                            }
                
    def get_account_cliente(self, username, password):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Cliente
                            WHERE username::text = %s AND password::text = %s;
                            ''', (str(username),str(password)))

                if (cur.rowcount == 0):
                    return 0
                else:
                    for tupla in cur:
                        return  {'id_cliente': tupla[0], 'nome': tupla[2], 'cognome': tupla[3]}

    def set_cliente(self, username, nome, cognome, password):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Cliente(username,nome,cognome,password)
                            VALUES (%s, %s, %s, %s)
                            RETURNING id;
                            ''', (username, nome, cognome, password))
                nuovo_id = cur.fetchone()[0]
                return cur.statusmessage, nuovo_id

    def set_id_cliente(self, id, username, nome, cognome, password):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Cliente
                            SET username = %s,
                                nome = %s,
                                cognome = %s,
                                password = %s
                            WHERE id::text = %s;
                            ''', (username, nome, cognome, password,str(id)))
                return cur.statusmessage

    def get_count_cliente(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT COUNT(*)
                            FROM Cliente;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                for tupla in cur:
                    return {'count': tupla[0]}

    def delete_cliente(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            DELETE
                            FROM Cliente
                            WHERE id::text = %s;
                            ''', (str(id),))
                return id
            
    def set_carrello(self, id_cliente):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Carrello(id_cliente)
                            VALUES (%s);
                            ''', (str(id_cliente),))
                return cur.statusmessage
            
    def delete_carrello(self, id_cliente):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            DELETE
                            FROM Carrello
                            WHERE id_cliente::text = %s;
                            ''', (str(id_cliente),))
                return id_cliente

    def get_oggetto(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Oggetto
                            ORDER BY id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({
                                'id'                : tupla[0], 
                                'titolo'            : tupla[1], 
                                'categoria'         : tupla[2], 
                                'prezzo'            : float(tupla[3]), 
                                'descrizione'       : tupla[4],
                                'foto'              : tupla[5], 
                                'qta_magazzino'     : tupla[6], 
                                'qta_scaffale'      : tupla[7], 
                                'sconto'            : tupla[8], 
                                'eta_consigliata'   : tupla[9], 
                                'sesso_consigliato' : tupla[10] 
                            })
                return data

    def get_id_oggetto(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Oggetto
                            WHERE id::text = %s;
                            ''', (str(id),))

                if (cur.rowcount == 0):
                    return {}
                for tupla in cur:
                    return {
                                'id'                : tupla[0], 
                                'titolo'            : tupla[1], 
                                'categoria'         : tupla[2], 
                                'prezzo'            : float(tupla[3]), 
                                'descrizione'       : tupla[4],
                                'foto'              : tupla[5], 
                                'qta_magazzino'     : tupla[6], 
                                'qta_scaffale'      : tupla[7], 
                                'sconto'            : tupla[8], 
                                'eta_consigliata'   : tupla[9], 
                                'sesso_consigliato' : tupla[10] 
                            }
                
    def get_titolo_oggetto(self, titolo):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Oggetto
                            WHERE titolo::text = %s;
                            ''', (str(titolo),))

                if (cur.rowcount == 0):
                    return {}
                for tupla in cur:
                    return {
                                'id'                : tupla[0], 
                                'titolo'            : tupla[1], 
                                'categoria'         : tupla[2], 
                                'prezzo'            : float(tupla[3]), 
                                'descrizione'       : tupla[4],
                                'foto'              : tupla[5], 
                                'qta_magazzino'     : tupla[6], 
                                'qta_scaffale'      : tupla[7], 
                                'sconto'            : tupla[8], 
                                'eta_consigliata'   : tupla[9], 
                                'sesso_consigliato' : tupla[10] 
                            }

    def set_oggetto(self, titolo, categoria, prezzo, descrizione, foto, qta_magazzino, qta_scaffale, sconto, eta_consigliata, sesso_consigliato):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Oggetto(titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                            ''', (titolo,categoria,prezzo,descrizione,foto,qta_magazzino,qta_scaffale,sconto,eta_consigliata,sesso_consigliato))
                return cur.statusmessage

    def set_id_oggetto(self, id, titolo, categoria, prezzo, descrizione, foto, qta_magazzino, qta_scaffale, sconto, eta_consigliata, sesso_consigliato):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Oggetto
                            SET titolo = %s,
                                categoria = %s,
                                prezzo = %s,
                                descrizione = %s,
                                foto = %s,
                                qta_magazzino = %s,
                                qta_scaffale = %s,
                                sconto = %s,
                                eta_consigliata = %s,
                                sesso_consigliato = %s
                            WHERE id::text = %s;
                            ''', (titolo, categoria, prezzo, descrizione, foto, qta_magazzino, qta_scaffale, sconto, eta_consigliata, sesso_consigliato, str(id)))
                return cur.statusmessage

    def delete_oggetto(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            DELETE
                            FROM Oggetto
                            WHERE id::text = %s;
                            ''', (str(id),))
                return id
            
    def get_ordine_oggetto(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM OrdineOggetto;
                            ''')
                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({'id'           : tupla[0],
                                 'id_ordine'    : tupla[1],
                                 'id_oggetto'   : tupla[2]
                    })
                return data
            
    def get_ordine_oggetto_cliente(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM OrdineOggetto AS OrdOgg INNER JOIN Oggetto AS Ogg ON OrdOgg.id_oggetto = Ogg.id
                                                            INNER JOIN Ordine AS Ord ON OrdOgg.id_ordine = Ord.id
                                                                INNER JOIN Cliente AS Cli ON Ord.id_cliente = Cli.id
                            ''')

                if (cur.rowcount == 0):
                    return {}
                """
                for tupla in cur.description:
                    print tupla[0]
                """
                data = []
                for tupla in cur:
                    data.append({
                                'id': tupla[0],
                                'oggetto': {
                                    'id'                : tupla[3], 
                                    'titolo'            : tupla[4], 
                                    'categoria'         : tupla[5], 
                                    'prezzo'            : tupla[6], 
                                    'descrizione'       : tupla[7],
                                    'foto'              : tupla[8], 
                                    'qta_magazzino'     : tupla[9], 
                                    'qta_scaffale'      : tupla[10], 
                                    'sconto'            : tupla[11], 
                                    'eta_consigliata'   : tupla[12], 
                                    'sesso_consigliato' : tupla[13] 
                                },
                                'ordine': {
                                    'id'                : tupla[15],
                                    'prezzo_totale'     : tupla[17],
                                    'data_acquisto'     : tupla[18],
                                    'ora_acquisto'      : tupla[19],
                                    'modalita_pagamento': tupla[20]
                                },
                                'cliente': {
                                    'id'                : tupla[21],
                                    'username'          : tupla[22],
                                    'nome'              : tupla[23],
                                    'cognome'           : tupla[24],
                                    'password'          : tupla[25]
                                }
                            })
                return data
            
    def get_ordine(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Ordine;
                            ''')
                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({'id'                   : tupla[0],
                                 'id_cliente'           : tupla[1],
                                 'prezzo_totale'        : tupla[2],
                                 'data_acquisto'        : tupla[3],
                                 'ora_acquisto'         : tupla[4],
                                 'modalita_pagamento'   : tupla[5]
                    })
                return data
            
    def get_id_ordine(self, id_cliente):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Ordine
                            WHERE id_cliente::text = %s;
                            ''', (str(id_cliente),))
                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({'id'                   : tupla[0],
                                 'id_cliente'           : tupla[1],
                                 'prezzo_totale'        : float(tupla[2]),
                                 'data_acquisto'        : str(tupla[3]),
                                 'ora_acquisto'         : str(tupla[4]),
                                 'modalita_pagamento'   : tupla[5]
                    })
                return data

    def get_carrello_oggetto(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM CarrelloOggetto AS CarOgg INNER JOIN Oggetto AS Ogg ON CarOgg.id_oggetto = Ogg.id
                                                            INNER JOIN Carrello AS Car ON CarOgg.id_carrello = Car.id
                                                                INNER JOIN Cliente AS Cli ON Car.id_cliente = Cli.id
                            ''')

                if (cur.rowcount == 0):
                    return {}
                '''
                for tupla in cur.description:
                    print tupla[0]
                '''
                data = []
                for tupla in cur:
                    data.append({
                                'id': tupla[0],
                                'oggetto': {
                                    'id'                : tupla[3], 
                                    'titolo'            : tupla[4], 
                                    'categoria'         : tupla[5], 
                                    'prezzo'            : tupla[6], 
                                    'descrizione'       : tupla[7],
                                    'foto'              : tupla[8], 
                                    'qta_magazzino'     : tupla[9], 
                                    'qta_scaffale'      : tupla[10], 
                                    'sconto'            : tupla[11], 
                                    'eta_consigliata'   : tupla[12], 
                                    'sesso_consigliato' : tupla[13] 
                                },
                                'carrello': {
                                    'id'                : tupla[1],
                                },
                                'cliente': {
                                    'id'                : tupla[16],
                                    'username'          : tupla[17],
                                    'nome'              : tupla[18],
                                    'cognome'           : tupla[19],
                                    'password'          : tupla[20]
                                }
                            })
                return data

    def get_carrello(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Carrello
                            ORDER BY id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({
                                'id'        : tupla[0], 
                                'id_cliente': tupla[1]
                            })
                return data
            
    def get_id_carrello(self, id_cliente):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Carrello
                            WHERE id_cliente::text = %s;
                            ''', (str(id_cliente),))

                if (cur.rowcount == 0):
                    return {}
                for tupla in cur:
                    return {'id': tupla[0]}

    def set_qta_scaffale_plus(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Oggetto
                            SET qta_scaffale = qta_scaffale + 1
                            WHERE id::text = %s;
                            ''', (str(id),))

                return cur.statusmessage

    def set_qta_scaffale_minus(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Oggetto
                            SET qta_scaffale = qta_scaffale - 1
                            WHERE id::text = %s;
                            ''', (str(id),))

                return cur.statusmessage


    def set_qta_magazzino_plus(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Oggetto
                            SET qta_magazzino = qta_magazzino + 1
                            WHERE id::text = %s;
                            ''', (str(id),))

                return cur.statusmessage

    def set_qta_magazzino_minus(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            UPDATE Oggetto
                            SET qta_magazzino = qta_magazzino - 1
                            WHERE id::text = %s;
                            ''', (str(id),))

                return cur.statusmessage

    def get_abbinamento(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Abbinamento
                            ORDER BY id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({
                                'id'         : tupla[0], 
                                'id_oggetto1': tupla[1],
                                'id_oggetto2': tupla[2]
                            })
                return data

    def set_abbinamento(self, id1, id2):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Abbinamento(id_oggetto1,id_oggetto2)
                            VALUES (%s, %s);
                            ''', (id1, id2))
                return cur.statusmessage

    def delete_abbinamento(self, id):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            DELETE
                            FROM Abbinamento
                            WHERE id::text = %s;
                            ''', (str(id),))
                return id
            
    def set_carrellooggetto(self, id_carrello, id_oggetto):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Carrellooggetto(id_carrello,id_oggetto)
                            VALUES (%s, %s)
                            RETURNING id;
                            ''', (id_carrello, id_oggetto))
                nuovo_id = cur.fetchone()[0]
                return cur.statusmessage, nuovo_id
            
    def delete_carrellooggetto(self, id_carrello, id_oggetto):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            DELETE
                            FROM Carrellooggetto
                            WHERE id_carrello = %s AND id_oggetto = %s;
                            ''', (id_carrello,id_oggetto))
                return cur.statusmessage
            
    def get_id_carrello_oggetto(self, id_carrello):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM CarrelloOggetto AS CarOgg INNER JOIN Oggetto AS Ogg ON CarOgg.id_oggetto = Ogg.id
                            WHERE CarOgg.id_carrello = %s;
                            ''', (id_carrello,))
                
                if (cur.rowcount == 0):
                    return {}
                '''
                for tupla in cur.description:
                    print tupla[0]
                '''
                data = []
                for tupla in cur:
                    data.append({
                                'id': tupla[0],
                                'oggetto': {
                                    'id'                : tupla[3], 
                                    'titolo'            : tupla[4], 
                                    'categoria'         : tupla[5], 
                                    'prezzo'            : float(tupla[6]), 
                                    'descrizione'       : tupla[7],
                                    'foto'              : tupla[8], 
                                    'qta_magazzino'     : tupla[9], 
                                    'qta_scaffale'      : tupla[10], 
                                    'sconto'            : tupla[11], 
                                    'eta_consigliata'   : tupla[12], 
                                    'sesso_consigliato' : tupla[13] 
                                }
                            })
                
                return data
            
    def set_ordine(self, id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Ordine(id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id;
                            ''', (id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento))
                nuovo_id = cur.fetchone()[0]
                return nuovo_id
            
    def set_ordineoggetto(self, id_ordine, id_oggetto):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Ordineoggetto(id_ordine, id_oggetto)
                            VALUES (%s, %s)
                            RETURNING id;
                            ''', (id_ordine, id_oggetto))
                nuovo_id = cur.fetchone()[0]
                return cur.statusmessage, nuovo_id
            
    def dt_get_oggetto(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT * 
                            FROM Oggetto
                            ORDER BY id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                data = []
                for tupla in cur:
                    data.append({
                                'id'                : tupla[0], 
                                'name'              : tupla[1], 
                                'category'          : tupla[2], 
                                'prezzo'            : float(tupla[3]), 
                                'qta_magazzino'     : tupla[6], 
                                'qta_scaffale'      : tupla[7], 
                                'sconto'            : tupla[8],
                                'age'               : tupla[9], 
                                'gender'            : tupla[10]
                            })
                return data

    def set_emozione(self, id_cliente, id_oggetto, eta, sesso, indice_gradimento):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            INSERT INTO Emozione(id_cliente, id_oggetto, eta, sesso, indice_gradimento)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id;
                            ''', (id_cliente, id_oggetto, eta, sesso, indice_gradimento))
                nuovo_id = cur.fetchone()[0]
                return cur.statusmessage, nuovo_id
            
    def get_emozioni(self):
        with self.connection:
            with self.connection.cursor() as cur:
                cur.execute('''
                            SELECT *
                            FROM Emozione AS E INNER JOIN Cliente AS C ON E.id_cliente = C.id
                                                    INNER JOIN Oggetto AS O ON E.id_oggetto = O.id
                            ORDER BY O.id;
                            ''')

                if (cur.rowcount == 0):
                    return {}
                
                # Estrai i nomi delle colonne dal cursore
                #column_names = [column[0] for column in cur.description]
                #print column_names

                data = []
                for tupla in cur:
                    data.append({
                                'id'                : tupla[0], 
                                'username'          : tupla[7], 
                                'titolo'            : tupla[12], 
                                'prezzo'            : float(tupla[14]), 
                                'eta'               : tupla[3], 
                                'sesso'             : tupla[4], 
                                'indice_gradimento' : tupla[5],
                                'foto'              : tupla[16]
                            })
                return data