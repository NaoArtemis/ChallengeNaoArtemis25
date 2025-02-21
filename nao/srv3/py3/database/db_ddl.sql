CREATE TABLE Cliente (
	id			SERIAL,
	username	VARCHAR,
	nome		VARCHAR,
	cognome		VARCHAR,
	password	VARCHAR,
	
	PRIMARY KEY(id)
);

CREATE TABLE Carrello (
	id			SERIAL,
	id_cliente	INT,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_cliente) REFERENCES Cliente(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE Oggetto (
	id					SERIAL,
	titolo				VARCHAR,
	categoria			VARCHAR,
	prezzo				DECIMAL,
	descrizione			VARCHAR,
	foto				VARCHAR,
	qta_magazzino		INT,
	qta_scaffale		INT,
	sconto				INT,
	eta_consigliata		INT,
	sesso_consigliato	VARCHAR,
	qr_code				VARCHAR,
	
	PRIMARY KEY(id)
);

CREATE TABLE CarrelloOggetto (
	id			SERIAL,
	id_carrello	INT,
	id_oggetto 	INT,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_carrello) REFERENCES Carrello(id) ON UPDATE CASCADE ON DELETE SET NULL,
	FOREIGN KEY(id_oggetto) REFERENCES Oggetto(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE Ordine (
	id					SERIAL,
	id_cliente			INT,
	prezzo_totale		DECIMAL,
	data_acquisto		DATE,
	ora_acquisto		TIME,
	modalita_pagamento	VARCHAR,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_cliente) REFERENCES Cliente(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE OrdineOggetto (
	id			SERIAL,
	id_ordine	INT,
	id_oggetto	INT,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_ordine) REFERENCES Ordine(id) ON UPDATE CASCADE ON DELETE SET NULL,
	FOREIGN KEY(id_oggetto) REFERENCES Oggetto(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE Emozione (
	id					SERIAL,
	id_cliente			INT,
	id_oggetto			INT,
	eta					INT,
	sesso				VARCHAR,
	indice_gradimento	INT,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_cliente) REFERENCES Cliente(id) ON UPDATE CASCADE ON DELETE SET NULL,
	FOREIGN KEY(id_oggetto) REFERENCES Oggetto(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE Abbinamento (
	id			SERIAL,
	id_oggetto1	INT,
	id_oggetto2	INT,
	
	PRIMARY KEY(id),
	FOREIGN KEY(id_oggetto1) REFERENCES Oggetto(id) ON UPDATE CASCADE ON DELETE SET NULL,
	FOREIGN KEY(id_oggetto2) REFERENCES Oggetto(id) ON UPDATE CASCADE ON DELETE SET NULL
);