SELECT * FROM Cliente;
SELECT * FROM Carrello;
SELECT * FROM Oggetto;
SELECT * FROM CarrelloOggetto;
SELECT * FROM Ordine;
SELECT * FROM OrdineOggetto;
SELECT * FROM Emozione;
SELECT * FROM Abbinamento;

SELECT * 
FROM Carrello AS Car INNER JOIN Cliente AS Cli 
ON Car.id_cliente = Cli.id;

SELECT * 
FROM CarrelloOggetto AS CarOgg INNER JOIN Carrello AS Car
ON CarOgg.id_carrello = Car.id;

SELECT * 
FROM Ordine AS Ord INNER JOIN Cliente AS Cli 
ON Ord.id_cliente = Cli.id;

SELECT * 
FROM OrdineOggetto AS OrdOgg INNER JOIN Ordine AS Ord 
ON OrdOgg.id_ordine = Ord.id;

SELECT * 
FROM OrdineOggetto AS OrdOgg INNER JOIN Oggetto AS Ogg 
ON OrdOgg.id_oggetto = Ogg.id;

SELECT * 
FROM Emozione AS Emo INNER JOIN Cliente AS Cli 
ON Emo.id_cliente = Cli.id;

SELECT * 
FROM Emozione AS Emo INNER JOIN Oggetto AS Ogg 
ON Emo.id_oggetto = Ogg.id;

SELECT * 
FROM Abbinamento AS Abb INNER JOIN Oggetto AS Ogg 
ON Abb.id_oggetto1 = Ogg.id;

SELECT * 
FROM Abbinamento AS Abb INNER JOIN Oggetto AS Ogg 
ON Abb.id_oggetto2 = Ogg.id;