INSERT INTO Cliente (username, nome, cognome, password) VALUES 
	('alberto_rubini', 			'Alberto', 	'Rubini', 			'c582dec943ff7b743aa0691df291cea6'),
	('antonio_galati', 			'Antonio', 	'Galati', 			'4e42f7dd43ecbfe104de58610557c5ba'),
	('arianna_antonelli', 		'Arianna', 	'Antonelli', 		'4124bc0a9335c27f086f24ba207a4912'),
	('aurora_savoia', 			'Aurora', 	'Savoia', 			'f970e2767d0cfe75876ea857f92e319b'),
	('chiara_demarchi', 		'Chiara', 	'De Marchi', 		'6865aeb3a9ed28f9a79ec454b259e5d0'),
	('davide_masini', 			'Davide', 	'Masini', 			'608e7dc116de7157306012b4f0be82ac'),
	('edoardo_polfranceschi', 	'Edoardo', 	'Polfranceschi', 	'6aa1e040c8b4607538970731e4040ed6'),
	('giacomo_santi', 			'Giacomo', 	'Santi', 			'1d8d5e912302108b5e88c3e77fcad378'),
	('giovanni_bellorio', 		'Giovanni',	'Bellorio', 		'7885444af42e4b30c518c5be17c8850b'),
	('laura_mascalzoni', 		'Laura', 	'Mascalzoni', 		'192292e35fbe73f6d2b8d96bd1b6697d'),
	('shenal_fernando', 		'Shenal', 	'Fernando', 		'60d31eb37595dd44584be5ef363283e3');

INSERT INTO Carrello (id_cliente) VALUES
	(1),
	(2),
	(3),
	(4),
	(5),
	(6),
	(7),
	(8),
	(9),
	(10),
	(11);
	
INSERT INTO Oggetto (titolo, 					categoria, 		prezzo, 	descrizione,															foto,																		qta_magazzino, 	qta_scaffale, 	sconto, 	eta_consigliata, 	sesso_consigliato) VALUES
	('angelic bracelet', 						'bracelet',		'155.00',	'Round cut. Pavé. Small. White. Rhodium plated', 						('../static/img/angelic_bracelet_1.png'),									4, 				0, 				10, 		15, 				'F'),
	('angelic necklace', 						'necklace', 	'230.00', 	'Round cut. White. Rhodium plated',										('../static/img/angelic_necklace_1.png'),									3, 				1,				10, 		15, 				'F'),
	('constella cocktail ring', 				'ring', 		'135.00', 	'Round cut. Pavé, White, Rhodium plated', 								('../static/img/constella_cocktail_ring_1.png'),							2, 				0, 				10,			45, 				'F'),
	('dad bracelet', 							'bracelet', 	'145.00', 	'White, Rhodium plated', 												('../static/img/dad_bracelet_1.png'),										5, 				1, 				10, 		30, 				'M'),
	('dancing swan necklace', 					'necklace', 	'155.00', 	'Swan, Blue, Rhodium plated', 											('../static/img/dancing_swan_necklace_1.png'),								7, 				0, 				10,			15,					'F'),
	('dextera bracelet', 						'bracelet', 	'155.00', 	'Pavé, Mixed links, Black, Ruthenium plated',  							('../static/img/dextera_bracelet_1.png'),									9, 				1, 				10, 		40, 				'M'),
	('dextera hoop earrings', 					'earrings', 	'155.00', 	'Asymmetrical design, Interlocking loop, White, Rhodium plated', 		('../static/img/dextera_hoop_earrings_1.png'),								6, 				1,				10,			15,					'F'),
	('dextera necklace', 						'necklace', 	'175.00', 	'Pavé, Mixed links, Black, Ruthenium plated', 							('../static/img/dextera_necklace_1.png'),									7, 				1, 				10,			40, 				'M'),
	('florere necklace', 						'necklace', 	'175.00', 	'Flower, Pink, Gold-tone plated', 										('../static/img/florere_necklace_1.png'),									4, 				0,				10, 		50, 				'F'),
	('florere stud earrings', 					'earrings', 	'195.00', 	'Flower, Pink, Gold-tone plated', 										('../static/img/florere_stud_earrings_1.png'),								5, 				1,				10, 		50, 				'F'),
	('gema bracelet', 							'bracelet', 	'195.00', 	'Mixed cuts, Blue, Rhodium plated', 									('../static/img/gema_bracelet_1.png'),										8, 				1, 				10, 		30, 				'F'),
	('gema drop earrings', 						'earrings', 	'195.00', 	'Asymmetrical design, Mixed cuts, Long, Multicolored, Rhodium plated', 	('../static/img/gema_drop_earrings_1.png'),									5, 				0, 				10, 		30, 				'F'),
	('gema necklace', 							'necklace', 	'195.00', 	'Mixed cuts. Multicolored. Rhodium plated', 							('../static/img/gema_necklace_1.png'),										4, 				1, 				10, 		30, 				'F'),
	('matrix tennis necklace', 					'necklace', 	'250.00', 	'Round cut, Yellow, Gold-tone plated', 									('../static/img/matrix_tennis_necklace_1.png'),								1, 				1, 				10, 		50, 				'F'),
	('matrix drop earrings', 					'earrings', 	'195.00', 	'Mixed cuts, Green, Rhodium plated', 									('../static/img/matrix_drop_earrings_1.png'),								3, 				0, 				10, 		50, 				'F'),
	('matrix ring',								'ring', 		'125.00', 	'Baguette cut, Gray, Ruthenium plated',									('../static/img/matrix_ring_1.png'),										7, 				0, 				10, 		50, 				'F'),
	('matrix stud earrings', 					'earrings', 	'95.00', 	'Rectangular cut, Green, Gold-tone plated',								('../static/img/matrix_stud_earrings_1.png'),								3, 				1, 				10, 		50, 				'F'),
	('mesmera bracelet', 						'bracelet', 	'400.00', 	'Oversized crystals, White, Rhodium plated', 							('../static/img/mesmera_bracelet_1.png'),									6, 				0, 				10, 		80, 				'F'),
	('mesmera cocktail ring', 					'ring', 		'135.00', 	'Octagon cut, White, Rhodium plated', 									('../static/img/mesmera_cocktail_ring_1.png'),								3, 				1, 				10, 		80, 				'F'),
	('mesmera necklace', 						'necklace', 	'950.00', 	'Statement, Mixed cuts, White, Rhodium plated', 						('../static/img/mesmera_necklace_1.png'),									4, 				1, 				10, 		80,					'F'),
	('millenia cocktail ring', 					'ring', 		'135.00', 	'Pear cut, Pavé, White, Rhodium plated', 								('../static/img/millenia_cocktail_ring_1.png'),								8, 				0, 				10, 		80, 				'F'),
	('crystalline delight watch', 				'watch', 		'300.00', 	'Swiss Made, Metal bracelet, White, Stainless steel', 					('../static/img/crystalline_delight_watch_1.png'),							8, 				0, 				10, 		30, 				'F'),
	('stella drop earrings', 					'earrings',		'135.00', 	'Kite cut, Star, White, Rose gold-tone plated',							('../static/img/stella_drop_earrings_1.png'),								7, 				1, 				10, 		15, 				'F'),
	('stella necklace', 						'necklace', 	'175.00', 	'Star, White, Rose gold-tone plated', 									('../static/img/stella_necklace_1.png'),									4, 				1, 				10, 		15, 				'F'),
	('stella stud earrings', 					'earrings', 	'125.00', 	'Round cut, Star, White, Rose gold-tone plated', 						('../static/img/stella_stud_earrings_1.png'),								7, 				0, 				10, 		15, 				'F'),
	('stone hoop earrings', 					'earrings', 	'125.00', 	'Pavé, Large, White. Rhodium plated', 									('../static/img/stone_hoop_earrings_1.png'),								4, 				1, 				10, 		15, 				'F'),
	('swarovski iconic swan earring jackets', 	'earrings', 	'125.00', 	'Swan. Black. Rose gold-tone plated', 									('../static/img/swarovski_iconic_swan_earring_jackets_1.png'),				9, 				0, 				10, 		30, 				'F'),
	('swarovski iconic swan pendant', 			'necklace', 	'115.00', 	'Swan. Black. Rose gold-tone plated', 									('../static/img/swarovski_iconic_swan_pendant_1.png'),						6, 				0, 				10, 		30, 				'F'),
	('swarovski swan stud earrings', 			'earrings', 	'129.00', 	'Swan. Black. Rose gold-tone plated', 									('../static/img/swarovski_swan_stud_earrings_1.png'),						5, 				1, 				10, 		30, 				'F'),
	('vittore ring', 							'ring', 		'75.00', 	'Round cut. White, Gold-tone finish', 									('../static/img/vittore_ring_1.png'),										3, 				0, 				10, 		50, 				'F'),
	('disney mickey mouse pendant', 			'necklace', 	'125.00', 	'White, Rhodium plated', 									  			('../static/img/disney_mickey_mouse_pendant_1.png'),						3, 				0, 				10, 		15, 				'F'),
	('lucent bangle', 							'bracelet', 	'215.00', 	'Crystals, Magnet, Gold-tone PVD', 									  	('../static/img/lucent_bangle_1.png'),										3, 				0, 				10, 		30, 				'F'),
	('lucent hoop earrings', 					'earrings', 	'300.00', 	'Statement, Octagon shape, Pink', 									  	('../static/img/lucent_hoop_earrings_1.png'),								3, 				0, 				10, 		30, 				'F'),
	('meteora drop earrings', 					'earrings', 	'89.00', 	'White, Rhodium plated', 									  			('../static/img/meteora_drop_earrings_1.png'),								3, 				0, 				10, 		30, 				'F'),
	('stilla necklace', 						'necklace', 	'175.00', 	'Mixed cuts, Multicolored, Gold-tone plated', 							('../static/img/stilla_necklace_1.png'),									3, 				0, 				10, 		30, 				'F'),
	('creativity pendant', 						'necklace', 	'75.00', 	'White, Rose gold-tone plated', 									  	('../static/img/creativity_pendant_1.png'),									3, 				0, 				10, 		30, 				'F'),
	('millenia necklace', 						'necklace', 	'550.00', 	'Oversized crystals, Octagon cut, Green, Gold-tone plated', 			('../static/img/millenia_necklace_1.png'),									3, 				0, 				10, 		50, 				'F'),
	('luna drop earrings', 						'earrings', 	'175.00', 	'Moon, White, Rhodium plated', 									  		('../static/img/luna_drop_earrings_1.png'),									3, 				0, 				10, 		50, 				'F'),
	('hyperbola drop earrings',	 				'earrings', 	'230.00', 	'Heart, Blue, Rhodium plated', 									  		('../static/img/hyperbola_drop_earrings_1.png'),							3, 				0, 				10, 		50, 				'F'),
	('lucent cocktail ring', 					'ring', 		'250.00', 	'Octagon cut, Green', 									  				('../static/img/lucent_cocktail_ring_1.png'),								3, 				0, 				10, 		50, 				'F'),
	('luna cocktail ring', 						'ring', 		'175.00', 	'Moon, White, Rhodium plated', 									  		('../static/img/luna_cocktail_ring_1.png'),									3, 				0, 				10, 		50, 				'F'),
	('florere cocktail ring', 					'ring', 		'400.00', 	'Pavé, Flower, Pink, Gold-tone plated', 								('../static/img/florere_cocktail_ring_1.png'),								3, 				0, 				10, 		50, 				'F');

INSERT INTO CarrelloOggetto (id_carrello, id_oggetto) VALUES
	(1, 1),
	(1, 4);

INSERT INTO Ordine (id_cliente, prezzo_totale, data_acquisto, ora_acquisto, modalita_pagamento) VALUES
	(7, 	'125.00', 	'2024-01-15', 	'12:30:00', 	'bonifico'),
	(4, 	'700.00', 	'2024-01-05', 	'11:50:44', 	'bonifico');

INSERT INTO OrdineOggetto (id_ordine, id_oggetto) VALUES
	(1, 27),
	(2, 22);

INSERT INTO Emozione (id_cliente, id_oggetto, eta, sesso, indice_gradimento	) VALUES
	(1, 	18, 22, 	'M', 	100);
	
INSERT INTO Abbinamento (id_oggetto1, id_oggetto2) VALUES
	(2, 22),
	(36, 34),
	(22, 2),
	(7, 31),
	(31, 7),
	(42, 37),
	(11, 5),
	(5, 11),
	(39, 2),
	(32, 40),
	(40, 32),
	(33, 40),
	(41, 38),
	(38, 41),
	(14, 42),
	(34, 36),
	(37, 42),
	(35, 42);





