/*
 Navicat Premium Data Transfer

 Source Server         : local
 Source Server Type    : PostgreSQL
 Source Server Version : 110002 (110002)
 Source Host           : localhost:5432
 Source Catalog        : naochallenge2024
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 110002 (110002)
 File Encoding         : 65001

 Date: 05/03/2024 11:37:05
*/


-- ----------------------------
-- Table structure for abbinamento
-- ----------------------------
DROP TABLE IF EXISTS "public"."abbinamento";
CREATE TABLE "public"."abbinamento" (
  "id" int4 NOT NULL DEFAULT nextval('abbinamento_id_seq'::regclass),
  "id_oggetto1" int4,
  "id_oggetto2" int4
)
;
ALTER TABLE "public"."abbinamento" OWNER TO "postgres";

-- ----------------------------
-- Records of abbinamento
-- ----------------------------
BEGIN;
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (3, 2, 22);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (4, 36, 34);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (5, 22, 2);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (6, 7, 31);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (7, 31, 7);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (8, 42, 37);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (9, 11, 5);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (10, 5, 11);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (11, 39, 2);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (12, 32, 40);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (13, 40, 32);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (14, 33, 40);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (15, 41, 38);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (16, 38, 41);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (17, 14, 42);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (18, 34, 36);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (19, 37, 42);
INSERT INTO "public"."abbinamento" ("id", "id_oggetto1", "id_oggetto2") VALUES (20, 35, 42);
COMMIT;

-- ----------------------------
-- Table structure for carrello
-- ----------------------------
DROP TABLE IF EXISTS "public"."carrello";
CREATE TABLE "public"."carrello" (
  "id" int4 NOT NULL DEFAULT nextval('carrello_id_seq'::regclass),
  "id_cliente" int4
)
;
ALTER TABLE "public"."carrello" OWNER TO "postgres";

-- ----------------------------
-- Records of carrello
-- ----------------------------
BEGIN;
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (1, 1);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (2, 2);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (3, 3);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (4, 4);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (5, 5);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (6, 6);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (7, 7);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (8, 8);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (9, 9);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (10, 10);
INSERT INTO "public"."carrello" ("id", "id_cliente") VALUES (11, 11);
COMMIT;

-- ----------------------------
-- Table structure for carrellooggetto
-- ----------------------------
DROP TABLE IF EXISTS "public"."carrellooggetto";
CREATE TABLE "public"."carrellooggetto" (
  "id" int4 NOT NULL DEFAULT nextval('carrellooggetto_id_seq'::regclass),
  "id_carrello" int4,
  "id_oggetto" int4
)
;
ALTER TABLE "public"."carrellooggetto" OWNER TO "postgres";

-- ----------------------------
-- Records of carrellooggetto
-- ----------------------------
BEGIN;
INSERT INTO "public"."carrellooggetto" ("id", "id_carrello", "id_oggetto") VALUES (1, 1, 1);
INSERT INTO "public"."carrellooggetto" ("id", "id_carrello", "id_oggetto") VALUES (2, 1, 4);
INSERT INTO "public"."carrellooggetto" ("id", "id_carrello", "id_oggetto") VALUES (11, 9, 39);
INSERT INTO "public"."carrellooggetto" ("id", "id_carrello", "id_oggetto") VALUES (12, 9, 2);
COMMIT;

-- ----------------------------
-- Table structure for cliente
-- ----------------------------
DROP TABLE IF EXISTS "public"."cliente";
CREATE TABLE "public"."cliente" (
  "id" int4 NOT NULL DEFAULT nextval('cliente_id_seq'::regclass),
  "username" varchar COLLATE "pg_catalog"."default",
  "nome" varchar COLLATE "pg_catalog"."default",
  "cognome" varchar COLLATE "pg_catalog"."default",
  "password" varchar COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."cliente" OWNER TO "postgres";

-- ----------------------------
-- Records of cliente
-- ----------------------------
BEGIN;
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (1, 'alberto_rubini', 'Alberto', 'Rubini', 'c582dec943ff7b743aa0691df291cea6');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (2, 'antonio_galati', 'Antonio', 'Galati', '4e42f7dd43ecbfe104de58610557c5ba');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (3, 'arianna_antonelli', 'Arianna', 'Antonelli', '4124bc0a9335c27f086f24ba207a4912');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (4, 'aurora_savoia', 'Aurora', 'Savoia', 'f970e2767d0cfe75876ea857f92e319b');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (5, 'chiara_demarchi', 'Chiara', 'De Marchi', '6865aeb3a9ed28f9a79ec454b259e5d0');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (6, 'davide_masini', 'Davide', 'Masini', '608e7dc116de7157306012b4f0be82ac');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (7, 'edoardo_polfranceschi', 'Edoardo', 'Polfranceschi', '6aa1e040c8b4607538970731e4040ed6');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (8, 'giacomo_santi', 'Giacomo', 'Santi', '1d8d5e912302108b5e88c3e77fcad378');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (9, 'giovanni_bellorio', 'Giovanni', 'Bellorio', '7885444af42e4b30c518c5be17c8850b');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (10, 'laura_mascalzoni', 'Laura', 'Mascalzoni', '192292e35fbe73f6d2b8d96bd1b6697d');
INSERT INTO "public"."cliente" ("id", "username", "nome", "cognome", "password") VALUES (11, 'shenal_fernando', 'Shenal', 'Fernando', '60d31eb37595dd44584be5ef363283e3');
COMMIT;

-- ----------------------------
-- Table structure for emozione
-- ----------------------------
DROP TABLE IF EXISTS "public"."emozione";
CREATE TABLE "public"."emozione" (
  "id" int4 NOT NULL DEFAULT nextval('emozione_id_seq'::regclass),
  "id_cliente" int4,
  "id_oggetto" int4,
  "eta" int4,
  "sesso" varchar COLLATE "pg_catalog"."default",
  "indice_gradimento" int4
)
;
ALTER TABLE "public"."emozione" OWNER TO "postgres";

-- ----------------------------
-- Records of emozione
-- ----------------------------
BEGIN;
INSERT INTO "public"."emozione" ("id", "id_cliente", "id_oggetto", "eta", "sesso", "indice_gradimento") VALUES (1, 1, 18, 22, 'M', 100);
COMMIT;

-- ----------------------------
-- Table structure for oggetto
-- ----------------------------
DROP TABLE IF EXISTS "public"."oggetto";
CREATE TABLE "public"."oggetto" (
  "id" int4 NOT NULL DEFAULT nextval('oggetto_id_seq'::regclass),
  "titolo" varchar COLLATE "pg_catalog"."default",
  "categoria" varchar COLLATE "pg_catalog"."default",
  "prezzo" numeric,
  "descrizione" varchar COLLATE "pg_catalog"."default",
  "foto" varchar COLLATE "pg_catalog"."default",
  "qta_magazzino" int4,
  "qta_scaffale" int4,
  "sconto" int4,
  "eta_consigliata" int4,
  "sesso_consigliato" varchar COLLATE "pg_catalog"."default",
  "qr_code" varchar COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."oggetto" OWNER TO "postgres";

-- ----------------------------
-- Records of oggetto
-- ----------------------------
BEGIN;
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (4, 'dad bracelet', 'bracelet', 145.00, 'White, Rhodium plated', '../static/img/dad_bracelet_1.png', 5, 1, 10, 30, 'M', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (6, 'dextera bracelet', 'bracelet', 155.00, 'Pavé, Mixed links, Black, Ruthenium plated', '../static/img/dextera_bracelet_1.png', 9, 1, 10, 40, 'M', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (7, 'dextera hoop earrings', 'earrings', 155.00, 'Asymmetrical design, Interlocking loop, White, Rhodium plated', '../static/img/dextera_hoop_earrings_1.png', 6, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (8, 'dextera necklace', 'necklace', 175.00, 'Pavé, Mixed links, Black, Ruthenium plated', '../static/img/dextera_necklace_1.png', 7, 1, 10, 40, 'M', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (10, 'florere stud earrings', 'earrings', 195.00, 'Flower, Pink, Gold-tone plated', '../static/img/florere_stud_earrings_1.png', 5, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (13, 'gema necklace', 'necklace', 195.00, 'Mixed cuts. Multicolored. Rhodium plated', '../static/img/gema_necklace_1.png', 4, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (14, 'matrix tennis necklace', 'necklace', 250.00, 'Round cut, Yellow, Gold-tone plated', '../static/img/matrix_tennis_necklace_1.png', 1, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (17, 'matrix stud earrings', 'earrings', 95.00, 'Rectangular cut, Green, Gold-tone plated', '../static/img/matrix_stud_earrings_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (19, 'mesmera cocktail ring', 'ring', 135.00, 'Octagon cut, White, Rhodium plated', '../static/img/mesmera_cocktail_ring_1.png', 3, 1, 10, 80, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (20, 'mesmera necklace', 'necklace', 950.00, 'Statement, Mixed cuts, White, Rhodium plated', '../static/img/mesmera_necklace_1.png', 4, 1, 10, 80, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (23, 'stella drop earrings', 'earrings', 135.00, 'Kite cut, Star, White, Rose gold-tone plated', '../static/img/stella_drop_earrings_1.png', 7, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (24, 'stella necklace', 'necklace', 175.00, 'Star, White, Rose gold-tone plated', '../static/img/stella_necklace_1.png', 4, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (26, 'stone hoop earrings', 'earrings', 125.00, 'Pavé, Large, White. Rhodium plated', '../static/img/stone_hoop_earrings_1.png', 4, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (29, 'swarovski swan stud earrings', 'earrings', 129.00, 'Swan. Black. Rose gold-tone plated', '../static/img/swarovski_swan_stud_earrings_1.png', 5, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (37, 'millenia necklace', 'necklace', 550.00, 'Oversized crystals, Octagon cut, Green, Gold-tone plated', '../static/img/millenia_necklace_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (34, 'meteora drop earrings', 'earrings', 89.00, 'White, Rhodium plated', '../static/img/meteora_drop_earrings_1.png', 3, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (11, 'gema bracelet', 'bracelet', 195.00, 'Mixed cuts, Blue, Rhodium plated', '../static/img/gema_bracelet_1.png', 8, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (30, 'vittore ring', 'ring', 75.00, 'Round cut. White, Gold-tone finish', '../static/img/vittore_ring_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (39, 'hyperbola drop earrings', 'earrings', 230.00, 'Heart, Blue, Rhodium plated', '../static/img/hyperbola_drop_earrings_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (5, 'dancing swan necklace', 'necklace', 155.00, 'Swan, Blue, Rhodium plated', '../static/img/dancing_swan_necklace_1.png', 7, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (9, 'florere necklace', 'necklace', 175.00, 'Flower, Pink, Gold-tone plated', '../static/img/florere_necklace_1.png', 4, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (12, 'gema drop earrings', 'earrings', 195.00, 'Asymmetrical design, Mixed cuts, Long, Multicolored, Rhodium plated', '../static/img/gema_drop_earrings_1.png', 5, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (15, 'matrix drop earrings', 'earrings', 195.00, 'Mixed cuts, Green, Rhodium plated', '../static/img/matrix_drop_earrings_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (16, 'matrix ring', 'ring', 125.00, 'Baguette cut, Gray, Ruthenium plated', '../static/img/matrix_ring_1.png', 7, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (18, 'mesmera bracelet', 'bracelet', 400.00, 'Oversized crystals, White, Rhodium plated', '../static/img/mesmera_bracelet_1.png', 6, 1, 10, 80, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (21, 'millenia cocktail ring', 'ring', 135.00, 'Pear cut, Pavé, White, Rhodium plated', '../static/img/millenia_cocktail_ring_1.png', 8, 1, 10, 80, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (22, 'crystalline delight watch', 'watch', 300.00, 'Swiss Made, Metal bracelet, White, Stainless steel', '../static/img/crystalline_delight_watch_1.png', 8, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (31, 'disney mickey mouse pendant', 'necklace', 125.00, 'White, Rhodium plated', '../static/img/disney_mickey_mouse_pendant_1.png', 3, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (32, 'lucent bangle', 'bracelet', 215.00, 'Crystals, Magnet, Gold-tone PVD', '../static/img/lucent_bangle_1.png', 3, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (33, 'lucent hoop earrings', 'earrings', 300.00, 'Statement, Octagon shape, Pink', '../static/img/lucent_hoop_earrings_1.png', 3, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (35, 'stilla necklace', 'necklace', 175.00, 'Mixed cuts, Multicolored, Gold-tone plated', '../static/img/stilla_necklace_1.png', 3, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (36, 'creativity pendant', 'necklace', 75.00, 'White, Rose gold-tone plated', '../static/img/creativity_pendant_1.png', 3, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (38, 'luna drop earrings', 'earrings', 175.00, 'Moon, White, Rhodium plated', '../static/img/luna_drop_earrings_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (40, 'lucent cocktail ring', 'ring', 250.00, 'Octagon cut, Green', '../static/img/lucent_cocktail_ring_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (41, 'luna cocktail ring', 'ring', 175.00, 'Moon, White, Rhodium plated', '../static/img/luna_cocktail_ring_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (42, 'florere cocktail ring', 'ring', 400.00, 'Pavé, Flower, Pink, Gold-tone plated', '../static/img/florere_cocktail_ring_1.png', 3, 1, 10, 50, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (25, 'stella stud earrings', 'earrings', 125.00, 'Round cut, Star, White, Rose gold-tone plated', '../static/img/stella_stud_earrings_1.png', 7, 1, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (27, 'swarovski iconic swan earring jackets', 'earrings', 125.00, 'Swan. Black. Rose gold-tone plated', '../static/img/swarovski_iconic_swan_earring_jackets_1.png', 9, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (28, 'swarovski iconic swan pendant', 'necklace', 115.00, 'Swan. Black. Rose gold-tone plated', '../static/img/swarovski_iconic_swan_pendant_1.png', 6, 1, 10, 30, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (2, 'angelic necklace', 'necklace', 230.00, 'Round cut. White. Rhodium plated', '../static/img/angelic_necklace_1.png', 3, 0, 10, 15, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (3, 'constella cocktail ring', 'ring', 135.00, 'Round cut. Pavé, White, Rhodium plated', '../static/img/constella_cocktail_ring_1.png', 2, 0, 10, 45, 'F', NULL);
INSERT INTO "public"."oggetto" ("id", "titolo", "categoria", "prezzo", "descrizione", "foto", "qta_magazzino", "qta_scaffale", "sconto", "eta_consigliata", "sesso_consigliato", "qr_code") VALUES (1, 'angelic bracelet', 'bracelet', 155.00, 'Round cut. Pavé. Small. White. Rhodium plated', '../static/img/angelic_bracelet_1.png', 4, 0, 10, 15, 'F', NULL);
COMMIT;

-- ----------------------------
-- Table structure for ordine
-- ----------------------------
DROP TABLE IF EXISTS "public"."ordine";
CREATE TABLE "public"."ordine" (
  "id" int4 NOT NULL DEFAULT nextval('ordine_id_seq'::regclass),
  "id_cliente" int4,
  "prezzo_totale" numeric,
  "data_acquisto" date,
  "ora_acquisto" time(6),
  "modalita_pagamento" varchar COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."ordine" OWNER TO "postgres";

-- ----------------------------
-- Records of ordine
-- ----------------------------
BEGIN;
INSERT INTO "public"."ordine" ("id", "id_cliente", "prezzo_totale", "data_acquisto", "ora_acquisto", "modalita_pagamento") VALUES (1, 7, 125.00, '2024-01-15', '12:30:00', 'bonifico');
INSERT INTO "public"."ordine" ("id", "id_cliente", "prezzo_totale", "data_acquisto", "ora_acquisto", "modalita_pagamento") VALUES (2, 4, 700.00, '2024-01-05', '11:50:44', 'bonifico');
INSERT INTO "public"."ordine" ("id", "id_cliente", "prezzo_totale", "data_acquisto", "ora_acquisto", "modalita_pagamento") VALUES (3, 9, 230.0, '2024-02-04', '22:33:04', 'carta');
INSERT INTO "public"."ordine" ("id", "id_cliente", "prezzo_totale", "data_acquisto", "ora_acquisto", "modalita_pagamento") VALUES (4, 9, 460.0, '2024-02-04', '22:50:32', 'carta');
COMMIT;

-- ----------------------------
-- Table structure for ordineoggetto
-- ----------------------------
DROP TABLE IF EXISTS "public"."ordineoggetto";
CREATE TABLE "public"."ordineoggetto" (
  "id" int4 NOT NULL DEFAULT nextval('ordineoggetto_id_seq'::regclass),
  "id_ordine" int4,
  "id_oggetto" int4
)
;
ALTER TABLE "public"."ordineoggetto" OWNER TO "postgres";

-- ----------------------------
-- Records of ordineoggetto
-- ----------------------------
BEGIN;
INSERT INTO "public"."ordineoggetto" ("id", "id_ordine", "id_oggetto") VALUES (1, 1, 27);
INSERT INTO "public"."ordineoggetto" ("id", "id_ordine", "id_oggetto") VALUES (2, 2, 22);
INSERT INTO "public"."ordineoggetto" ("id", "id_ordine", "id_oggetto") VALUES (3, 3, 39);
INSERT INTO "public"."ordineoggetto" ("id", "id_ordine", "id_oggetto") VALUES (4, 4, 2);
INSERT INTO "public"."ordineoggetto" ("id", "id_ordine", "id_oggetto") VALUES (5, 4, 39);
COMMIT;

-- ----------------------------
-- Primary Key structure for table abbinamento
-- ----------------------------
ALTER TABLE "public"."abbinamento" ADD CONSTRAINT "abbinamento_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table carrello
-- ----------------------------
ALTER TABLE "public"."carrello" ADD CONSTRAINT "carrello_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table carrellooggetto
-- ----------------------------
ALTER TABLE "public"."carrellooggetto" ADD CONSTRAINT "carrellooggetto_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table cliente
-- ----------------------------
ALTER TABLE "public"."cliente" ADD CONSTRAINT "cliente_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table emozione
-- ----------------------------
ALTER TABLE "public"."emozione" ADD CONSTRAINT "emozione_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table oggetto
-- ----------------------------
ALTER TABLE "public"."oggetto" ADD CONSTRAINT "oggetto_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table ordine
-- ----------------------------
ALTER TABLE "public"."ordine" ADD CONSTRAINT "ordine_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table ordineoggetto
-- ----------------------------
ALTER TABLE "public"."ordineoggetto" ADD CONSTRAINT "ordineoggetto_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Foreign Keys structure for table abbinamento
-- ----------------------------
ALTER TABLE "public"."abbinamento" ADD CONSTRAINT "abbinamento_id_oggetto1_fkey" FOREIGN KEY ("id_oggetto1") REFERENCES "public"."oggetto" ("id") ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE "public"."abbinamento" ADD CONSTRAINT "abbinamento_id_oggetto2_fkey" FOREIGN KEY ("id_oggetto2") REFERENCES "public"."oggetto" ("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table carrello
-- ----------------------------
ALTER TABLE "public"."carrello" ADD CONSTRAINT "carrello_id_cliente_fkey" FOREIGN KEY ("id_cliente") REFERENCES "public"."cliente" ("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table carrellooggetto
-- ----------------------------
ALTER TABLE "public"."carrellooggetto" ADD CONSTRAINT "carrellooggetto_id_carrello_fkey" FOREIGN KEY ("id_carrello") REFERENCES "public"."carrello" ("id") ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE "public"."carrellooggetto" ADD CONSTRAINT "carrellooggetto_id_oggetto_fkey" FOREIGN KEY ("id_oggetto") REFERENCES "public"."oggetto" ("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table emozione
-- ----------------------------
ALTER TABLE "public"."emozione" ADD CONSTRAINT "emozione_id_cliente_fkey" FOREIGN KEY ("id_cliente") REFERENCES "public"."cliente" ("id") ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE "public"."emozione" ADD CONSTRAINT "emozione_id_oggetto_fkey" FOREIGN KEY ("id_oggetto") REFERENCES "public"."oggetto" ("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table ordine
-- ----------------------------
ALTER TABLE "public"."ordine" ADD CONSTRAINT "ordine_id_cliente_fkey" FOREIGN KEY ("id_cliente") REFERENCES "public"."cliente" ("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table ordineoggetto
-- ----------------------------
ALTER TABLE "public"."ordineoggetto" ADD CONSTRAINT "ordineoggetto_id_oggetto_fkey" FOREIGN KEY ("id_oggetto") REFERENCES "public"."oggetto" ("id") ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE "public"."ordineoggetto" ADD CONSTRAINT "ordineoggetto_id_ordine_fkey" FOREIGN KEY ("id_ordine") REFERENCES "public"."ordine" ("id") ON DELETE SET NULL ON UPDATE CASCADE;
