# -*- coding: utf-8 -*-

import re
import random
import requests
import json
from main import nao_animatedSayText, nao_speech_to_text

class Dialogo:

    db_helper = ""
    dialog = []

    def Angelic_bracelet(self):
        return "L Angelic_bracelet  e un classico intramontabile. Il braccialetto rodiato propone una fila di Clear Crystal a taglio circolare, ognuno incorniciato dal pave e di Clear Crystal. Il gioiello si abbina perfettamente agli altri accessori della collezione Angelic. Il bracciale appartiene alla collezione Angelic, con cristalli e placcato in rodio"

    def Angelic_necklace(self):
        return "L angelic necklace  e un raffinato collier si abbina perfettamente ad altre creazioni dell apprezzatissima linea Angelic Swarovski. Ideale per le occasioni speciali e da regalare,  e interpretato in Clear Crystal e metallo rodiato lucido. Questa collana appartiene alla collezione Angelic, con cristalli e placcato in rodio"

    def Constella_cocktail_ring(self):
        return "Il Constella cocktail ring  e un anello Constella, testimone della potenza della semplicita,  e un must del guardaroba. Risultato di una sapiente lavorazione, la pietra centrale dal taglio Princess ruba la scena ed  e valorizzata da un elegante pave e sulla galleria e dalla fascia placcata rodio. Indossalo da solo perch e brilli in tutto il suo splendore. L anello  appartiene alla collezzione Constella, con   e placcatura in rodio"

    def Dad_bracelet(self):
        return "Il dad bracelet  e placcato rodio ed  e un regalo ideale per la Festa del Papa. Al centro, la parola “PAPA”  e messa in rilievo con un delicato pave e di cristalli, mentre la massiccia catena a maglie presenta un estensore che garantisce una vestibilita perfetta. Pensato per questo giorno speciale, ma non solo, questo gioiello  e disponibile nelle dimensioni L e XL. Il bracciale appartiene alla collezione Mothers day, con cristalli e placcatura in rodio"

    def Dancing_swan_necklace(self):
        return "Simbolo di graziosa eleganza e nobilta, il cigno  e l icona ideale per i gioielli Swarovski. Questa collana presenta uno splendido set di cigni con numerose pietre bianche scintillanti. Al centro risiede un “dancing element” che brilla luminoso. Questa collana rodiata  e un regalo affascinante per una persona speciale. Questa collana appartiene alla collezione Dancing Swan, con cristalli, placcatura in rodio e zirconi"

    def Dextera_bracelet(self):
        return "Dettagli intricati incontrano una costruzione audace in questo moderno braccialetto della famiglia Dextera. Impreziosito dalla nostra celebre tecnica del pave e, presenta file di cristalli grigio-argento incastonati in una montatura placcata rutenio nero. Indossalo da solo o impilato, a seconda del tuo umore e del tuo outfit. Il bracciale appartiene alla collezione Dextera con cristalli e placcatura in rutenio"

    def Dextera_necklace(self):
        return "Ispirata alla nostra tradizione industriale, questa collana a tutto tondo raggiunge l equilibrio tra audace e raffinato. Caratterizzato da pave e di cristalli grigio argento e catene intrecciate,  e rifinito con una montatura placcata rutenio. Indossalo con altri pezzi della famiglia Dextera per un look audace. Questa collana appartiene alla collezione Dextera con cristalli e placcatura in rutenio"

    def Florere_necklace(self):
        return "Sublima il tuo stile con la bellezza luminosa della natura. Questa brillante collana della famiglia Florere  e realizzata con un fiore che sboccia, creato dalla combinazione di placcatura in tonalita oro e cristalli rosa sfaccettati. Il pendente  e disposto su una catena scorrevole con una singola pietra splendente all estremita. Indossa questo bellissimo gioiello per sprigionare la gioia primaverile. La collana appartiene alla collezione Florere, con zirconi e placcatura in rodio"

    def Florere_stud_earrings(self):
        return "Ispirato dalla meravigliosa bellezza della natura, questo gioiello luminoso sboccia con gli eleganti cristalli. Caratterizzati da una montatura placcata color oro, cinque petali rosa sono disposti attorno ad una pietra dalla tonalit a pi u scura. Scegli questi orecchini per illuminare la tua giornata. QUesti orecchini appartengono alla collezione Florere, con zirconi e placcatura color oro"

    def Gema_drop_earrings(self):
        return "Questi orecchini a buco Gema di un lussureggiante savoir-faire infonderanno in ogni tuo outfit tutta la bellezza della natura. Il mix di cristalli multicolor  e disposto in maniera originale ma rigorosa su una raffinata montatura placcata rodio che fa risplendere ogni sfumatura. Da abbinare a una collana della famiglia Gema o da indossare da soli per un effetto glamour disinvolto al calar della notte. Questi orecchini appartengono alla collezione Gema, con cristalli e placcatura in rodio"

    def Gema_necklace(self): 
        return "Gioiello davvero spettacolare della famiglia Gema, questa straordinaria collana  e realizzata con una serie di sfavillanti cristalli in un mix di tagli e colori. Questo gioiello regala gioia e migliora subito l umore in qualsiasi modo tu decida di indossarlo. Questa collana fa parte della collezione Gema, con cristalli e placcatura in rodio"

    def Matrix_drop_earrings(self):
        return "Questi versatili orecchini della famiglia Matrix si potranno indossare in due modi diversi. Il design placcato rodio presenta orecchini a lobo con Swarovski  zirconi rettangolari verdi e orecchini jacket con un  elegante goccia realizzata con cristalli Cry rotondi e uno Swarovski  zirconi rettangolare verde all  estremit a. Gli orecchini jacket hanno tre fori per regolarne la lunghezza. Inoltre, potrai anche staccarli per indossare solamente gli orecchini a lobo. Un gioiello straordinario che aggiunger a un immediato tocco glamour ad ogni stile. Questi orecchini fanno parte della colazione Matrix, con zirconi e placcatura in rodio"

    def Matrix_ring(self):
        return "Combinando un profondo aspetto metallico con una lucentezza accattivante, questo anello Matrix costituir a un aggiunta versatile al tuo portagioielli. In un processo che unisce scienza e magia, le pietre grigie con taglio baguette Princess sono incastonate su una fascia placcata rutenio. Un accessorio accattivante, perfetto da impilare, che consente molteplici combinazioni per tutte le occasioni. Questo anello fa parte della collezione Matrix, con cristalli e placcatura in rutenio"

    def Matrix_stud_earrings(self):
        return "Semplici ma affascinanti, questi borchie della famiglia Matrix offrono stile. Gli zirconi Swarovski rettangolari verde brillante sono inseriti in una raffinata montatura a doppia griffe placcata nella tonalit a oro. Scegli questi orecchini sfaccettati per un tocco di verde da far girare la testa. Questi orecchini fanno parte della collezione Matrix, con zirconi e placcatura color oro"

    def Mesmera_bracelet(self):
        return "Questo sorprendente braccialetto rigido mette in risalto le qualita ipnotiche della famiglia Mesmera. Innovativi grappoli di cristalli sono sapientemente disposti a forma di mosaico, che nasconde abilmente la montatura delle griffe placcate rodio. Indossalo come una dichiarazione diurna con le maniche della camicia arrotolate o come un semplice abito a trapezio per la sera; sia stravagante che senza tempo in egual misura. Questo braccialetto fa parte della famiglia Mesmera, disegnata dal Direttore Creativo Giovanna Engelbert per la Collezione I. Come chiuderlo e aprirlo(): posiziona la bacchetta arrotondata nel punto pi u largo del foro opposto e falla scorrere fino al punto pi u basso. Una volta in posizione, abbassare il fermo di sicurezza con anello a D sulla parte superiore per garantire il fissaggio. Per aprire, sollevare il fermo di sicurezza con anello a D e far scorrere il bastone verso il punto pi u largo per rilasciare. Questo bracciale fa parte della collezione Mesmera, con cristalli e placcatura in rodio"

    def Mesmera_cocktail_ring(self):
        return "Scenografico ma raffinato, questo anello dal motivo audace racchiude la forza ipnotica della famiglia Mesmera. Caratterizzato da un classico cristallo taglio Octagon al centro, questo gioiello senza tempo presenta un  elegante placcatura in rodio. Indossalo da solo o con altri anelli per un look da sera che non passer a inosservato. Parte della famiglia Mesmera, questo anello  e firmato dal Direttore Creativo Giovanna Engelbert per la Collection I. Questo anello fa parte della collezione Mersmera con cristalli e placcatura in rodio"

    def Mesmera_necklace(self):
        return "Accendi il tuo stile con i dettagli drammatici di questa collana Mesmera. Realizzato per un lusso stravagante, il design rodiato  e stato realizzato con una gamma completa di cristalli sfaccettati. I tagli e le dimensioni misti sono un omaggio al rinomato savoir-faire di Swarovski, regalandoti un gioiello meraviglioso e ipnotico. Regalalo a una persona cara come scelta definitiva. Questa collana fa parte della collezione Mesmera, con cristalli e placcatura in rodio"

    def Millenia_cocktail_ring(self):
        return "Indossato da solo o impilato, questo anello da cocktail Millenia catturer a sicuramente l attenzione. Abbagliante con un unica pietra taglio pera,  e incastonato su una fascia rodiata impreziosita da pave e per uno stile disinvolto e chic. Questo anello fa parte della collezione Millenia, con zirconi e placcatura in rodio"

    def Stella_drop_earrings(self):
        return "Sei nato per brillare. Prova questi squisiti orecchini pendenti ispirati alle stelle della famiglia Stella. Impreziositi da pietre taglio aquilone in raffinate montature a griffe, questi eleganti modelli sono rifiniti con placcatura nella tonalit a oro rosa. Abbinali ad altri meravigliosi design per creare un look unico come te. Questi orecchini fanno parte della collezione Stella, con zirconi e placcatura color oro rosa"

    def Stella_necklace(self):  
        return "Fai miracoli in questo bellissimo ciondolo della famiglia Stella. Presentando un elegante combinazione di perle di cristallo e pietre scintillanti rifinite con placcatura nella tonalit a oro rosa, questo design asimmetrico far a sicuramente girare la testa ovunque lo indossi. Modellalo da solo e preparati a brillare. Questa collana fa parte della collezione Stella, con perle di cristalli e zirconi"

    def Stella_stud_earrings(self): 
        return "Questi orecchini da sogno sono come la luce nell oscurit a. Parte della famiglia Stella, questo design scintillante offre un elegante mix di pietre brillanti a taglio rotondo e una perla di cristallo centrale rifinita con placcatura nella tonalit a oro rosa. Illumineranno ogni look, sia che li indossi in una pila di orecchie o che li lasci brillare da soli. Questi orecchini fanno parte della collezione stella, con perle di cristalli e zirconi"

    def Stone_hoop_earrings(self):
        return "Il gioiello perfetto, indispensabile per la tua collezione. Realizzati in metallo rodiato, questi orecchini a cerchio brillano di lucentezza grazie al pave e chiaro. Un look minimal, da indossare da solo o da studiare per creare uno stile ancora pi u stravagante. Abbinandosi facilmente a qualsiasi outfit, questi orecchini ti accompagneranno nell  arco di tutta la giornata. Questi orecchini fanno parte della collezione Stone, con cristalli e placcatura in rodio"
    
    def Swarovski_iconic_swan_earring_jackets(self):
        return "Raffinati e versatili, gli earring jackets placcati oro rosa riprendono il motivo del cigno, icona-simbolo Swarovski. La delicatezza del pave e di cristallo nero connota i gioielli di una luminosit a discreta. Indossati soli definiscono uno stile ricercato, mentre l  aggiunta degli elementi con Crystal Pearl rende gli orecchini pi u attuali.  Quesi orecchini fanno parte della collezione Iconic Swan, con cristalli e perle di cristali"
    
    def Swarovski_iconic_swan_pendant(self):
        return "L elegante pendente rodiato sfoggia l  iconico motivo a cigno Swarovski. La delicatezza dei superlativi cristalli neri fissati con lesclusiva tecnica Pointiage  connota il gioiello di una luminosit a discreta, perfetta da abbinare alle mise formali o casual. Il pendente misura 1 x 1 cm ed  e abbinato ad una catena di 38 cm. Per interpretare l  attualit a dello stile multi metallo indossate il gioiello con lo stesso modello in versione placcata oro rosa. Questa collana fa parte della collezione Iconic Swan, con cristalli e placcatura in rodio"

    def Swarovski_swan_stud_earrings(self):
        return "L iconico Cigno Swarovski prende forma elegantemente in questo paio di orecchini a lobo. Realizzato con le ali in alto, pronto a spiccare il volo, il cigno  e decorato con pave e di scintillanti cristalli trasparenti su una brillante placcatura rodio. Scegli questo gioiello per un delicato tocco di ispirazione quotidiana. Questi orecchini fanno parte della collezione Iconic Swan, con cristalli e placcatura in rodio"

    def Vittore_ring(self):
        return "Questo anello Vittore  e ideale per un sottile tocco di glitter. Combina il classico abbinamento di cristalli Cry dal taglio Round con una finitura color oro e ti conferir a un look raffinato e lussuoso. Realizzato per brillare su tutti i lati, potrai indossare questo gioiello con altri stili, come, ad esempio, un affascinante anello cocktail, per un contrasto efficace. Questo anello fa parte della collezione Vittore, con zirconi"

    def Luna_drop_earrings(self):
        return "Gli orecchini Luna combinano ispirazione lunare e stile sfavillante. Placcati in rodio, presentano eleganti mezzelune impreziosite da cristalli di cinque dimensioni, creando un sofisticato effetto pavé. Parte della famiglia Luna, questi gioielli sono adatti per ogni occasione, offrendo una scelta disinvolta. Il placcaggio rodio garantisce una lucentezza duratura e il design versatile si adatta a diversi outfit, rappresentando un elegante dichiarazione di stile."

    def Meteora_drop_earrings(self):
        return "Gli orecchini Meteora, ispirati al cosmo, arricchiscono il tuo stile con un tocco di meraviglia. Placcati in rodio, presentano catenine pendenti con motivi a cupola a forma di meteora, splendidamente impreziositi da cristalli Cry rotondi. Questi orecchini rappresentano una dichiarazione di eleganza semplice ma potente, aggiungendo luminosita al viso e un tocco di magia a ogni look. Una scelta facile per illuminare il tuo stile quotidiano con la brillantezza distintiva di Swarovski."

    def Lucent_hoop_earrings(self):
        return "Gli orecchini a cerchio Lucent in rosa sono un esempio lampante di cristallo saturo di colore completamente sfaccettato. Creati con maestria, questi orecchini sono rifiniti con una sottile chiusura a perno e sono versatili per qualsiasi outfit, dal casual all elegante. Parte della linea Lucent, sono un eccellente interpretazione del lusso sportivo firmato dalla Direttrice Creativa Giovanna Engelbert. La loro versatilita li rende adatti a molteplici occasioni, trasformando ogni giorno in un opportunita per brillare con la distintiva brillantezza Swarovski."

    def Dxtera_hoop_earrings(self):
        return "Gli orecchini a cerchio intrecciati Dextera presentano due anelli tubolari placcati rodio, di cui uno impreziosito da pavé all-over. Con un tocco asimmetrico, gli anelli sono disposti in modo diverso su ciascun gioiello. La possibilita di rimuovere il secondo cerchio offre una versatilita unica, permettendo di personalizzare il look a seconda dell occasione. Oltre a essere una dichiarazione di stile, questi orecchini offrono una soluzione creativa e elegante per esprimere la propria individualita."

    def Luna_cocktail_ring(self):
        return "L anello Luna cattura la magia della luna con una cupola in pavé di cristalli Cry. Il design placcato rodio, elegante e affascinante, si presenta come un elegante mezzaluna. La sua versatilita lo rende adatto a diverse fasi del tuo stile, illuminando con riflessi ogni outfit. Un gioiello che trasforma ogni gesto in un affermazione di eleganza, l anello Luna e un compagno luminoso per ogni occasione."

    def Angelic_necklace(self):
        return "Il raffinato collier Angelic Swarovski si abbina perfettamente ad altre creazioni della linea. Ideale per occasioni speciali, e realizzato in Clear Crystal e metallo rodiato lucido. Con un design senza tempo, questo collier e un regalo prezioso e un elegante dichiarazione di stile, aggiungendo un tocco di grazia e brillantezza a qualsiasi look."

    def Florere_cocktail_ring(self):
        return "Lo sfavillante anello Florere con motivo floreale e un esempio brillante del savoir-faire di Swarovski. Il design a doppio gambo, placcato in tonalita oro, si adatta confortevolmente al dito, con petali straordinari impreziositi da cristalli rosa di varie dimensioni. Pensato per essere il protagonista della scena, questo anello aggiunge una nota affascinante a qualsiasi look, offrendo una combinazione unica di eleganza e stile floreale."

    def Stilla_necklace(self):
        return "Indossa la collana Stilla per sublimare il tuo stile con gocce di colore brillante. Realizzata con una montatura placcata oro e una chiusura a moschettone, la collana presenta tagli di Swarovski Zirconia in sette colori complementari. Ogni pietra e circondata da cristalli quadrati piu piccoli, creando una collana che continua a brillare. Combina questa eclettica collana con eleganti orecchini della stessa famiglia o regalala a una persona cara per un tocco di eleganza vibrante."

    def Matrix_tennis_necklace(self):
        return "Risplendi con la collana Matrix Tennis, caratterizzata da una fila di brillanti pietre rotonde gialle. Ogni pietra e incastonata in una catena placcata oro con una chiusura sicura e raffinata. Questo gioiello diventera un accessorio essenziale per il tuo look quotidiano, aggiungendo una nota di lusso alla tua collezione con la sua brillante cascata di colori."

    def Creativity_pendant(self):
        return "Il pendente Creativity e una rivisitazione contemporanea del classico pendente tondo. Realizzato con pavé di Clear Crystal e metallo rodiato, aggiunge una nota moderna e si indossa splendidamente sovrapposto con un pendente piu lungo. Con le sue dimensioni di 1,5 x 2 cm, e un aggiunta elegante e versatile alla tua collezione di gioielli, offrendo un tocco di brillantezza e stile contemporaneo."

    def Gema_bracelet(self):
        return "Il bracciale Gema con sfumature blu ghiaccio e un aggiunta straordinaria a qualsiasi collezione di gioielli. Con un design placcato rodio che gioca con tagli e colori ispirati allo stile bohémien, questo bracciale e un opera d arte lussuosa e dal tocco disinvolto. Indossalo da solo per un glamour rilassato o sovrapponilo ad altri gioielli della stessa famiglia per un look speciale, offrendo un mix irresistibile di eleganza e originalita."

    def Hyperbola_drop_earrings(self):
        return "Gli orecchini Hyperbola sono il mezzo perfetto per aggiungere un tocco passionale al tuo stile. Placcati in rodio, presentano un design tridimensionale con Swarovski Zirconia trasparenti e una pietra danzante blu a forma di cuore. Abbinati a un pendente coordinato, enfatizzano il glamour e l originalita, offrendo una dichiarazione audace di stile e personalita."

    def Lucent_bangle(self):
        return "Il bracciale rigido ottagonale Lucent con cristallo verde acceso e un aggiunta elegante a qualsiasi outfit. La montatura rivestita in PVD oro e la chiusura magnetica innovativa lo rendono un accessorio versatile. Parte della linea Lucent, questo bracciale rigido e un opzione moderna e chic per la Collection III, offrendo un mix di stile geometrico e brillantezza vivace."

    def Lucent_cocktail_ring(self):
        return "L anello Lucent, ispirato all iconico Swarovski Nirvana, e un capolavoro d impatto con una tonalita verde intensa. Con oltre 138 sfaccettature e una fascia metallica coordinata, puo essere indossato da solo o in abbinamento ad altri anelli per un effetto esplosivo di colore. Firmato dalla Direttrice Creativa Giovanna Engelbert, e parte della famiglia Lucent e della Collection I, offrendo un elegante dichiarazione di stile con una cascata di colore vibrante."

    def Millenia_necklace(self):
        return "La vibrante collana Millenia e formata da cristalli taglio Octagon verdi, montati su una catena placcata oro distintiva. Aggiungi una cascata prismatica di colori al tuo look quotidiano, indossando questo gioiello su sfondi monocromatici. Firmata dalla Direttrice Creativa Giovanna Engelbert, fa parte della famiglia Millenia e della Collection I, offrendo un tocco distintivo di brillantezza e stile audace."

    def Disney_mickey_mouse_pendant(self):
        return "In celebrazione dei 100 anni della Disney, il pendente placcato rodio raffigura Topolino aggrappato alla catena con il suo sorriso iconico. Un gioiello tridimensionale perfetto per chi ama l avventura e conserva il cuore di un bambino, offrendo un tocco giocoso e nostalgico di magia Disney."

    def Crystalline_delight_watch(self):
        return "Chic e facile da indossare, l orologio Crystalline Delight di Swarovski brilla con circa 1.000 Clear Crystal. Il quadrante moderno, la cassa in acciaio inossidabile e il bracciale ispirato ai gioielli creano un accessorio elegante e resistente all acqua fino a 50 metri. Prodotto in Svizzera, e una perfetta fusione di stile e funzionalita, offrendo un tocco di lusso e brillantezza in ogni momento."

    def Gema_necklace(self):
        return "La collana Gema offre uno stile unico con una varieta di tagli e colori vivaci. Realizzata con una catena placcata rodio, presenta cristalli di dimensioni e tagli straordinari, creando un affascinante armonia. Aggiungi gioia prismatica al tuo look indossando questa collana e abbinala ad altri gioielli Gema per un look irresistibile, offrendo un mix di colore vivace e stile distintivo."

    def analizza_genere(self, frase):
        parole_chiave_maschili  = ["zio","marito","maschio","fratello", "papà", "padre", "nonno", "amico","figlio", "fratello,", "papà,", "padre,", "nonno,", "amico,","figlio,", "fratello.", "papà.", "padre.", "nonno.", "amico.","figlio."]
        parole_chiave_femminili = ["zia","moglie","femmina","sorella", "mamma", "madre", "nonna", "amica", "sorella,", "mamma,", "madre,", "nonna,", "amica,", "sorella.", "mamma.", "madre.", "nonna.", "amica.","figlia","figlia,","figlia."]

        parole = frase.lower().split() 
        
        genere_maschile = any(parola in parole for parola in parole_chiave_maschili)
        genere_femminile = any(parola in parole for parola in parole_chiave_femminili)
        
        if genere_maschile and not genere_femminile:
            return "male"
        elif genere_femminile and not genere_maschile:
            return "female"
        else:
            return ""

    def estrai_eta(self, frase):
        risposta = frase
        matches = re.findall(r'\b\d+\b', risposta)
        lista_numeri = len(matches)
        trovato = False
        for i in range(lista_numeri):
            if int(matches[i])<100:
                trovato = True
                return int(matches[i])
        if not trovato:
            return 0

    def estrai_budget(self, frase):
        risposta = frase
        matches = re.findall(r'\b\d+\b', risposta)
        lista_numeri = len(matches)
        trovato = False
        for i in range(lista_numeri):
            if int(matches[i])>=100:
                trovato = True
                return int(matches[i])
        if not trovato:
            return 0

    def estrai_categoria(self, frase):
        parole_chiave_necklace  = ["collana", "collane", "collana,", "collane,", "collana.", "collane."]
        parole_chiave_ring      = ["anello", "anelli", "anello,","anelli,", "anello.","anelli."]
        parole_chiave_bracelet  = ["bracciale", "bracciali", "braccialetto", "braccialetti", "bracciale,", "bracciali,", "braccialetto,", "braccialetti,", "bracciale.","bracciali.", "braccialetto.", "braccialetti."]               
        parole_chiave_earring   = ["orecchino", "orecchini", "orecchino,","orecchini,", "orecchino.", "orecchini."]
        parole_chiave_watch     = ["orologio", "orologio,", "orologio.", "orologi", "orologi,", "orologi."]
        
        category                = ["necklace","ring","bracelet","earrings"]
        parola_chiave           = ["uguale","indifferente"]
        
        parole = frase.lower().split()  

        if any(parola in parole for parola in parole_chiave_necklace):             
            return "necklace"
        elif any(parola in parole for parola in parole_chiave_ring):
            return "ring"
        elif any(parola in parole for parola in parole_chiave_bracelet):
            return "bracelet"
        elif any(parola in parole for parola in parole_chiave_earring):
            return "earrings"
        elif any(parola in parole for parola in parole_chiave_watch):
            return "watch"
        
        elif any(parola in parole for parola in parola_chiave):
            random_category = random.choice(category)
            return random_category
        else:
            return ""

    def estrai_regalo(self, frase):
        parole = frase.lower().split()
        parole_chiave_regalo = ["regalo","regalo,","regalo."]
        parole_chiave_me = ["me"]
        
        if any(parola in parole for parola in parole_chiave_regalo):
            return "regalo"
        elif any(parola in parole for parola in parole_chiave_me):
            return "me"
        else:
            return ""


    #decision tree
    def recommend_jewelry(self, customer_info, product_info):
        gender      = customer_info[0]
        age         = customer_info[1]
        budget      = customer_info[2]
        category    = customer_info[3]

        '''
        name = primo_dizionario[list(primo_dizionario.keys())[1]]
        category = primo_dizionario[list(primo_dizionario.keys())[2]]
        storage = primo_dizionario[list(primo_dizionario.keys())[3]]
        '''
        
        gioielli_consigliati = []
        
        if gender == 'male': 
            if age > 60:
                return []
            elif age < 20: 
                if category == 'bracelet':
                    if budget >= 155:
                        for item in product_info:
                            if item['gender'] == 'M' and item['age'] < 20 and item['category'] == 'bracelet' and item['prezzo'] >= 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati 
                    else:
                        return []
                elif category == 'necklace':
                    if budget < 175:
                        
                        return []
                    else:
                        for item in product_info:
                            if item['gender'] == 'M' and item['age'] < 20 and item['category'] == 'necklace' and item['prezzo'] >= 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                else:
                    return []
            elif 20 <= age <= 60:
                if category == 'bracelet':
                    if budget < 145:   
                        return []
                    elif 145 <= budget < 155:
                        for item in product_info:
                            if item['gender'] == 'M' and 20 <= item['age'] <= 60 and item['category'] == 'bracelet' and 145 <= item['prezzo'] < 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else: 
                        for item in product_info:
                            if item['gender'] == 'M' and 20 <= item['age'] <= 60 and item['category'] == 'bracelet' and item['prezzo'] >= 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'necklace':
                    if budget < 175:
                        return []
                    else:
                        for item in product_info:
                            if item['gender'] == 'M' and 20 <= item['age'] <= 60 and item['category'] == 'necklace' and item['prezzo'] >= 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati   
                else:
                    return []
        
        elif gender == 'female':
            if 0 < age < 20:
                if category == 'bracelet':
                    if budget >= 155:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'bracelet' and item['prezzo'] >= 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        return []
                elif category == 'necklace':
                    if budget < 125:
                        return []
                    elif 125 <= budget < 155:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'necklace' and 125 <= item['prezzo'] < 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 155 <= budget < 175:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'necklace' and 155 <= item['prezzo'] < 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 175 <= budget < 230:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'necklace' and 175 <= item['prezzo'] < 230:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif budget >= 230:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'necklace' and item['prezzo'] >= 230:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'earrings':
                    if budget < 125:
                        return []
                    elif 125 <= budget < 135:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'earrings' and 125 <= item['prezzo'] < 135:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 135 <= budget < 155:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'earrings' and 135 <= item['prezzo'] < 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 0 < item['age'] < 20 and item['category'] == 'earrings' and item['prezzo'] >= 155:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                else:
                    return []
            elif 20 <= age < 40:
                if category == 'bracelet':
                    if budget < 195:
                        return []
                    elif 195 <= budget < 215:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'bracelet' and 195 <= item['prezzo'] < 215:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'bracelet' and item['prezzo'] >= 215:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati    
                elif category == 'necklace':
                    if budget < 75:
                        return []
                    elif 75 <= budget < 115:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'necklace' and 75 <= item['prezzo'] < 115:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 115 <= budget < 175:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'necklace' and 115 <= item['prezzo'] < 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 175 <= budget < 195:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'necklace' and 175 <= item['prezzo'] < 195:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'necklace' and item['prezzo'] >= 195:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'earrings':
                    if budget < 89:
                        return []
                    elif 89 <= budget < 125:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'earrings' and 89 <= item['prezzo'] < 125:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 125 <= budget < 129:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'earrings' and 125 <= item['prezzo'] < 129:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 129 <= budget < 195:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'earrings' and 129 <= item['prezzo'] < 195:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 195 <= budget < 300:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'earrings' and 195 <= item['prezzo'] < 300:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'earrings' and item['prezzo'] > 300:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'watch':
                    if budget >= 300:
                        for item in product_info:
                            if item['gender'] == 'F' and 20 <= item['age'] < 40 and item['category'] == 'watch' and item['prezzo'] >= 300:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        return []
                else:
                    return []
            elif 40 <= age < 60:
                if category == 'necklace':
                    if budget < 175:
                        return []
                    elif 175 <= budget < 250:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'necklace' and 175 <= item['prezzo'] < 250:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 250 <= budget < 550:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'necklace' and 250 <= item['prezzo'] < 550:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'necklace' and item['prezzo'] >= 550:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'earrings':
                    if budget < 95:
                        return []
                    elif 95 <= budget < 175:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'earrings' and 95 <= item['prezzo'] < 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 175 <= budget < 195:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'earrings' and 175 <= item['prezzo'] < 195:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 195 <= budget < 230:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'earrings' and 195 <= item['prezzo'] < 230:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'earrings' and item['prezzo'] >= 230:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'ring':
                    if budget < 75:
                        return []
                    elif 75 <= budget < 125:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and 75 <= item['prezzo'] < 125:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 125 <= budget < 135:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and 125 <= item['prezzo'] < 135:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 135 <= budget < 175:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and 135 <= item['prezzo'] < 175:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 175 <= budget < 250:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and 175 <= item['prezzo'] < 250:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    elif 250 <= budget < 400:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and 250 <= item['prezzo'] < 400:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and 40 <= item['age'] < 60 and item['category'] == 'ring' and item['prezzo'] > 400:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                else:
                    return []
            elif age >= 60:
                if category == 'bracelet':
                    if budget < 400:
                        return []
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and item['age'] >= 60 and item['category'] == 'bracelet' and item['prezzo'] >= 400:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'necklace':
                    if budget < 950:
                        return []
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and item['age'] >= 60 and item['category'] == 'necklace' and item['prezzo'] >= 950:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati
                elif category == 'ring':
                    if budget < 135:
                        return []
                    else:
                        for item in product_info:
                            if item['gender'] == 'F' and item['age'] >= 60 and item['category'] == 'ring' and item['prezzo'] >= 135:
                                gioielli_consigliati.append(item['id'])
                        return gioielli_consigliati 
                else:
                    return []

    def get_product_name_by_id(self, product_info, target_product_id):
        product = next((product for product in product_info if product['id'] == target_product_id), None)
        if product:
            return product['name']

    def Descrizione_prodotto(self, funzione_prodotto):
        # Verifica se il metodo specificato è presente nell'istanza corrente della classe ed è chiamabile
        if hasattr(self, funzione_prodotto) and callable(getattr(self, funzione_prodotto)):
            # Chiama il metodo specificato e memorizza il risultato
            result = getattr(self, funzione_prodotto)()
            # Restituisce il risultato del metodo
            return result

    def estrai_si_no(self, frase):
        parole_chiave_no = ["no", "non mi piace", "non mi ispira", "no grazie","a posto cosi"] 
        parole_chiave_si = ["si", "mi piace", "si grazie"]      
        frase = frase.lower()
        
        if any(parola in frase for parola in parole_chiave_no):
            return "no"
        elif any(parola in frase for parola in parole_chiave_si):
            return "si"
        else:
            return ""

    def funzione_abbinamento(self, product_name):
        abbinamenti = {}
        product = self.db_helper.dt_get_oggetto()
        data    = self.db_helper.get_abbinamento()
        for item in data:
            for p in product:
                if item['id_oggetto1'] == p['id']:
                    key = p['name']
                    break
            for p in product:
                if item['id_oggetto2'] == p['id']:
                    value = p['name']
                    break
            abbinamenti[key] = value
        return abbinamenti.get(product_name, None)

    def morphcast(self, emozioni, id_cliente, id_oggetto):
        angry     = int(emozioni['emo']['Angry'] * 100)
        disgust   = int(emozioni['emo']['Disgust'] * 100)
        happy     = int(emozioni['emo']['Happy'] * 100)
        sad       = int(emozioni['emo']['Sad'] * 100)
        surprise  = int(emozioni['emo']['Surprise'] * 100)
        attention = int(emozioni['att'] * 100)
        indice_gradimento = 50 - angry - disgust + happy - sad + surprise + attention

        nao_animatedSayText("In questo momento sei: " + str(emozioni['emo_dom']) + ". Il tuo indice di gradimento vale " + str(indice_gradimento))
        
        # salva indice di gradimento nel db
        data = self.db_helper.set_emozione(id_cliente, id_oggetto, emozioni['age'], emozioni['gen'], indice_gradimento)

        if indice_gradimento > 40:
            return "si"
        else:
            return "no"

    def main(self, id_cliente, nome, cognome, id_carrello, backup_nao):
        product_info = self.db_helper.dt_get_oggetto()

        #dialogo
        text_to_say = "Buongiorno " + str(nome) + " " + str(cognome) + ", sono peara, il nao del team naonecsus."
        nao_animatedSayText(text_to_say)
        self.dialog.append({"nao":text_to_say})
        print(text_to_say)

        text_to_say = "Come posso aiutarti? Anche se sono un robot di gioielli ne so un bel po"
        nao_animatedSayText(text_to_say)
        self.dialog.append({"nao":text_to_say})
        print(text_to_say)

        if backup_nao:
            risposta1 = raw_input()
        else:
            risposta1 = nao_speech_to_text(5)
        self.dialog.append({"utente":risposta1})
        print(risposta1)
        regalo = self.estrai_regalo(risposta1)

        if backup_nao:
            emozioni = {'data': {'det': 1, 'gen': 'female', 'age': 50, 'emo': {'Angry': 0.2, 'Disgust': 0.3, 'Happy': 0.9, 'Neutral': 0.5, 'Sad': 0.1, 'Surprise': 0.6}, 'att': 0.8, 'emo_dom':'Happy'} }
        else:
            #get api_morphcast
            url = "http://127.0.0.1:5010/api/morphcast"
            response = requests.get(url)
            emozioni = response.json()
        emozioni = emozioni['data']
    
        if regalo == "regalo":
            gender   = str(self.analizza_genere(risposta1))
            age      = int(self.estrai_eta(risposta1))
            prezzo   = int(self.estrai_budget(risposta1))
            category = str(self.estrai_categoria(risposta1))
        else:
            gender   = emozioni['gen']
            age      = emozioni['age']
            prezzo   = int(self.estrai_budget(risposta1))
            category = str(self.estrai_categoria(risposta1))

        profilo_utente  = [gender, age, prezzo, category]
        posizioni_vuote = [pos for pos, val in enumerate(profilo_utente) if val == "" or val == 0]
        
        user_input = True
        while user_input == True:
            while len(posizioni_vuote) != 0:
                for i in range(len(posizioni_vuote)):
                    if posizioni_vuote[i] == 0:
                        text_to_say = "posso chiederti per chi è il gioiello?"
                        nao_animatedSayText(text_to_say)
                        self.dialog.append({"nao":text_to_say})
                        print(text_to_say)
                        if backup_nao:
                            risposta2 = raw_input()
                        else:
                            risposta2 = nao_speech_to_text(5)
                        self.dialog.append({"utente":risposta2})
                        print(risposta2)
                        gender = str(self.analizza_genere(risposta2))          
                    elif posizioni_vuote[i] == 1:
                        text_to_say = "E quanti anni ha?"
                        nao_animatedSayText(text_to_say)
                        self.dialog.append({"nao":text_to_say})
                        print(text_to_say)
                        if backup_nao:
                            risposta3 = raw_input()
                        else:
                            risposta3 = nao_speech_to_text(5)
                        self.dialog.append({"utente":risposta3})
                        print(risposta3)
                        age = int(self.estrai_eta(risposta3))
                    elif posizioni_vuote[i] == 2:
                        text_to_say = "Hai un bagget per lacquisto del gioiello?"
                        nao_animatedSayText(text_to_say)
                        self.dialog.append({"nao":text_to_say})
                        print(text_to_say)
                        if backup_nao:
                            risposta4 = raw_input()
                        else:
                            risposta4 = nao_speech_to_text(5)

                        # controllo presenza simboli
                        simboli_valute = {
                                            "USD": "$",     # Dollaro statunitense
                                            "EUR": "€",     # Euro
                                            "GBP": "£",     # Sterlina britannica
                                            "JPY": "¥",     # Yen giapponese
                                            "CHF": "CHF",   # Franco svizzero
                                            "CAD": "$",     # Dollaro canadese
                                            "AUD": "$",     # Dollaro australiano
                                            "CNY": "¥",     # Yuan cinese
                                            "RUB": "₽",     # Rublo russo
                                            "INR": "₹"      # Rupia indiana
                                        }
                        for abbreviazione, simbolo in simboli_valute.items():
                            risposta4 = risposta4.replace(simbolo, abbreviazione)
                        
                        self.dialog.append({"utente":risposta4})
                        print(risposta4)
                        prezzo = int(self.estrai_budget(risposta4))
                    elif posizioni_vuote[i] == 3:
                        text_to_say = "A quale tipologia di gioielli pensavi? Collane, bracciali, orecchini, anelli o orologi?"
                        nao_animatedSayText(text_to_say)
                        self.dialog.append({"nao":text_to_say})
                        print(text_to_say)
                        if backup_nao:
                            risposta5 = raw_input()
                        else:
                            risposta5 = nao_speech_to_text(5)
                        self.dialog.append({"utente":risposta5})
                        print(risposta5)
                        category = str(self.estrai_categoria(risposta5))
                            
                profilo_utente  = [gender, age, prezzo, category]
                posizioni_vuote = [pos for pos, val in enumerate(profilo_utente) if val == "" or val == 0]
    
            id_gioiello_consigliato = self.recommend_jewelry(profilo_utente, product_info)
            if len(id_gioiello_consigliato) > 0:
                i = 0
                while i < len(id_gioiello_consigliato):
                    product_name = self.get_product_name_by_id(product_info, id_gioiello_consigliato[i])

                    #post id_gioiello_consigliato
                    #DA TESTARE
                    data = {"id_oggetto": id_gioiello_consigliato}
                    url = "http://127.0.0.1:5010/api/post_gioiello_consigliato" + str(data)
                    response = requests.post(url, json=data)

                    
                    text_to_say = "Fammi pensare... io ti consiglierei: " + str(product_name)
                    nao_animatedSayText(text_to_say)
                    self.dialog.append({"nao":text_to_say})
                    print(text_to_say)
                    funzione_prodotto = product_name.capitalize().replace(' ', '_')
                    descrizione = self.Descrizione_prodotto(funzione_prodotto)
                    text_to_say = str(descrizione)
                    nao_animatedSayText(text_to_say)
                    self.dialog.append({"nao":text_to_say})
                    print(text_to_say)

                    if backup_nao == False:
                        #get api_morphcast
                        url = "http://127.0.0.1:5010/api/morphcast"
                        response = requests.get(url)
                        emozioni = response.json()
                        emozioni = emozioni['data']

                    risposta_a1_ = self.morphcast(emozioni, id_cliente, id_gioiello_consigliato[i])
                    
                    if risposta_a1_ == "si":
                        for a in product_info:
                            if a['name']== product_name:
                                magazzino=a['qta_magazzino']
                                text_to_say = "vuoi aggiungere " + str(product_name) + " al carrello?"
                                nao_animatedSayText(text_to_say)
                                self.dialog.append({"nao":text_to_say})
                                print(text_to_say)
                                if backup_nao:
                                    r1 = raw_input()
                                else:
                                    r1 = nao_speech_to_text(5)
                                self.dialog.append({"utente":r1})
                                print(r1)
                                aggiungi = self.estrai_si_no(r1)
                                if aggiungi == "si":
                                    if magazzino > 0:
                                        #aggiungo oggetto al carrello id_carrello
                                        data = self.db_helper.set_carrellooggetto(id_carrello['id'], id_gioiello_consigliato[i])

                                        for b in product_info:
                                            if b['name']== product_name:
                                                if b['sconto']>0:
                                                    prezzo_scontato = b['prezzo'] - ( b['prezzo'] * b['sconto'] /100 )
                                                    text_to_say = "Il prodotto scelto è in sconto e da " + str(b['prezzo']) + " viene " + str(prezzo_scontato)
                                                    nao_animatedSayText(text_to_say)
                                                    self.dialog.append({"nao":text_to_say})
                                                    print(text_to_say)
                                    else:
                                        text_to_say = "Abbiamo terminato il prodotto richiesto, prova ad andare in un altro store o ripassa settimana prossima"
                                        nao_animatedSayText(text_to_say)
                                        self.dialog.append({"nao":text_to_say})
                                        print(text_to_say)
                        if aggiungi == "si":
                            text_to_say = "Vuoi che ti consigli qualche abbinamento da fare?"
                            nao_animatedSayText(text_to_say)
                            self.dialog.append({"nao":text_to_say})
                            print(text_to_say)
                            if backup_nao:
                                risposta_ = raw_input()
                            else:
                                risposta_ = nao_speech_to_text(5)
                            self.dialog.append({"utente":risposta_})
                            print(risposta_)
                            risposta__= self.estrai_si_no(risposta_)
                        else:
                            risposta__= "no"
                            
                        if risposta__ == "si":
                            abbinamento = self.funzione_abbinamento(product_name)
                            id_abbinamento = self.db_helper.get_titolo_oggetto(abbinamento)

                            #post id_abbinamento
                            #DA TESTARE
                            data = {"id_oggetto": id_abbinamento['id']}
                            url = "http://127.0.0.1:5010/api/post_gioiello_consigliato" + str(data)
                            response = requests.post(url, json=data)

                            
                            text_to_say = "Perfetto, allora ti consiglierei di abbinare " + str(abbinamento) + " con " + str(product_name)
                            nao_animatedSayText(text_to_say)
                            self.dialog.append({"nao":text_to_say})
                            print(text_to_say)
                            funzione_prodotto2 = abbinamento.capitalize().replace(' ', '_')
                            descrizione2 = self.Descrizione_prodotto(funzione_prodotto2)
                            text_to_say = str(descrizione2)
                            nao_animatedSayText(text_to_say)
                            self.dialog.append({"nao":text_to_say})
                            print(text_to_say)

                            for b in product_info:
                                if b['name'] == abbinamento:
                                    magazzino_ = b['qta_magazzino']
                                    text_to_say = "vuoi aggiungere " + str(abbinamento) + " al carrello?"
                                    nao_animatedSayText(text_to_say)
                                    self.dialog.append({"nao":text_to_say})
                                    print(text_to_say)
                                    if backup_nao:
                                        r2 = raw_input()
                                    else:
                                        r2 = nao_speech_to_text(5)
                                    self.dialog.append({"utente":r2})
                                    print(r2)
                                    aggiungi_ = self.estrai_si_no(r2)
                                    if aggiungi_ == "si":
                                        if magazzino_ > 0:
                                            #aggiungo oggetto al carrello id_carrello
                                            id_abbinamento = self.db_helper.get_titolo_oggetto(abbinamento)
                                            data = self.db_helper.set_carrellooggetto(id_carrello['id'], id_abbinamento['id'])

                                            for b in product_info:
                                                if b['name'] == product_name:
                                                    if b['sconto'] > 0:
                                                        prezzo_scontato = b['prezzo'] - ( b['prezzo'] * b['sconto'] /100 )
                                                        text_to_say = "Il prodotto scelto è in sconto e da " + str(b['prezzo']) + " viene " + str(prezzo_scontato)
                                                        nao_animatedSayText(text_to_say)
                                                        self.dialog.append({"nao":text_to_say})
                                                        print(text_to_say)
                                        else:
                                            text_to_say = "Abbiamo terminato il prodotto richiesto, prova ad andare in un altro store o ripassa settimana prossima"
                                            nao_animatedSayText(text_to_say)
                                            self.dialog.append({"nao":text_to_say})
                                            print(text_to_say)
                                    else:
                                        text_to_say = "ok"
                                        nao_animatedSayText(text_to_say)
                                        self.dialog.append({"nao":text_to_say})
                                        print(text_to_say)
                        
                            if backup_nao == False:
                                #get api_morphcast
                                url = "http://127.0.0.1:5010/api/morphcast"
                                response = requests.get(url)
                                emozioni = response.json()
                                emozioni = emozioni['data']
                            risposta3 = self.morphcast(emozioni, id_cliente, id_abbinamento['id'])

                            if risposta3 == "si":
                                user_input = False
                                break
                            else:
                                text_to_say = "Ok mi dispiace"
                                nao_animatedSayText(text_to_say)
                                self.dialog.append({"nao":text_to_say})
                                print(text_to_say)
                                user_input = False
                                break            
                        else:
                            break
                    i += 1
                    if i == len(id_gioiello_consigliato):
                        text_to_say = "Faccio fatica a trovare un prodotto adatto a te, mi dispiace"
                        nao_animatedSayText(text_to_say)
                        self.dialog.append({"nao":text_to_say})
                        print(text_to_say)
                        user_input=False
                        break     
            else:
                text_to_say = "Faccio fatica a trovare un prodotto adatto a te, mi dispiace"
                nao_animatedSayText(text_to_say)
                self.dialog.append({"nao":text_to_say})
                print(text_to_say)
                user_input = False
                break

            text_to_say = "Vuoi acquistare qualche altro prodotto?"
            nao_animatedSayText(text_to_say)
            self.dialog.append({"nao":text_to_say})
            print(text_to_say)
            if backup_nao:
                user_input = raw_input()
            else:
                user_input = nao_speech_to_text(5)
            self.dialog.append({"utente":user_input})
            print(user_input)
            user_input_1 = self.estrai_si_no(user_input)

            if user_input_1 == "no":
                text_to_say = "Controlla e conferma il tuo ordine nell'app"
                nao_animatedSayText(text_to_say)
                self.dialog.append({"nao":text_to_say})
                print(text_to_say)
                text_to_say = "i tuoi gioielli ti aspettano in cassa"
                nao_animatedSayText(text_to_say)
                self.dialog.append({"nao":text_to_say})
                print(text_to_say)          
                user_input=False
                break
            else:
                text_to_say = "posso chiederti per chi è il gioiello?"
                nao_animatedSayText(text_to_say)
                self.dialog.append({"nao":text_to_say})
                print(text_to_say)
                if backup_nao:
                    risposta1_ = raw_input()
                else:
                    risposta1_ = nao_speech_to_text(5)
                self.dialog.append({"utente":risposta1_})
                print(risposta1_)
                regalo = self.estrai_regalo(risposta1_)
                if regalo == "me":
                    gender  = emozioni['gen']
                    age     = emozioni['age']
                    prezzo  = int(self.estrai_budget(risposta1_))
                    category= str(self.estrai_categoria(risposta1_))
                    profilo_utente = [gender, age, prezzo, category]
                    posizioni_vuote = [pos for pos, val in enumerate(profilo_utente) if val == "" or val == 0]
                    user_input = True
                else:
                    gender   = str(self.analizza_genere(risposta1_))
                    age      = int(self.estrai_eta(risposta1_))
                    prezzo   = int(self.estrai_budget(risposta1_))
                    category = str(self.estrai_categoria(risposta1_))
                    profilo_utente = [gender, age, prezzo, category]
                    posizioni_vuote = [pos for pos, val in enumerate(profilo_utente) if val == "" or val == 0]
                    user_input = True

        text_to_say = "Spero di averti aiutato al meglio delle mie possibilita, grazie per aver acquistato da svaroschi utilizzando la tecnologia naonecsus!"
        nao_animatedSayText(text_to_say)
        self.dialog.append({"nao":text_to_say})
        print(text_to_say)


    def __init__(self, db_helper, id_cliente, nome, cognome, id_carrello, backup_nao):
        self.db_helper = db_helper
        self.dialog = []
        self.main(id_cliente, nome, cognome, id_carrello, backup_nao)