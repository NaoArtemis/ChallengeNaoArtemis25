from gtts import gTTS

# Definisci il testo in italiano
testo = "Gli orecchini Hyperbola sono il mezzo perfetto per aggiungere un tocco passionale al tuo stile. Placcati in rodio, presentano un design tridimensionale, trasparenti e una pietra danzante blu a forma di cuore. Abbinati a un pendente coordinato, enfatizzano il glamour e l'originalità, offrendo una dichiarazione audace di stile e personalità."

# Crea un oggetto gTTS con la lingua italiana
tts = gTTS(text=testo, lang='it')

# Salva l'audio convertito in un file
tts.save("output_italiano.mp3")
