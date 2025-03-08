from pathlib import Path
from openai import OpenAI

testo = "Gli orecchini Hyperbola sono il mezzo perfetto per aggiungere un tocco passionale al tuo stile. Placcati in rodio, presentano un design tridimensionale, trasparenti e una pietra danzante blu a forma di cuore. Abbinati a un pendente coordinato, enfatizzano il glamour e l'originalità, offrendo una dichiarazione audace di stile e personalità."

client = OpenAI(api_key="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")

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=testo
)

response.stream_to_file(speech_file_path)