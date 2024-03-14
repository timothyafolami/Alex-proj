from dotevn import load_dotenv
from openai import OpenAI
import os

load_dotenv()

OpenAI_key = os.getenv("opena_api_key")

client = OpenAI(api_key=OpenAI_key)

audio_file = open("test.m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text"
)
# import gradio as gr


