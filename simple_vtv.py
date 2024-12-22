import gradio as gr
import assemblyai as aai
from deep_translator import GoogleTranslator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def voice_to_voice(audio_file):
    
    #transcribe audio
    transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    es_translation, de_translation, zh_translation = text_translation(text)

    es_audio_path = text_to_speech(es_translation)
    de_audio_path = text_to_speech(de_translation)
    zh_audio_path = text_to_speech(zh_translation)

    es_path = Path(es_audio_path)
    de_path = Path(de_audio_path)
    zh_path = Path(zh_audio_path)

    return es_path, de_path, zh_path


def audio_transcription(audio_file):

    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)

    return transcription

def text_translation(text):
    translator = GoogleTranslator()
    
    es_text = GoogleTranslator(source='en', target='es').translate(text)
    de_text = GoogleTranslator(source='en', target='de').translate(text)
    zh_text = GoogleTranslator(source='en', target='zh-CN').translate(text)
    
    return es_text, de_text, zh_text

def text_to_speech(text):

    client = ElevenLabs(
        api_key=os.getenv('ELEVENLABS_API_KEY'),
    )

    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id=os.getenv('ELEVENLABS_VOICE_ID'),
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2", # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Spanish"), gr.Audio(label="German"), gr.Audio(label="Mandarin")]
)

if __name__ == "__main__":
    demo.launch()