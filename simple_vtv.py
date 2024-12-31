import gradio as gr
import assemblyai as aai
from deep_translator import GoogleTranslator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def voice_to_voice(audio_file):
    """
    Main function that handles the complete voice-to-voice translation pipeline.
    Takes an audio file and returns four translated audio files in Spanish, German, Mandarin, and Greek.
    """
    
    # Step 1: Convert input audio to text using AssemblyAI
    transcription_response = audio_transcription(audio_file)

    # Handle transcription errors
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    # Step 2: Translate the transcribed text to multiple languages
    es_translation, de_translation, zh_translation, el_translation = text_translation(text)

    # Step 3: Convert translated text back to speech using ElevenLabs
    es_audio_path = text_to_speech(es_translation)
    de_audio_path = text_to_speech(de_translation)
    zh_audio_path = text_to_speech(zh_translation)
    el_audio_path = text_to_speech(el_translation)  # Greek audio

    # Convert file paths to Path objects for Gradio
    es_path = Path(es_audio_path)
    de_path = Path(de_audio_path)
    zh_path = Path(zh_audio_path)
    el_path = Path(el_audio_path)

    return es_path, de_path, zh_path, el_path


def audio_transcription(audio_file):
    """
    Transcribes audio file to text using AssemblyAI's API.
    Returns the transcription response object.
    """
    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)

    return transcription

def text_translation(text):
    """
    Translates input text to Spanish, German, Mandarin, and Greek using Google Translate.
    Returns tuple of translated texts.
    """
    es_text = GoogleTranslator(source='en', target='es').translate(text)
    de_text = GoogleTranslator(source='en', target='de').translate(text)
    zh_text = GoogleTranslator(source='en', target='zh-CN').translate(text)
    el_text = GoogleTranslator(source='en', target='el').translate(text)  # Greek translation
    
    return es_text, de_text, zh_text, el_text

def text_to_speech(text):
    """
    Converts text to speech using ElevenLabs API.
    Returns the path to the generated audio file.
    """
    client = ElevenLabs(
        api_key=os.getenv('ELEVENLABS_API_KEY'),
    )

    # Generate speech with specified parameters
    response = client.text_to_speech.convert(
        voice_id=os.getenv('ELEVENLABS_VOICE_ID'),
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",  # Multilingual model for better language support
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    # Generate unique filename for the audio file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Save the audio stream to file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    return save_file_path

# Create Gradio UI components
audio_input = gr.Audio(
    sources=["microphone"],  # Allow microphone input only
    type="filepath"
)

# Define the Gradio interface
demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Spanish"), 
        gr.Audio(label="German"), 
        gr.Audio(label="Mandarin"),
        gr.Audio(label="Greek")  # Add Greek output
    ]
)

if __name__ == "__main__":
    demo.launch()