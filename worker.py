from openai import OpenAI
import requests
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import torch
import soundfile as sf
import numpy as np
import io
import numpy as np

openai_client = OpenAI()


# Initialize Whisper model and processor (this should be done once, possibly as global variables)
stt_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Initialize TTS model and processor
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

def speech_to_text(audio_binary):
    try:
        # Convert audio binary to input features
        input_features = stt_processor(
            audio_binary, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Generate token ids
        predicted_ids = stt_model.generate(input_features)
        
        # Decode the token ids to text
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print('recognised text: ', transcription)
        return transcription
        
    except Exception as e:
        print(f"Error in speech to text conversion: {str(e)}")
        return "null"



def text_to_speech(text, voice=""):
    try:
        # Process text input
        inputs = tts_processor(text=text, return_tensors="pt")
        
        # Generate speech
        speech = tts_model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings=None,
        )
        
        # Convert to bytes
        speech_np = speech.numpy()
        bytes_io = io.BytesIO()
        sf.write(bytes_io, speech_np, 16000, format='wav')
        audio_bytes = bytes_io.getvalue()
        
        return audio_bytes
        
    except Exception as e:
        print(f"Error in text to speech conversion: {str(e)}")
        return None


def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=4000
    )
    print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text

def process_message(user_message):
    try:
        # Add system prompt and user message
        prompt = "Assistant: I am a helpful AI assistant. I can answer questions, translate sentences, summarize news, and give recommendations.\nUser: " + user_message + "\nAssistant:"
        
        # Encode the input
        inputs = gpt_tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        outputs = gpt_model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            no_repeat_ngram_size=2,
            pad_token_id=gpt_tokenizer.eos_token_id
        )
        
        # Decode and clean response
        response_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text.split("Assistant:")[-1].strip()
        
        return response_text
    except Exception as e:
        print(f"Error in message processing: {str(e)}")
        return "I apologize, I encountered an error processing your message."
        
