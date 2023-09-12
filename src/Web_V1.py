import speech_recognition as sr
from curse_detector import CurseDetector
import os
import wave
import pyaudio
import numpy as np
import mysql.connector
from datetime import datetime
from google.cloud import speech
from pydub import AudioSegment

credential_path = 'C:/Users/user/Desktop/sttv1-398306-8720d8b20a7e.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path 

def blur_audio(start_sec, end_sec):
    """Blurs audio between start and end times."""
    audio_path="test.wav"
    
    audio = AudioSegment.from_file(audio_path)
    
    start_millisec = int(start_sec * 1000)
    end_millisec = int(end_sec * 1000)

    before_blur_part=audio[:start_millisec]
    silence_duration = end_millisec - start_millisec
    silence_segment = AudioSegment.silent(duration=silence_duration)
    after_blur_part=audio[end_millisec:]

    final_audio=before_blur_part + silence_segment + after_blur_part
    
    final_audio.export(audio_path, format="wav")


def recognize_speech(filename, curse):
    client = speech.SpeechClient()

    with open(filename, "rb") as audio_file:
        input_audio = audio_file.read()
    
    audio = speech.RecognitionAudio(content=input_audio)
    print("2")
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_word_time_offsets=True,
    )
    print("3")
    try:
        print("음성을 인식 중입니다...")
        
        response = client.recognize(config=config, audio=audio)
        start_idx=int(0)
        end_idx= int(0)
        print("4")
        print(response.results)
        for result in response.results:
            text = result.alternatives[0].transcript
            print(text)
            
            
            text,filter_word,word_list = curse.masking(result.alternatives[0].transcript)
            
            print("origin : {}".format(result.alternatives[0].transcript))
            print("filter : " + str(text))
            
            for word_info in result.alternatives[0].words:
                end_idx += len(word_info.word)
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                
                # If the word is a curse, blur it in the original audio
                for i in range(0,len(filter_word)):
                    if ((filter_word[i][0]<end_idx) and (filter_word[i][0]>=start_idx)) or ((filter_word[i][0]+filter_word[i][1]<end_idx) and (filter_word[i][0]+filter_word[i][1]>=start_idx)) :
                        blur_audio(start_time,end_time)
                start_idx = end_idx
    except Exception as e:
        print(f"An error occurred: {e}")
    




        