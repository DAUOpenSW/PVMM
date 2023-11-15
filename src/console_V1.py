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

import time
# start = time.time()
# print("Runtime :", time.time() - start)

credential_path = "C:/Users/eoduq/Desktop/PVMM/PVMM/src/embeding_models/sttv1-398306-8720d8b20a7e.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def save_to_db(uid, origin_text, filter_text, score):
    try:
        connection = mysql.connector.connect(user='js',
                                             password='js',
                                             host='localhost',
                                             database='js')
        cursor = connection.cursor()

        query = '''INSERT INTO PVMM (uid, origin_text, filter_text, score, time_stamp)
                   VALUES (%s, %s, %s, %s, %s)'''
        timestamp = datetime.now()
        cursor.execute(query, (uid, origin_text, filter_text, float(score), timestamp))
        connection.commit()

    except mysql.connector.Error as error:
        print(f"Error connecting to MySQL server: {error}")
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return True

def get_threshold(stream, chunk, RATE,seconds=5):
    print("소음치 측정 중")
    audio_energy_values = []

    for _ in range(int(seconds / chunk * RATE)):
        data = stream.read(chunk)
        audio_energy = np.frombuffer(data, dtype=np.int16).max()
        audio_energy_values.append(audio_energy)

    average = np.mean(audio_energy_values)
    std_dev = np.std(audio_energy_values)

    return int(average + std_dev+400)

def record_audio(output_file, stop_when_silence=3):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    SILENCE_THRESHOLD = get_threshold(stream, CHUNK,RATE)  # 소리 강도 임계 값 (잠잠한 환경에 적합)
    print(f"측정된 소음치 : {SILENCE_THRESHOLD}")
    print("당신의 목소리를 녹음 중입니다...")
    frames = []
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_energy = np.frombuffer(data, dtype=np.int16).max()  # audio_energy 변환 수정

        # 소리가 임계값 이하일 경우 카운트 증가, 아니면 카운트 초기화,,
        if audio_energy < SILENCE_THRESHOLD:
            silence_count += 1
        else:
            silence_count = 0

        # 지정된 시간 동안 소리가 없을 때 녹음 종료
        if silence_count >= RATE/CHUNK * stop_when_silence:
            break

    print("녹음이 끝났습니다.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 녹음한 음성 파일 저장
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
def blur_audio(start_sec, end_sec,idx):
    """Blurs audio between start and end times."""
    audio_path="test"+str(idx)+".wav"
    
    audio = AudioSegment.from_file(audio_path)
    
    start_millisec = int(start_sec * 1000)
    end_millisec = int(end_sec * 1000)

    before_blur_part=audio[:start_millisec]
    silence_duration = end_millisec - start_millisec
    silence_segment = AudioSegment.silent(duration=silence_duration)
    after_blur_part=audio[end_millisec:]

    final_audio=before_blur_part + silence_segment + after_blur_part
    idx=int(idx)+1
    audio_path="test"+str(idx)+".wav"
    final_audio.export(audio_path, format="wav")


def recognize_speech(filename, curse):
    client = speech.SpeechClient()

    with open(filename, "rb") as audio_file:
        input_audio = audio_file.read()

    audio = speech.RecognitionAudio(content=input_audio)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_word_time_offsets=True,
    )

    try:
        print("음성을 인식 중입니다...")
        start = time.time()
        
        response = client.recognize(config=config, audio=audio)
        start_idx=int(0)
        end_idx= int(0)
        for result in response.results:
            text = result.alternatives[0].transcript
            
            text,filter_word,word_list = curse.masking(result.alternatives[0].transcript)
            
            print("origin : {}".format(result.alternatives[0].transcript))
            print("filter : " + str(text))
            print(filter_word)
            print(result.alternatives[0].words)
            for word_info in result.alternatives[0].words:
                end_idx += len(word_info.word)
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                
                # If the word is a curse, blur it in the original audio,,,,
                for i in range(0,len(filter_word)):
                    if ((filter_word[i][0]<end_idx) and (filter_word[i][0]>=start_idx)) or ((filter_word[i][0]+filter_word[i][1]<end_idx) and (filter_word[i][0]+filter_word[i][1]>=start_idx)) :
                        print(i)
                        print(start_time)
                        print(end_time)
                        blur_audio(start_time,end_time,i)
                start_idx = end_idx
            print("Runtime :", time.time() - start)
    except Exception as e:
        print(f"An error occurred: {e}")
    




if __name__ == "__main__":
    output_file = "test0.wav"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    weights_paths = ['C:/Users/eoduq/Desktop/PVMM/PVMM/src/models/weights6.h5']

    curse = CurseDetector(weights_paths)
    curse.masking("녹음 준비 완료",flag=False)
    while(True):
        # 음성 녹음 및 저장,,,ad
        record_audio(output_file)
        
        # 저장된 음성 파일을 텍스트로 변환
        recognize_speech(output_file,curse)
        break
    
    
    
