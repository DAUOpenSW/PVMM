import pyaudio
import os
from curse_detector import CurseDetector
from google.cloud import speech_v1p1beta1 as speech

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100


weights_paths = ['C:/hw/Curse-detection-v2/src/models/weights6.h5']
curse = CurseDetector(weights_paths)

def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue
        
        # Check if this result is the final result for this portion of the audio.
        if result.is_final:
            print("filter : " + str(curse.masking(result.alternatives[0].transcript)))
            print("origin : {}".format(result.alternatives[0].transcript))
            

def main():
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,   # We still want interim results so we can get results while audio is still being recorded.
    )

    mic = pyaudio.PyAudio()
    
    stream = mic.open(
         format=pyaudio.paInt16,
         channels=1,
         rate=RATE,
         input=True,
         frames_per_buffer=CHUNK)

    while True:   # Add an infinite loop to keep recording and transcribing.
          audio_generator =(stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * 5))) 
     
          requests =(speech.StreamingRecognizeRequest(audio_content=data)
               for data in audio_generator)

          responses = client.streaming_recognize(streaming_config, requests)

          listen_print_loop(responses)

if __name__ == "__main__":
    print(curse.masking("loding complete"))
    main()
