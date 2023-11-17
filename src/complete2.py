import pyaudio
import os
from curse_detector import CurseDetector
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
# 이건 뭔 파일이냐 ??
# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100

weights_paths = ['C:/hw/Curse-detection-v2/src/models/weights6.h5']
curse = CurseDetector(weights_paths)

def listen_print_loop(responses):
    for response in responses:
        start_idx=int(0)
        end_idx= int(0)
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue
        
        # Check if this result is the final result for this portion of the audio.
        if result.is_final:
            text,filter_word,word_list = curse.masking(result.alternatives[0].transcript)
            print("filter : " + str(text))
            print("origin : {}".format(result.alternatives[0].transcript))
            # Get start and end time of each word
            for word_info in result.alternatives[0].words:
                end_idx += len(word_info.word)
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                
                # If the word is a curse, blur it in the original audio
                for i in range(0,len(filter_word)):
                    if ((filter_word[i][0]<end_idx) and (filter_word[i][0]>=start_idx)) or ((filter_word[i][0]+filter_word[i][1]<end_idx) and (filter_word[i][0]+filter_word[i][1]>=start_idx)) :
                        blur_audio(start_time,end_time)
                start_idx = end_idx

def blur_audio(start_sec, end_sec):
    """Blurs audio between start and end times."""
    audio_path="./path_to_your_audio_file.wav"
    
    audio = AudioSegment.from_file(audio_path)
    
    start_millisec = int(start_sec * 1000)
    end_millisec = int(end_sec * 1000)

    before_blur_part=audio[:start_millisec]
    blur_part=audio[start_millisec:end_millisec].low_pass_filter(10)
    after_blur_part=audio[end_millisec:]

    final_audio=before_blur_part+blur_part+after_blur_part
    
    final_audio.export("blurred_"+audio_path, format="wav")

def main():
  
    client=speech.SpeechClient()

    config=speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code="ko-KR",
    enable_word_time_offsets=True,
    )

    streaming_config=speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,   # We still want interim results so we can get results while audio is still being recorded.
    )

    mic=pyaudio.PyAudio()

    stream=mic.open(
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
    text,word,word_list= curse.masking("loding complete",flag=False)
    print(text)
    main()
