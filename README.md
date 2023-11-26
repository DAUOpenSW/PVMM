# PVMM
## 변경 사항
* 모델 최종 출력  .txt -> .wav(음성)로 수정.
* 음성의 욕설 부분은 블러 처리 되어 출력.
* ex) 아 진짜 씨발 기훈이형 일 좆같이 할래? -> 아 진짜 (---) 기훈이형 일 (---)이 할래?

# 프로젝트 시연 영상(.txt, 문자열 블러 처리)

[![Video Label](/src/imgs/logo.png)](https://youtu.be/9pCCKXYSrt8?si=SUoRJZ16m9R-fcXL)

##### 이미지를 누르면 시연영상을 확인할 수 있습니다.

# 프로젝트 시연 영상(.wav, 음성 블러 처리)

[결과 영상](https://youtu.be/VRiUKeyz3M4)

<br><br>

## ✔목차
* [프로젝트 정보](#🔎프로젝트-정보)
* [프로젝트 소개](#🖐프로젝트-소개)
* [팀원 소개](#🙋‍♀️팀원-소개)
* [모델](#모델)
* [데이터](#데이터)
* [학습 과정](#학습-과정)
* [Ref](#📝ref)

<br><br>

## 🔎프로젝트 정보
> 동아대학교 컴퓨터공학과 학생 팀  
> 개발 기간: 2023.07.14 ~

<br><br>

## 🖐프로젝트 소개
> 본 프로젝트는 2023 공개SW 개발자 대회 출품작으로, 머신러닝을 이용한 실시간 음성 한국어 욕설 필터링 시스템을 API 서비스로 제공하는 프로젝트입니다.

<br><br>

## 🙋‍♀️팀원 소개
|<img width="300" src="https://github.com/DAUOpenSW/Kind_Words_Cloud/assets/91776984/1f6c5417-5801-4748-866d-d260fcd5c36b"/>|<img width="300" src="https://github.com/DAUOpenSW/Kind_Words_Cloud/assets/91776984/21996af7-da7f-4559-bca5-6486a4eb5f4f"/>|<img width="300" src="https://github.com/DAUOpenSW/Kind_Words_Cloud/assets/91776984/001b876f-cbe4-4ed3-8fa5-9009ab4b2bb7"/>|<img width="300" src="https://github.com/DAUOpenSW/Kind_Words_Cloud/assets/91776984/38fc7d5d-df49-47a3-b302-ea6993a839dd"/>|<img width="300" src="https://github.com/DAUOpenSW/Kind_Words_Cloud/assets/91776984/725920a7-f2d2-4b60-a9ae-c6ff4cd12440"/>|
|:---:|:---:|:---:|:---:|:---:|
|컴퓨터공학과<br>4학년|컴퓨터공학과<br>4학년|컴퓨터공학과<br>4학년|컴퓨터공학과<br>4학년|컴퓨터공학과<br>4학년|
| [김현우](https://github.com/HIT18216) | [김혜영](https://github.com/hyeyeoung) | [박성민](https://github.com/ParkSeungMin1) | [서지헌](https://github.com/MyCoooi) | [이영우](https://github.com/Dandyoung) |
|개발|PM|개발|개발|개발|

<br><br><br>

# 모델
Bidirectional-LSTM을 사용하였고 어텐션 메커니즘을 적용하여 욕설 마스킹 기능을 구현했습니다. 
<br>
Google Cloud STT API를 사용하여 Time Stamp 및 STT를 구현하였습니다.
<br>
Pydub 라이브러리를 활용하여 오디오 블러처리를 구현하였습니다.

모델 구조는 아래와 같습니다

![1](src/imgs/model.png)

더 자세한 내용은 [코드](https://github.com/DAUOpenSW/PVMM/blob/main/src/models.py)를 참고해 주세요.

# 데이터

욕설 데이터셋은 약 41,000개의 문장에 대해 욕설 여부를 분류한 데이터셋입니다.

![dataset](/src/imgs/dataset_table.png)

# 학습 과정
## 1. 전처리

- 연속적인 글자 단축 (ㅋㅋㅋㅋ → ㅋㅋ)
- 초성, 중성, 종성으로 분리 (안녕 → ㅇㅏㄴㄴㅕㅇ)

## 2. 임베딩**

- **fasttext 임베딩**

  fasttext를 활용하여 의미 기반의 임베딩 수행
  
  이 레포지토리에선 미리 학습된 fasttext 모델을 사용합니다.
  
  때문에 예측을 위해선 fasttext 모델이 `embedding_models`폴더에 `fasttext.bin`이라는 이름으로 옮겨져 있어야 합니다.
  
  fasttext 모델은 [여기](https://drive.google.com/file/d/1kW7cRDRe7HMQskSytv9gUbkUFhG8LrIn/view?usp=drive_link)에서 다운로드받을 수 있습니다.
  
- **mfcc 임베딩**

  비슷한 발음의 단어를 비슷한 벡터로 임베딩 (MFCC 알고리즘 활용)

## 📝Ref
https://github.com/2runo/Curse-detection-v2
