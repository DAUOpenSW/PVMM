from flask_cors import CORS
from flask import Flask, request, jsonify,redirect, url_for, render_template
from Web_V1 import *

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/upload', methods=['POST'])
def upload_audio():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    weights_paths = ['src/models/weights6.h5']

    curse = CurseDetector(weights_paths)
    try:
        uploaded_audio = request.files['audio']
        if not uploaded_audio:
            return jsonify({'error': '음성 파일이 전송되지 않았습니다.'}), 400
        # 음성 파일을 저장하고 필요한 처리 수행
        # audio_path = os.path.join('uploads', 'uploaded_audio.wav')
        audio_path = 'test.wav'
        uploaded_audio.save(audio_path)
        # result_path = os.path.join('uploads', 'result_audio.wav')
        # 여기에서 추가적인 처리를 수행할 수 있습니다.
        recognize_speech(audio_path, curse)
        return jsonify({'message': '음성 파일이 성공적으로 업로드되었습니다.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='localhost', port=5000, debug=True)
