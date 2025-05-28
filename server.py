# server.py
import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import time
from apscheduler.schedulers.background import BackgroundScheduler
from run_swing_analyer import run_swing_analyzer
import shutil

app = Flask(__name__)
CORS(app)  # CORS 허용

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 파일 삭제 함수
def cleanup_old_files(folder):
    """폴더 내 하루 지난 파일과 하위 폴더 전체 삭제"""
    now = time.time()

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if not os.path.exists(file_path):
            continue

        file_mtime = os.path.getmtime(file_path)

        if now - file_mtime > 12 * 60 * 60:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"파일 삭제 성공: {file_path}")
                else:
                    shutil.rmtree(file_path)  # 폴더 재귀적 삭제
                    print(f"폴더 삭제 성공: {file_path}")
            except Exception as e:
                print(f"삭제 실패 {file_path}: {str(e)}")


@app.route('/upload', methods=['POST'])
def upload_file():
    """영상 업로드 핸들러"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video part"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # 고유 파일명 생성
            file_id = str(uuid.uuid4())
            input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_input.mp4")
            file.save(input_path)

            # 영상 처리 실행
            run_swing_analyzer(file_id)

            return jsonify({
                "message": "File processed successfully",
                "download_url": f"/download/video/{file_id}",
                "zip_url": f"/download/images/{file_id}"
            }), 200

        return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/video/<file_id>', methods=['GET'])
def download_file(file_id):
    """처리된 파일 다운로드"""
    output_path = os.path.join(PROCESSED_FOLDER+f"/{file_id}/", f"{file_id}_output.mp4")
    if not os.path.exists(output_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f"processed_{file_id}.mp4"
    )

@app.route('/download/images/<file_id>', methods=['GET'])
def download_images(file_id):
    """처리된 이미지 다운로드"""
    processed_folder = os.path.join(PROCESSED_FOLDER, file_id)
    if not os.path.exists(processed_folder):
        return jsonify({"error": "Processed folder not found"}), 404

    # 이미지 파일 목록
    image_files = [f for f in os.listdir(processed_folder) if f.startswith(file_id) and f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        return jsonify({"error": "No images found"}), 404

    # ZIP 파일로 압축
    zip_filename = f"{file_id}_images.zip"
    zip_path = os.path.join(PROCESSED_FOLDER, zip_filename)

    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', processed_folder)

    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({"status": "healthy"}), 200


def schedule_cleanup():
    """정리 작업 스케줄링"""
    scheduler = BackgroundScheduler(daemon=True)

    def cleanup_job():
        with app.app_context():
            cleanup_old_files(UPLOAD_FOLDER)
            cleanup_old_files(PROCESSED_FOLDER)

    # 1시간마다 실행
    scheduler.add_job(cleanup_job, 'interval', hours=1)
    scheduler.start()

if __name__ == '__main__':
    schedule_cleanup()
    app.run(host='0.0.0.0', port=5005, threaded=True)
