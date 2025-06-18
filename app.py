from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import torch
import joblib
from werkzeug.utils import secure_filename
import uuid
import time
import subprocess
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
app.config['TEMP_FRAMES'] = 'static/temp_frames'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FRAMES'], exist_ok=True)

# Load models
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model_density = joblib.load('traffic_density_model.pkl')
model_toll = joblib.load('toll_time_model.pkl')
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
density_map = {0: 'Low', 1: 'Medium', 2: 'High'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def cleanup_temp_files():
    """Remove temporary frame files"""
    for filename in os.listdir(app.config['TEMP_FRAMES']):
        file_path = os.path.join(app.config['TEMP_FRAMES'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting temp file {file_path}: {e}")

def process_with_ffmpeg(input_path, output_path, fps, width, height):
    """Process video using FFmpeg as fallback"""
    try:
        # Create temp directory for frames
        temp_dir = os.path.join(app.config['TEMP_FRAMES'], str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        # First extract frames
        frame_pattern = os.path.join(temp_dir, 'frame_%04d.png')
        extract_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', 'fps={}'.format(fps),
            '-s', '{}x{}'.format(width, height),
            frame_pattern
        ]
        subprocess.run(extract_cmd, check=True)

        # Process each frame
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        total = 0
        density_label = ''
        wait_time = 0

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)

            results = model(frame)
            df = results.pandas().xyxy[0]
            vehicles = df[df['name'].isin(vehicle_classes)]

            car_count = len(vehicles[vehicles['name'] == 'car'])
            truck_count = len(vehicles[vehicles['name'] == 'truck'])
            total = len(vehicles)

            density_pred = model_density.predict([[total]])[0]
            wait_time = max(0, model_toll.predict([[total, car_count, truck_count, density_pred]])[0])
            density_label = density_map[density_pred]

            results.render()
            output_frame = results.ims[0].copy()
            info = f"Vehicles: {total} | Density: {density_label} | Wait: {wait_time:.2f} min"
            cv2.putText(output_frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(frame_path, output_frame)

        # Recombine frames with FFmpeg
        combine_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale={}:{}'.format(width, height),
            output_path
        ]
        subprocess.run(combine_cmd, check=True)

        # Cleanup
        shutil.rmtree(temp_dir)
        
        return total, density_label, wait_time

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", error="No file selected")
            
        file = request.files['file']
        mode = request.form['mode']
        
        if file.filename == '':
            return render_template("index.html", error="No file selected")
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            ext = filename.split('.')[-1]
            saved_name = f"{file_id}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
            file.save(filepath)

            if mode == 'image':
                return redirect(url_for('process_image', path=saved_name))
            else:
                return redirect(url_for('process_video', path=saved_name))
        else:
            return render_template("index.html", error="Invalid file type")

    return render_template("index.html")

@app.route('/process_image/<path>')
def process_image(path):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], path)
        
        if not os.path.exists(filepath):
            return render_template("index.html", error="Image file not found")

        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", error="Could not read image")

        results = model(img)
        df = results.pandas().xyxy[0]
        vehicles = df[df['name'].isin(vehicle_classes)]

        car_count = len(vehicles[vehicles['name'] == 'car'])
        truck_count = len(vehicles[vehicles['name'] == 'truck'])
        total = len(vehicles)

        density_pred = model_density.predict([[total]])[0]
        wait_time = max(0, model_toll.predict([[total, car_count, truck_count, density_pred]])[0])
        density_label = density_map[density_pred]

        results.render()
        annotated = results.ims[0].copy()
        info = f"Vehicles: {total} | Density: {density_label} | Wait: {wait_time:.2f} min"
        cv2.putText(annotated, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = f"static/uploads/processed_{path}"
        print("Sanju: " + out_path)
        cv2.imwrite(out_path, annotated)
        os.chmod(out_path, 0o644)

        return render_template("index.html", 
                            image=f"/{out_path}", 
                            total=total, 
                            density=density_label, 
                            wait=f"{wait_time:.2f}")

    except Exception as e:
        return render_template("index.html", error=f"Error processing image: {str(e)}")

@app.route('/process_video/<path>')
def process_video(path):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], path)
        
        if not os.path.exists(filepath):
            return render_template("index.html", error="Video file not found")
            
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            return render_template("index.html", error="Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure even dimensions
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        output_name = f"processed_{path.split('.')[0]}.mp4"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)

        # Try FFmpeg first as it's more reliable
        try:
            total, density_label, wait_time = process_with_ffmpeg(
                filepath, output_path, fps, width, height
            )
        except Exception as ffmpeg_error:
            print(f"FFmpeg processing failed: {ffmpeg_error}")
            # Fallback to OpenCV if FFmpeg fails
            return process_video_opencv_fallback(cap, path, fps, width, height)

        cap.release()
        os.chmod(output_path, 0o644)

        return render_template("index.html", 
                            video=f"/{output_path}?t={int(time.time())}", 
                            total=total, 
                            density=density_label, 
                            wait=f"{wait_time:.2f}")

    except Exception as e:
        if 'cap' in locals():
            cap.release()
        return render_template("index.html", error=f"Error processing video: {str(e)}")

def process_video_opencv_fallback(cap, path, fps, width, height):
    """Fallback video processing using OpenCV"""
    output_name = f"processed_{path.split('.')[0]}.mp4"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
    
    # Try multiple codecs
    codecs_to_try = ['avc1', 'mp4v', 'X264', 'MJPG']
    out = None
    
    for codec in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
        out = None

    if out is None:
        cap.release()
        return render_template("index.html", error="Could not initialize video writer with any codec")

    total = 0
    density_label = ''
    wait_time = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]
        vehicles = df[df['name'].isin(vehicle_classes)]

        car_count = len(vehicles[vehicles['name'] == 'car'])
        truck_count = len(vehicles[vehicles['name'] == 'truck'])
        total = len(vehicles)

        density_pred = model_density.predict([[total]])[0]
        wait_time = max(0, model_toll.predict([[total, car_count, truck_count, density_pred]])[0])
        density_label = density_map[density_pred]

        results.render()
        output_frame = results.ims[0].copy()
        info = f"Vehicles: {total} | Density: {density_label} | Wait: {wait_time:.2f} min"
        cv2.putText(output_frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Ensure frame matches output dimensions
        if output_frame.shape[0] != height or output_frame.shape[1] != width:
            output_frame = cv2.resize(output_frame, (width, height))
            
        out.write(output_frame)
        processed_frames += 1

    cap.release()
    out.release()

    if processed_frames == 0:
        return render_template("index.html", error="No frames processed in the video")

    os.chmod(output_path, 0o644)
    return render_template("index.html", 
                        video=f"/{output_path}?t={int(time.time())}", 
                        total=total, 
                        density=density_label, 
                        wait=f"{wait_time:.2f}")

if __name__ == '__main__':
    app.run(debug=True)