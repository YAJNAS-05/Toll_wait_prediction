-----------------------------
🚦 TRAFFIC ANALYZER PROJECT
-----------------------------

▶ Student Details:
1. Jayashree Annupama RK -- 2303717624322022
2. Pooja S -- 2303717624322039
3. Sanjay Karthik S -- 2303717624321049

----------------------------------
▶ Features / Bugs / Missing:
----------------------------------

✅ Features Implemented:
- Image upload for vehicle detection
- Video upload for real-time vehicle detection
- Traffic density prediction using trained ML model
- Toll wait time prediction using regression
- Automatic annotation of image/video frames
- Spinner/progress animation while analyzing video
- Bootstrap-based responsive UI

⚠ Known Bugs:
- Video processing may take time for large files (no progress bar % shown)
- Browser may need refresh after large video analysis
- No audio alerts added (optional feature)
- App doesn't restrict upload size (may crash if very large files are uploaded)
- No clear error message if ML model files are missing
- Video player might not appear if browser does not support .mp4

❌ Missing:
- Live webcam input support
- History of past uploads (no result log or database)
- Mobile responsiveness could be improved (layout breaks on very small screens)
- No compression or optimization for uploaded/processed video files
- No admin or user login authentication
- No option to download the processed video/image

----------------------------------
▶ Special Instructions: How to Run the Project
----------------------------------

This section explains how to set up and run the project locally on any Windows/Linux/Mac machine with Python installed.

🛠 Step 1: Install Python (if not already installed)
- Download and install Python 3.8 or above from: https://www.python.org/downloads/

🧱 Step 2: Install Required Python Libraries
Open a terminal or command prompt in your project folder and run:

pip install flask torch torchvision opencv-python scikit-learn joblib

This will install:
- Flask: for the web app
- Torch & Torchvision: for the YOLOv5 model
- OpenCV: for image/video processing
- Scikit-learn: for ML predictions
- Joblib: to load saved ML models

📁 Step 3: Project Folder Structure
Make sure your folder looks like this:

traffic_app/
├── app.py
├── traffic_density_model.pkl
├── toll_time_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── uploads/

🚀 Step 4: Run the App
In terminal/command prompt, run:

python app.py

You will see:

 * Running on http://127.0.0.1:5000

🌐 Step 5: Open in Browser
Go to: http://127.0.0.1:5000

You will see the web interface to upload files.

📸 Step 6: Using the Web App
1. Click "Choose File" to upload an image or video.
2. Select the mode from the dropdown (Image or Video).
3. Click the "Analyze" button.
4. A spinner will show while the analysis is happening.
5. After processing:
   - Image: output with predictions is shown.
   - Video: output video will play with overlays.

❗ Troubleshooting:
- Make sure the uploads/ folder has write permissions.
- If video doesn’t show: check that the output is .mp4 and browser supports it.
- For slow video processing, keep video resolution under 720p.
- Ensure .pkl files exist and are not corrupted.

----------------------------------
▶ Team Contribution:
----------------------------------

🔹 Sanjay Karthik S
- ML model training (density + toll wait)
- Flask backend and model integration

🔹 Jayashree Annupama RK
- Bootstrap HTML and frontend layout
- Spinner/loader logic
- Result formatting and deployment prep

🔹 Pooja S
- YOLO model integration
- Video frame analysis
- Debugging and testing with real inputs

-----------------------------
✅ THANK YOU!
-----------------------------
