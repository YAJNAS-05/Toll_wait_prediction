
# 🚦 Toll Wait Time Prediction and Traffic Monitoring App

This project is a machine learning-powered web application designed to predict **toll booth wait times** and monitor **traffic density** using computer vision and pre-trained models.

---

## 📂 Project Structure

```
traffic_app/
│
├── app.py                        # Main Flask application
├── readme.txt                    # Initial notes
├── yolov5s.pt                    # YOLOv5 pre-trained weights for vehicle detection
├── toll_time_model.pkl          # Trained ML model for toll wait time prediction
├── traffic_density_model.pkl    # Trained ML model for traffic density classification
├── templates/
│   └── index.html               # Frontend HTML interface
```

---

## ⚙️ Features

- **Vehicle Detection** using YOLOv5
- **Traffic Density Classification** using a trained ML model
- **Toll Wait Time Prediction** based on traffic conditions
- Simple web interface built with **Flask**

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YAJNAS-05/Toll_wait_prediction.git
cd Toll_wait_prediction
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install key packages manually:
```bash
pip install flask torch torchvision scikit-learn opencv-python
```

### 3. Run the App
```bash
python app.py
```

Then open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 🧠 Models Used

- `yolov5s.pt`: Pre-trained YOLOv5 model for real-time vehicle detection.
- `traffic_density_model.pkl`: Custom-trained classifier for predicting traffic congestion level.
- `toll_time_model.pkl`: ML regression model to predict estimated waiting time at tolls.

---

## 📸 Demo

_Screenshots or demo GIF can be added here._

---

## 🛠️ Future Improvements

- Add support for real-time video feed from traffic cameras
- Improve UI design
- Deploy on cloud (e.g., Render, AWS, or Streamlit Cloud)

---

## 📄 License

This project is for educational purposes. License can be added here.

---

## 🙋‍♂️ Author

Developed by [YAJNAS-05](https://github.com/YAJNAS-05)
