# How to Run the NOXIS Deepfake Forensics Engine

Follow these steps to run the application locally on your Windows machine:

### 1. Start the Backend Server

Open your terminal (PowerShell or Command Prompt), navigate to the project directory, and run the Flask application:

```bash
cd "c:\Users\varun\OneDrive\Desktop\DEEPFAKE IDENTIFIER"
py backend/app.py
```
*(Note: If `py` is not recognized, try `python backend/app.py` or `python3 backend/app.py`)*

Wait until you see the message indicating the server is running on `http://localhost:5000` or `http://127.0.0.1:5000`.

### 2. Open the Web Interface

Once the server is running, open your preferred web browser (e.g., Chrome, Edge, Firefox) and go to:

**[http://localhost:5000](http://localhost:5000)**

### 3. Use the Application

1. Click **Launch Detection** on the landing page.
2. Click the upload area or drag and drop a video file (`.mp4`, `.avi`, `.mov`, etc.).
3. Click the **Begin Forensic Scan** button.
4. Wait for the engine to analyze the video frames using the EfficientNet-B3 + BiLSTM model.
5. Review the final verdict (REAL/FAKE), confidence score, and the Grad-CAM attention heatmaps!

---

### Troubleshooting

- **Dependencies missing?** Ensure you have installed the requirements:
  ```bash
  pip install -r requirements.txt
  ```
- **Port already in use?** Make sure you don't have another application running on port 5000.
