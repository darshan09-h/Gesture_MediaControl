from flask import Flask, render_template, Response, jsonify
import cv2
from logic import GestureController

app = Flask(__name__)

# Global state
camera_active = False
detector = None  # Lazy initialization

def gen_frames():
    global camera_active, detector
    
    # Initialize the camera and model only when needed
    cap = cv2.VideoCapture(0)
    if detector is None:
        detector = GestureController()
        
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
            
        # Pass frame to logic.py for processing
        try:
            processed_frame = detector.process(frame)
        except Exception as e:
            print(f"Error in processing: {e}")
            continue

        # Encode for web
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_power')
def toggle_power():
    global camera_active
    camera_active = not camera_active
    return jsonify({'status': 'on' if camera_active else 'off'})

if __name__ == "__main__":
    app.run(debug=True, port=5000)