from inference_sdk import InferenceHTTPClient
import cv2
import time
import threading
import queue
from flask import Flask, Response, jsonify, send_from_directory
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Roboflow API Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UFHEmXSGinK9eFmOFc6K"
)

# Twilio Configuration (replace with your credentials)
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"
WHATSAPP_TO = "whatsapp:+1234567890"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Shared state for fall detection and notification
fall_detected_global = False
last_notification_time = 0
NOTIFICATION_COOLDOWN = 60  # Seconds
frame_queue = queue.Queue(maxsize=1)  # Queue for inference frames

def detect_fall(frame):
    cv2.imwrite("temp_frame.jpg", frame)
    result = CLIENT.run_workflow(
        workspace_name="work-qj6em",
        workflow_id="custom-workflow-3",
        images={"image": "temp_frame.jpg"},
        use_cache=True
    )
    fall_detected = "fall" in str(result).lower()  # Adjust based on your model
    return fall_detected, result

def send_whatsapp_notification(timestamp):
    global last_notification_time
    current_time = time.time()
    if current_time - last_notification_time < NOTIFICATION_COOLDOWN:
        return
    try:
        message = twilio_client.messages.create(
            body=f"🚨 Fall Detected! Time: {timestamp}",
            from_=TWILIO_WHATSAPP_FROM,
            to=WHATSAPP_TO
        )
        print(f"WhatsApp message sent: {message.sid}")
        last_notification_time = current_time
    except TwilioRestException as e:
        print(f"Twilio error: {e}")

def inference_thread():
    global fall_detected_global
    while True:
        frame = frame_queue.get()
        fall_detected, result = detect_fall(frame)
        fall_detected_global = fall_detected
        if fall_detected:
            send_whatsapp_notification(time.strftime("%Y-%m-%d %H:%M:%S"))
        frame_queue.task_done()

# Start inference thread
threading.Thread(target=inference_thread, daemon=True).start()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Add frame to queue for inference if not full
        if frame_queue.qsize() < 1:
            frame_queue.put(frame.copy())
        
        # Stream frame immediately
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    print("Serving index.html")
    return app.send_static_file('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    success, frame = cap.read()
    if success:
        fall_detected, result = detect_fall(frame)  # Direct call for API endpoint
        if fall_detected:
            send_whatsapp_notification(time.strftime("%Y-%m-%d %H:%M:%S"))
        return jsonify({
            "status": "Fall Detected" if fall_detected else "Safe",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": result
        })
    return jsonify({"error": "Camera error"})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    print("Starting Flask app")
    app.run(host="0.0.0.0", port=5000, debug=True)