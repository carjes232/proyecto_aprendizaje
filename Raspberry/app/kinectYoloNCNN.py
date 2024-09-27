import freenect
import cv2
import frame_convert
import time
import random
import numpy as np
import os
import threading
import logging
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import torch
from threading import Lock

# Initialize Flask app
app = Flask(__name__)

# YOLO model
model = YOLO("../models/yolo/yolov8nd_complete_ncnn_model")

# Global variables
keep_running = True
last_motor_move_time = time.time()
last_led_change_time = time.time()
current_angle = -5
rgb_image_path = 'RGBImage.jpg'  # Use JPEG for better compression
annotated_image_path = 'YOLO_Processed.jpg'
inference_time_file = 'inference_time.txt'
yolo_ready = False
start_time = 0
inference_time = 0

# Initialize a lock
frame_lock = Lock()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# YOLO inference function is now integrated into the frame generator

# Frame generator for video streaming
def generate_frames():
    global yolo_ready, start_time, annotated_image_path, inference_time, inference_time_file

    while keep_running:
        if yolo_ready:
            try:
                with frame_lock:
                    # Perform YOLO inference
                    start_time = time.time()
                    print('Empieza inferencia')
                    results = model(rgb_image_path)
                    inference_time = time.time() - start_time

                    # Save YOLO processed image
                    annotated_image = results[0].plot()
                    ret, buffer = cv2.imencode('.jpg', annotated_image)
                    frame = buffer.tobytes()

                    # Write inference time to file
                    with open(inference_time_file, 'w') as f:
                        f.write(f"{inference_time:.2f}")

                    # Reset the flag
                    yolo_ready = False

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                logger.error(f"Error during YOLO inference: {e}")
                time.sleep(1)
        else:
            time.sleep(0.1)

# Kinect RGB image capture function
def save_rgb_image(data):
    global yolo_ready, start_time
    with frame_lock:
        start_time = time.time()  # Start timing when the image is saved

        rgb_image = frame_convert.video_cv(data)  # Convert RGB frame
        # Save the RGB image
        cv2.imwrite(rgb_image_path, rgb_image)
        logger.info(f"Saved RGB image: {rgb_image_path}")

        yolo_ready = True

# Kinect RGB callback
def display_rgb(dev, data, timestamp):
    global yolo_ready
    if not yolo_ready:
        try:
            save_rgb_image(data)
        except Exception as e:
            logger.error(f"Error saving RGB image: {e}")

# Flask route to display the live YOLO-processed video feed
@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Live YOLO Video Feed</title>
            <script>
                function fetchInferenceTime() {
                    fetch('/inference-time')
                        .then(response => response.text())
                        .then(data => {
                            document.getElementById('inference-time').innerText = data + ' seconds';
                        })
                        .catch(error => {
                            console.error('Error fetching inference time:', error);
                        });
                }

                // Fetch inference time every second
                setInterval(fetchInferenceTime, 1000);
            </script>
        </head>
        <body>
            <h1>YOLO Video Feed</h1>
            <h2>Processed Image</h2>
            <img src="{{ url_for('video_feed') }}" alt="Live YOLO Video Feed">
            <p>Last YOLO Inference Time: <span id="inference-time">0.00 seconds</span></p>
        </body>
        </html>
    ''')

# Flask route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to return inference time
@app.route('/inference-time')
def get_inference_time():
    try:
        # Log to confirm the endpoint is being hit
        logger.info("Fetching inference time from the file")

        with open(inference_time_file, 'r') as f:
            inference_time_str = f.read().strip()

        logger.info(f"Inference time read from file: {inference_time_str}")
        return inference_time_str
    except FileNotFoundError:
        logger.error("inference_time.txt file not found, returning 0.00")
        return "0.00"

# Kinect motor and LED management functions
def manage_motor(dev):
    global last_motor_move_time, current_angle
    if time.time() - last_motor_move_time > 30:
        current_angle = 5 if current_angle == -5 else -5
        try:
            freenect.set_tilt_degs(dev, current_angle)
            logger.info(f"Moved Kinect to {current_angle} degrees.")
        except Exception as e:
            logger.error(f"Failed to move motor: {e}")
        last_motor_move_time = time.time()

def change_led(dev):
    global last_led_change_time
    if time.time() - last_led_change_time > 20:
        led_color = random.randint(0, 6)
        try:
            freenect.set_led(dev, led_color)
            logger.info(f"Changed LED color to {led_color}.")
        except Exception as e:
            logger.error(f"Failed to change LED color: {e}")
        last_led_change_time = time.time()

# Kinect body function to manage motor and LED
def body(dev, ctx):
    global keep_running
    manage_motor(dev)
    change_led(dev)
    if not keep_running:
        raise freenect.Kill

if __name__ == '__main__':
    # Start the Kinect loop with RGB capture processing
    threading.Thread(target=lambda: freenect.runloop(video=display_rgb, body=body), daemon=True).start()

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
