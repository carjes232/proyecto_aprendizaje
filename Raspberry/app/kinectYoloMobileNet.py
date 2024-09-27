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
import tensorflow as tf
import re
from threading import Lock
import signal
import sys

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../models/centernet_mobilenetv2_fpn_od/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label map
def load_label_map(label_map_path):
    label_map = {}
    current_id = None

    with open(label_map_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Match the ID line
            if line.startswith('id:'):
                match = re.match(r'id:\s*(\d+)', line)
                if match:
                    current_id = match.group(1)

            # Match the display_name line
            elif line.startswith('display_name:'):
                if current_id is not None:
                    display_name = line.split(':', 1)[1].strip().strip('"')
                    label_map[current_id] = display_name
                    current_id = None  # Reset for the next item

    return label_map

label_map = load_label_map('label_map.txt')
print("Label Map Loaded:", label_map)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global variables
keep_running = True
last_motor_move_time = time.time()
last_led_change_time = time.time()
current_angle = -5
rgb_image_path = 'RGBImage.jpg'  # Use JPEG for better compression
tflite_ready = False  # Renamed from yolo_ready for clarity
start_time_inference = 0
inference_time_value = 0.0  # In-memory inference time tracking

# Initialize a lock
frame_lock = Lock()

# Handle graceful shutdown
def signal_handler(sig, frame):
    global keep_running
    logger.info("Interrupt received, stopping...")
    keep_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Frame generator for video streaming
def generate_frames():
    global tflite_ready, start_time_inference, inference_time_value

    while keep_running:
        if tflite_ready:
            try:
                with frame_lock:
                    # Load and preprocess the image
                    input_image = cv2.imread(rgb_image_path)
                    if input_image is None:
                        logger.error(f"Failed to read image from {rgb_image_path}")
                        tflite_ready = False
                        continue

                    # Do NOT convert BGR to RGB
                    # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

                    # Resize and prepare the input tensor
                    input_tensor = cv2.resize(input_image, (320, 320))  # Adjust size if needed
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    input_tensor = input_tensor.astype(np.float32)  # Remove normalization

                    # Log input tensor details
                    logger.info(f"Input Tensor Shape: {input_tensor.shape}")
                    logger.info(f"Input Tensor Data Type: {input_tensor.dtype}")

                    # Perform TFLite inference
                    start_time_inference = time.time()
                    logger.info('Starting TFLite inference')
                    interpreter.set_tensor(input_details[0]['index'], input_tensor)
                    interpreter.invoke()
                    inference_time = time.time() - start_time_inference
                    inference_time_value = inference_time
                    logger.info(f"Inference Time: {inference_time:.2f} seconds")

                    # Extract detection results
                    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
                    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
                    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

                    # Log output tensor details
                    logger.info(f"Boxes Shape: {boxes.shape}")
                    logger.info(f"Classes Shape: {classes.shape}")
                    logger.info(f"Scores Shape: {scores.shape}")
                    logger.debug(f"Sample Scores: {scores[:5]}")

                    # Threshold for displaying detections
                    threshold = 0.3

                    # Annotate the image
                    for i in range(len(scores)):
                        if scores[i] >= threshold:
                            ymin, xmin, ymax, xmax = boxes[i]
                            (startX, startY, endX, endY) = (
                                int(xmin * input_image.shape[1]),
                                int(ymin * input_image.shape[0]),
                                int(xmax * input_image.shape[1]),
                                int(ymax * input_image.shape[0])
                            )

                            # Adjust class ID by adding 1 to match label_map IDs
                            class_id = int(classes[i]) + 1
                            class_id_str = str(class_id)

                            # Debug: Print class ID and label
                            logger.info(f"Detection {i}: Class ID from model: {classes[i]} mapped to Label Map ID: {class_id_str} -> {label_map.get(class_id_str, 'N/A')}")

                            # Draw bounding box
                            cv2.rectangle(input_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                            # Draw label and score
                            label = f"{label_map.get(class_id_str, 'N/A')}: {scores[i]:.2f}"
                            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            startY_label = max(startY, label_size[1])
                            cv2.rectangle(
                                input_image,
                                (startX, startY_label - label_size[1]),
                                (startX + label_size[0], startY_label + base_line),
                                (0, 255, 0),
                                cv2.FILLED
                            )
                            cv2.putText(
                                input_image, label,
                                (startX, startY_label),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1
                            )

                    # Convert back to BGR for saving/displaying with OpenCV
                    # Since input_image is in BGR, no need to convert
                    image_bgr = input_image.copy()

                    # Encode the annotated image
                    ret, buffer = cv2.imencode('.jpg', image_bgr)
                    frame = buffer.tobytes()

                    # Reset the flag
                    tflite_ready = False

                    logger.info("Inference completed and frame yielded")

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                logger.error(f"Error during TFLite inference: {e}")
                time.sleep(1)
        else:
            time.sleep(0.1)

# Kinect RGB image capture function
def save_rgb_image(data):
    global tflite_ready, start_time_inference
    with frame_lock:
        start_time_inference = time.time()  # Start timing when the image is saved

        rgb_image = frame_convert.video_cv(data)  # Convert RGB frame
        # Do NOT convert BGR to RGB here
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Save the RGB image without color conversion
        cv2.imwrite(rgb_image_path, rgb_image)
        logger.info(f"Saved RGB image: {rgb_image_path}")

        tflite_ready = True

# Kinect RGB callback
def display_rgb(dev, data, timestamp):
    global tflite_ready
    if not tflite_ready:
        try:
            save_rgb_image(data)
        except Exception as e:
            logger.error(f"Error saving RGB image: {e}")

# Flask route to display the live TFLite-processed video feed
@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Live TFLite Video Feed</title>
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
            <h1>TFLite Video Feed</h1>
            <h2>Processed Image</h2>
            <img src="{{ url_for('video_feed') }}" alt="Live TFLite Video Feed">
            <p>Last TFLite Inference Time: <span id="inference-time">0.00 seconds</span></p>
        </body>
        </html>
    ''')

# Flask route to stream the video
@app.route('/video_feed')
def video_feed():
    logger.info("video_feed route accessed")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to return inference time
@app.route('/inference-time')
def get_inference_time():
    try:
        logger.info("Fetching inference time")
        return f"{inference_time_value:.2f}"
    except Exception as e:
        logger.error(f"Error fetching inference time: {e}")
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

def run_kinect():
    try:
        freenect.runloop(video=display_rgb, body=body)
    except freenect.Kill:
        logger.info("Kinect runloop killed.")
    except Exception as e:
        logger.error(f"Error in Kinect runloop: {e}")

if __name__ == '__main__':
    # Start the Kinect loop with RGB capture processing
    logger.info("Starting Kinect runloop thread.")
    threading.Thread(target=run_kinect, daemon=True).start()

    # Start the Flask server
    logger.info("Starting Flask server.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
