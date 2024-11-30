import freenect
import cv2
import frame_convert
import time
import random
import numpy as np
import threading
import logging
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import torch
from queue import Queue
print(freenect.__file__)
# Inicializar la aplicación Flask
app = Flask(__name__)

# Configuración del dispositivo (CPU en Raspberry Pi)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Cargar un modelo YOLO ligero (por ejemplo, YOLOv5s)
# Asegúrate de tener un modelo optimizado para uso en CPU
model = YOLO("../models/yolo/yolov8n_ncnn_model")

# Realizar una inferencia dummy para cargar los pesos y evitar retrasos en la primera inferencia real
def dummy_inference():
    dummy_image = np.zeros((320, 240, 3), dtype=np.uint8)  # Imagen negra de menor resolución
    model(dummy_image)
    logger.info("Inferencia dummy completada para calentar el modelo.")

dummy_inference()

# Variables globales
keep_running = True
last_motor_move_time = time.time()
last_led_change_time = time.time()
current_angle = -5
inference_time_file = 'inference_time.txt'
yolo_ready = False
start_time = 0
inference_time = 0

# Colas para manejar frames y resultados
frame_queue = Queue(maxsize=5)  # Limitar el tamaño para evitar acumulación
result_queue = Queue(maxsize=5)

# Función para capturar la imagen RGB de Kinect y ponerla en la cola
def capture_frames():
    global keep_running
    while keep_running:
        try:
            # Capturar frame de Kinect
            rgb_frame, _ = freenect.sync_get_video()
            if rgb_frame is not None:
                # Convertir el frame
                rgb_image = frame_convert.video_cv(rgb_frame)

                # Poner el frame en la cola
                if not frame_queue.full():
                    frame_queue.put(rgb_image)
                else:
                    logger.warning("Frame queue llena. Frame descartado.")
            time.sleep(0.05)  # Pausa para reducir la carga (aprox. 20 FPS)
        except Exception as e:
            logger.error(f"Error capturando frames: {e}")

# Función para realizar inferencia YOLO en los frames de la cola
def yolo_inference():
    global keep_running
    while keep_running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                start_time = time.time()
                with torch.no_grad():
                    results = model(frame)
                inference_time = time.time() - start_time

                # Procesar resultados
                annotated_frame = results[0].plot()
                # Poner el resultado en la cola
                if not result_queue.full():
                    result_queue.put((annotated_frame, inference_time))
                else:
                    logger.warning("Result queue llena. Resultado descartado.")

                # Guardar tiempo de inferencia
                with open(inference_time_file, 'w') as f:
                    f.write(f"{inference_time:.2f}")

            except Exception as e:
                logger.error(f"Error durante la inferencia YOLO: {e}")
        else:
            time.sleep(0.01)  # Pausa si no hay frames

# Función para manejar el motor y el LED de Kinect
def manage_kinect(dev, ctx):
    global last_motor_move_time, current_angle, last_led_change_time, keep_running
    while keep_running:
        current_time = time.time()

        # Gestionar motor
        if current_time - last_motor_move_time > 30:
            current_angle = 5 if current_angle == -5 else -5
            try:
                freenect.set_tilt_degs(dev, current_angle)
                logger.info(f"Motor movido a {current_angle} grados.")
            except Exception as e:
                logger.error(f"Fallo al mover el motor: {e}")
            last_motor_move_time = current_time

        # Cambiar LED
        if current_time - last_led_change_time > 20:
            led_color = random.randint(0, 6)
            try:
                freenect.set_led(dev, led_color)
                logger.info(f"Color del LED cambiado a {led_color}.")
            except Exception as e:
                logger.error(f"Fallo al cambiar el color del LED: {e}")
            last_led_change_time = current_time

        time.sleep(1)  # Pausa para evitar sobrecarga

# Callback RGB de Kinect (ya no se usa)
def display_rgb(dev, data, timestamp):
    pass

# Ruta de Flask para mostrar el feed de video procesado por YOLO en vivo
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
                            document.getElementById('inference-time').innerText = data + ' segundos';
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
            <p>Last YOLO Inference Time: <span id="inference-time">0.00 segundos</span></p>
        </body>
        </html>
    ''')

# Ruta de Flask para transmitir el video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Generador de frames para el streaming
def generate_frames():
    while keep_running:
        if not result_queue.empty():
            annotated_frame, inference_time = result_queue.get()
            try:
                # Codificar la imagen en JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()

                # Yield del frame en formato byte
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f"Error generando frame: {e}")
        else:
            time.sleep(0.01)  # Pausa si no hay resultados

# Ruta de Flask para devolver el tiempo de inferencia
@app.route('/inference-time')
def get_inference_time():
    try:
        # Log para confirmar que se está accediendo al endpoint
        logger.info("Obteniendo tiempo de inferencia del archivo")

        with open(inference_time_file, 'r') as f:
            inference_time_str = f.read().strip()

        logger.info(f"Tiempo de inferencia leído del archivo: {inference_time_str}")
        return inference_time_str
    except FileNotFoundError:
        logger.error("Archivo inference_time.txt no encontrado, devolviendo 0.00")
        return "0.00"

if __name__ == '__main__':
    # Iniciar el hilo de captura de frames
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    logger.info("Hilo de captura de frames iniciado.")

    # Iniciar el hilo de inferencia YOLO
    inference_thread = threading.Thread(target=yolo_inference, daemon=True)
    inference_thread.start()
    logger.info("Hilo de inferencia YOLO iniciado.")

    # Iniciar el hilo de gestión de motor y LED de Kinect
    kinect_thread = threading.Thread(target=lambda: freenect.runloop(video=display_rgb, body=manage_kinect), daemon=True)
    kinect_thread.start()
    logger.info("Hilo de gestión de Kinect iniciado.")

    # Iniciar el servidor Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)

    # Cuando se detenga Flask, detener todos los hilos
    keep_running = False
