#!/usr/bin/env python
import time
import cv2
import numpy as np
import tensorflow as tf
import re
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TensorFlow Lite model
model_path = "model.tflite"  # Path to your TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.info(f"Loaded TensorFlow Lite model from '{model_path}'.")
except Exception as e:
    logger.error(f"Failed to load TensorFlow Lite model: {e}")
    sys.exit(1)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info(f"Input Details: {input_details}")
logger.info(f"Output Details: {output_details}")

# Path to the image
image_path = '../bus.jpg'  # Update this path as needed
output_image_path = "results_yolo_tflite.jpg"
output_text_path = "tflite_inference_times.txt"
label_map_path = 'label_map.txt'  # Path to your label map

# Function to load label map
def load_label_map(label_map_path):
    label_map = {}
    current_id = None

    try:
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
    except FileNotFoundError:
        logger.error(f"Label map file '{label_map_path}' not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading label map: {e}")
        sys.exit(1)

    return label_map

# Load label map
label_map = load_label_map(label_map_path)
logger.info(f"Label Map Loaded: {label_map}")

# Load and preprocess the image
image_np = cv2.imread(image_path)

# Check if the image was loaded successfully
if image_np is None:
    logger.error(f"Could not load image at path '{image_path}'. Please check the file path and integrity.")
    sys.exit(1)
else:
    logger.info(f"Loaded image '{image_path}' with shape {image_np.shape}.")

def preprocess_image(image):
    """
    Preprocess the image to fit the input requirements of the TensorFlow Lite model.
    """
    try:
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        resized_image = cv2.resize(image, (width, height))
        input_data = np.expand_dims(resized_image, axis=0)
        input_data = input_data.astype(np.float32)

        # If the model requires normalization, uncomment the following line
        # input_data = input_data / 255.0

        return input_data
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        sys.exit(1)

def postprocess_and_annotate(image, boxes, classes, scores, label_map, threshold=0.5):
    """
    Post-process the model output to extract bounding boxes, class IDs, and scores,
    and annotate the image with them.
    """
    height, width, _ = image.shape

    # Iterate over all detections
    for i in range(len(scores)):
        if scores[i] >= threshold:
            # Extract box coordinates and scale them back to the image size
            ymin, xmin, ymax, xmax = boxes[i]
            startX, startY, endX, endY = (
                int(xmin * width),
                int(ymin * height),
                int(xmax * width),
                int(ymax * height)
            )
            
            # Class ID and score
            class_id = int(classes[i])
            score = scores[i]

            # Get the label from the label map
            class_id_str = str(class_id+1)
            label = f"{label_map.get(class_id_str, 'N/A')}: {score:.2f}"

            # Draw the bounding box
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Draw label background
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(startY, label_size[1] + 10)
            cv2.rectangle(
                image,
                (startX, label_ymin - label_size[1] - 10),
                (startX + label_size[0], label_ymin + base_line - 10),
                (0, 255, 0),
                cv2.FILLED
            )

            # Put the label text above the bounding box
            cv2.putText(
                image, label,
                (startX, label_ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    return image

# Preprocess the image for the model
input_data = preprocess_image(image_np)

# Run inference multiple times and record times
num_runs = 11  # We will ignore the first run
inference_times = []

for i in range(num_runs):
    start_time = time.time()
    
    try:
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        if i > 0:  # Ignore the first run
            inference_times.append(inference_time)
            logger.info(f"Run {i}/{num_runs-1} - Inference Time: {inference_time:.4f} seconds")
            
            # Get the output tensors
            # The exact order and meaning of outputs depend on your model
            # Here, we assume output_details[0] = boxes, output_details[1] = classes, output_details[2] = scores
            try:
                boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
                classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
                scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
            except IndexError as e:
                logger.error(f"Error accessing output tensors: {e}")
                sys.exit(1)
            
            # Post-process and annotate the image only for the last run
            if i == num_runs - 1:
                image_np = postprocess_and_annotate(image_np, boxes, classes, scores, label_map, threshold=0.5)
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        sys.exit(1)

# Calculate and print average inference time and FPS
avg_inference_time = sum(inference_times) / len(inference_times)
fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
logger.info(f"Average Inference Time: {avg_inference_time:.4f} seconds, FPS: {fps:.2f}")

# Save inference times and FPS to a file
try:
    with open(output_text_path, 'w') as f:
        for idx, t in enumerate(inference_times, 1):
            f.write(f"Inference {idx}: {t:.4f} seconds\n")
        f.write(f"Average Inference Time: {avg_inference_time:.4f} seconds\n")
        f.write(f"FPS: {fps:.2f}\n")
    logger.info(f"Inference times and FPS saved to '{output_text_path}'.")
except Exception as e:
    logger.error(f"Failed to write inference times to file: {e}")

# Save the annotated image (from the last run)
try:
    cv2.imwrite(output_image_path, image_np)
    logger.info(f"Annotated image saved to '{output_image_path}'.")
except Exception as e:
    logger.error(f"Failed to save annotated image: {e}")
