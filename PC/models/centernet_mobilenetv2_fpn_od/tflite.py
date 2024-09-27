import tensorflow as tf
import numpy as np
import cv2
import time
import re

# Paths to the model, label map, and image
model_path = "centernet_mobilenetv2_fpn_od/model.tflite"
label_map_path = "centernet_mobilenetv2_fpn_od/label_map.txt"
image_path = "Bus.jpg"
output_image_path = "results_tf.jpg"

# Load the TensorFlow Lite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label map (robust parsing)
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

# Load the image
image_np = cv2.imread(image_path)
original_height, original_width, _ = image_np.shape

# Prepare the input tensor
input_size = input_details[0]['shape'][1:3]  # e.g., (320, 320)
input_tensor = cv2.resize(image_np, (input_size[1], input_size[0]))  # Resize to model's expected size
input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = input_tensor.astype(np.float32)

# Run inference 5 times and record times
num_runs = 10
inference_times = []

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}")
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    inference_times.append(inference_time)

# Calculate average inference time
avg_inference_time = sum(inference_times) / num_runs
print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

# Save inference times to file
with open('tflite_times.txt', 'w') as f:
    for idx, t in enumerate(inference_times, 1):
        f.write(f"Inference {idx}: {t:.4f} seconds\n")
    f.write(f"Average Inference Time: {avg_inference_time:.4f} seconds\n")
print("Inference times saved to 'tflite_times.txt'")

# Extract detection results from the last run
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

# Threshold for displaying detections
threshold = 0.3

# Process each detection
for i in range(len(scores)):
    if scores[i] >= threshold:
        ymin, xmin, ymax, xmax = boxes[i]
        (startX, startY, endX, endY) = (
            int(xmin * original_width),
            int(ymin * original_height),
            int(xmax * original_width),
            int(ymax * original_height)
        )

        # Adjust class ID by adding 1 to match label_map IDs
        class_id = int(classes[i]) + 1
        class_id_str = str(class_id)

        # Debug: Print class ID and label
        print(f"Detection {i}: Class ID from model: {classes[i]} mapped to Label Map ID: {class_id_str} -> {label_map.get(class_id_str, 'N/A')}")

        # Draw bounding box
        cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Draw label and score
        label = f"{label_map.get(class_id_str, 'N/A')}: {scores[i]:.2f}"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        startY_label = max(startY, label_size[1])
        cv2.rectangle(
            image_np,
            (startX, startY_label - label_size[1]),
            (startX + label_size[0], startY_label + base_line),
            (0, 255, 0),
            cv2.FILLED
        )
        cv2.putText(
            image_np, label,
            (startX, startY_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

# Save the annotated image
cv2.imwrite(output_image_path, image_np)
print(f"Annotated image saved to {output_image_path}")
