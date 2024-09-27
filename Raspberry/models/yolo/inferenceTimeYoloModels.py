import time
import cv2
from ultralytics import YOLO
image_path = '../testImage.jpg'
# Paths to the model and image
name = 'yolov5nd_red_ncnc'
model_path = "yolov5nd_reduced_ncnn_model"  
output_image_path = f"results_yolo_{name}.jpg"
output_text_path = f"yolo_{name}.txt"

# Load the YOLO model
model = YOLO(model_path)

# Load and preprocess the image
image_np = cv2.imread(image_path)

# Run inference multiple times and record times
num_runs = 11  # Run one extra for the initial ignored run
inference_times = []

for i in range(num_runs):
    start_time = time.time()
    results = model(image_np)  # Run inference
    end_time = time.time()

    inference_time = end_time - start_time

    if i > 0:  # Ignore the first run
        inference_times.append(inference_time)
        print(f"Run {i}/{num_runs-1} - Inference Time: {inference_time:.4f} seconds")

# Calculate and print average inference time and FPS
avg_inference_time = sum(inference_times) / len(inference_times)
fps = 1 / avg_inference_time
print(f"Average Inference Time: {avg_inference_time:.4f} seconds, FPS: {fps:.2f}")

# Save inference times and FPS to a file
with open(output_text_path, 'w') as f:
    for idx, t in enumerate(inference_times, 1):
        f.write(f"Inference {idx}: {t:.4f} seconds\n")
    f.write(f"Average Inference Time: {avg_inference_time:.4f} seconds\n")
    f.write(f"FPS: {fps:.2f}\n")
print(f"Inference times and FPS saved to '{output_text_path}'")

# Annotate and save the image with detections
annotated_image = results[0].plot()
cv2.imwrite(output_image_path, annotated_image)
print(f"Annotated image saved to '{output_image_path}'")
