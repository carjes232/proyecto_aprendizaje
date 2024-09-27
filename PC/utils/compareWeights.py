import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import time

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, img_size=640):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    return image

# Function to perform inference and measure time
def perform_inference(model, image):
    # Ensure the model and image are on the CPU
    model.cpu()
    image = image.cpu()

    # Measure inference time
    start_time = time.time()
    results = model(image)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Inference time on CPU: {inference_time:.4f} seconds")
    
    return results

# Function to plot detections and save to a file
def plot_detections(image, results, confidence_threshold=0.25, output_path='results.jpg'):
    # Convert PIL image to NumPy array
    img = np.array(image)
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # Move to CPU and convert to NumPy array
    
    for *box, score, class_idx in detections:
        if score < confidence_threshold:
            continue  # Skip detections with low confidence
        
        x1, y1, x2, y2 = box
        label = results.names[int(class_idx)]
        confidence = score.item()
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1 - 10, f"{label} {confidence:.2f}", 
                color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # Load the YOLOv5 Nano model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()
    
    # Path to your image
    image_path = 'bus.jpg'  # Replace with your image path
    
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Perform inference and measure time
    results = perform_inference(model, image)
    
    # Plot the detections and save the output image
    plot_detections(image, results, output_path='results.jpg')
    print("Results saved to 'results.jpg'")

if __name__ == "__main__":
    main()
