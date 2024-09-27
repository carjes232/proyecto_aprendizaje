from ultralytics import YOLO
import torch


# Load a model onto the CPU
model = YOLO("yolov5nd_reduced.pt")

# Optionally, export the model to NCNN format
model.export(format="ncnn", half =True)

