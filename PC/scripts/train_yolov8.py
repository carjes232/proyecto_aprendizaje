from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt") 
    results = model.train(data="dataYoloV8Train.yaml", epochs=50, imgsz=640)