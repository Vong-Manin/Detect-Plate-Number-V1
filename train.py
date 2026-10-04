from ultralytics import YOLO

def main():
    # Load the official YOLOv8 Nano base weights
    model = YOLO("yolov8n.pt")

    # Run training using your new folder path
    model.train(
        data="data/Car_PlateSample/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu"
    )

if __name__ == '__main__':
    main()