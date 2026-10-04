from ultralytics import YOLO


def main():
  # ផ្ទុកម៉ូដែលល្អបំផុតពីលើកមុន
  model = YOLO("runs/detect/train-2/weights/best.pt")

  # Train បន្ថែមលើរូបភាពចាស់ + រូបភាពថ្មីដែលបានកែរួច
  model.train(
      data="dataset/Car_PlateSample/data.yaml",
      epochs=20,
      imgsz=640,
      batch=8,
      device="cpu", 
      # មិនបាច់ដាក់ hsv_v=0.6 ឬ perspective ខ្ពស់ទៀតទេ ទុកឱ្យវាប្រើកម្រិតស្តង់ដារល្មមៗ
      name="fine_tune_plate",
  )


if __name__ == "__main__":
  main()
  