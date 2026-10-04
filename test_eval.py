from ultralytics import YOLO


def run_test():
  # 1. Load your best trained weights
  model = YOLO("runs/detect/fine_tune_plate-2/weights/best.pt")

  # 2. Run evaluation specifically on the test split
  metrics = model.val(
      data="dataset/Car_PlateSample/data.yaml",
      split="test",  # This tells YOLO to evaluate on the test folder
      imgsz=640,
      batch=8,
      device="cpu",  # Set to 0 if you are using GPU
  )

  # 3. Print the benchmark results
  print("\n--- Final Test Dataset Results ---")
  print(f"Precision:  {metrics.box.mp:.4f}")
  print(f"Recall:     {metrics.box.mr:.4f}")
  print(f"mAP@50:     {metrics.box.map50:.4f}")
  print(f"mAP@50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
  run_test()