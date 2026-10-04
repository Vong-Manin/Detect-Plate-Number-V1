import cv2
from ultralytics import YOLO

# 1. ផ្ទុកទម្ងន់ម៉ូដែលថ្មីចុងក្រោយពី fine_tune_plate-2
model = YOLO(r"runs/detect/fine_tune_plate-2/weights/best.pt")

# 2. ដាក់ឈ្មោះ ឬផ្លូវទៅកាន់រូបភាពឡាន Prius ពេលយប់របស់អ្នក
image_path = "image copy.png"  # ប្តូរប្រសិនបើរូបភាពរបស់អ្នកមានឈ្មោះផ្សេង (ឧ. prius_night.jpg)

# ដំណើរការ Detection (អាចកំណត់ conf=0.25 ដើម្បីឱ្យចាប់បានកាន់តែទូលាយ)
results = model(image_path, conf=0.30)
# 3. គូរប្រអប់ បង្ហាញពិន្ទុ និង Save រូបភាព
for r in results:
  # បង្ហាញពិន្ទុ Confidence លើ Terminal
  boxes = r.boxes
  print(f"\nរកឃើញផ្លាកលេខចំនួន: {len(boxes)}")
  for box in boxes:
    conf = float(box.conf[0])
    print(f"Confidence Score ថ្មី: {conf:.4f} ({conf * 100:.1f}%)")

  annotated_frame = r.plot()

  # រក្សាទុករូបភាពលទ្ធផលទៅជា result.jpg
  cv2.imwrite("result.jpg", annotated_frame)
  print("ដំណើរការជោគជ័យ! បានរក្សាទុករូបភាពទៅក្នុង result.jpg")

  # បង្ហាញផ្ទាំង Popup លើអេក្រង់
  cv2.imshow("License Plate Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()