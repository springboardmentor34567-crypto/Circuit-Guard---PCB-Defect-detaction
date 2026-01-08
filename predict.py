from ultralytics import YOLO

model = YOLO(r"C:\Users\asus\OneDrive\Desktop\yolo by ultralytics\runs\detect\train10\weights\best.pt")
model.predict(
    source=r"C:\Users\asus\OneDrive\Desktop\yolo by ultralytics\data\images\val",
    save=True,
    project="runs\detect",
    name="val_predictions",
    exist_ok=True
)
