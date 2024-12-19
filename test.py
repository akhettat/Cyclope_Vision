import time
import torch
from ultralytics import YOLO 

print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    
model = YOLO("runs/train/custom_yolov11_training_v2/weights/best.pt")
#model("C:/Users/esteb/Desktop/Dataset/photoavecschematrapeze-directsignaletique-1024x655.jpg")
time1 = time.time_ns()
results = model("runs/train/custom_yolov11_training/les-personnes-qui-traversent-la-route-a-un-passage-cloute-sur-london-wall-city-of-london-uk-mmx975.jpg")
time2 = time.time_ns()
duration_ms = (time2 - time1) / 1e6
print(f"Temps d'inférence : {duration_ms:.2f} ms")
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")