from ultralytics import YOLO
from PIL import Image

if __name__ == "__main__":
    weight = './weights/detection/yolov8n_best.pt'
    datasetPath = r"D:\USERPROG\Pyproject\ultralytics\datasets3\dataset.yaml"
    model = YOLO(weight)
    result = model.val(data=datasetPath)
    print(result)
    print("test")
