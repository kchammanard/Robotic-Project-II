from glob import glob
import os
from ultralytics import YOLO
import cv2
import string
PATH_TO_MODEL = 'yolov8/training_results/sign_language/weights/best.pt'
model = YOLO(PATH_TO_MODEL)

classes = dict( (i, key) for i,key in enumerate(string.ascii_lowercase))

def predict_image():
    folder = "imglib/"
    images = glob(os.path.join(folder,"*jpg"))
    for image in images:
        img = cv2.imread(image)
        results = model.predict(source = img, conf = 0.5)[0]
        if results.boxes:
            for i,obj in enumerate(results.boxes):
                x1, y1, x2, y2, conf, cls = obj.data.cpu().detach().numpy()[0]
                print(classes[int(cls)])
        else:
            print("NULL")


if __name__== "__main__":
    predict_image()
    
