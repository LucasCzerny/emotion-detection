import torch
from torchvision import transforms

import cv2

from model import EmotionDetectorModel
from config import *

model = EmotionDetectorModel()
state = torch.load(f'{MODEL_OUTPUT_PATH}/model_0_20240509_221615')
model.load_state_dict(state)

face_cascade = cv2.CascadeClassifier('opencv/face_detection_cascade.xml')

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((NN_IMAGE_SIZE, NN_IMAGE_SIZE)),
    transforms.ToTensor()
])

webcam_stream = cv2.VideoCapture(1) 

while True:
    _, frame = webcam_stream.read()

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        transformed_image = image_transform(face_img)
        predicted = torch.argmax(model(transformed_image))

        cv2.putText(frame, LABELS[predicted], (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Emotion Dectection', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break