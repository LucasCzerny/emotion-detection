import torch
from torchvision import transforms

import cv2

from config import *

model = torch.load(...)
image_transform = transforms.Compose([
    transforms.Resize((NN_IMAGE_SIZE, NN_IMAGE_SIZE)),
    transforms.ToTensor()
])

webcam_stream = cv2.VideoCapture(0) 

while True:
    _, frame = webcam_stream.read()
    cv2.imshow('frame', frame) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break