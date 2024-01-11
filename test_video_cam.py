# 1) load the model
import models
import cv2
from PIL import Image
import torch.optim
import torch.nn.parallel
from torch.nn import functional as F
import os
import argparse
import moviepy.editor as mpy
from utils import extract_frames, load_frames, render_frames

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 1.2
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

model = models.load_model('resnet3d50')

categories = models.load_categories('category_momentsv2.txt')

# Load the video frame transform
transform = models.load_transform()

# Open the device at the ID 0
# Use the camera ID based on
# /dev/videoID needed
cap = cv2.VideoCapture(0)

#Check if camera was opened correctly
if not (cap.isOpened()):
    print("Could not open video device")

# 2) fetch one frame at a time from your camera
while(True):
    
    # frame is a numpy array, that you can predict on 
    ret, frame = cap.read()

    # convert your frame in PIL Image format
    frameimg = Image.fromarray(frame)

    input = torch.stack([transform(frameimg)], 1).unsqueeze(0)
        
    # Make video prediction
    with torch.no_grad():
        logits = model(input)
        h_x = F.softmax(logits, 1).mean(dim=0)
        probs, idx = h_x.sort(0, True)
    prediction = categories[idx[0]]
    # 4) Adding the label on your frame
    __draw_label(frame, 'Label: {}'.format(prediction), (20,20), (255,0,0))

    # 5) Display the resulting frame
    cv2.imshow("preview",frame)
   
    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
