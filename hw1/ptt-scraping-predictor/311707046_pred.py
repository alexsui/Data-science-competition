import torchvision
import torch
from torchvision import transforms
import torch.nn as nn
import cv2
from  pathlib import Path
from PIL import Image
import sys
param_path ="./model_param.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.resnet101()
model.fc = nn.Linear(2048,1)
model.load_state_dict(torch.load(param_path,map_location=device))
model.to(device)
test_transform = transforms.Compose([
    transforms.Resize(size=(232,232)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

image_file = sys.argv[1]
output_file = "311707046.txt"
image_list = []
output_list = []
with open(image_file,"r") as f:
    for l in f:
        image_list.append(l.strip())
for image in image_list:
    model.eval()
    with torch.inference_mode():
        #crop face
        img = cv2.imread(str(image))
        # crop face from image
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_detector.detectMultiScale(img)
        if len(faces)==1:
            x, y, w, h = faces[0]
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face_bbox = (x, y, x+w, y+h)
            target_img = pil_img.crop(face_bbox)

        else:
            target_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_transform = test_transform(target_img)
        pred = torch.sigmoid(model(image_transform.unsqueeze(0)))
        output_list.append(int(torch.round(pred).item()))
output = "".join(map(str,output_list))
with open(output_file,'w') as f:
    f.write(output)