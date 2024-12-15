from ultralytics import YOLO
from torch.nn.utils import prune
import cv2
from matplotlib import pyplot as plt

model = YOLO("yolov8n.pt")

# print(model) # list of all the modules present in the YOLOv8 model
# print(model.info(detailed= True)) # detailed info on the interiors of these modules

# for name, layer in model.named_modules(): 
#     print(name)

for name, module in model.named_modules():
    if name == "model.model.6.m.1.cv1.conv":
        layer = module
        break


prune.l1_unstructured(layer, name='weight', amount = 0.6)
prune.remove(layer, 'weight')


image = "Zentree_Labs\stop.jpg"
results = model(image)
annotated_image = results[0].plot()  
cv2.imwrite("Zentree_Labs\stop_res.jpg", annotated_image)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

