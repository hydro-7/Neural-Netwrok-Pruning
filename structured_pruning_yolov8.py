import torch
from ultralytics import YOLO
from torch.nn.utils import prune
import cv2
from matplotlib import pyplot as plt

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

model = YOLO("yolov8n.pt")

# print(model) # list of all the modules present in the YOLOv8 model
# print(model.info(detailed= True)) # detailed info on the interiors of these modules

# for name, layer in model.named_modules(): 
#     print(name)

for name, module in model.named_modules():
    if name == "model.model.0.conv":
        layer = module
        break

# a = layer.weight
# print(layer.weight)

prune.ln_structured(layer, name='weight', amount = 0.5, dim = 1, n = float('-inf')) # ln_structured

# layer = prune.random_structured(layer, name = 'weight', amount = 0.1, dim = 1) # random_structured

# columns_pruned = int(torch.sum(torch.sum(layer.weight, dim=1) == 0).item())
# print(f"columns pruned : {columns_pruned}")

# b = layer.weight
# print("\n\n")
# print(a == b)
# print(layer.weight_mask)
# pruned_weight = layer.weight_mask
# pruned_params = pruned_weight.numel() - pruned_weight.nonzero(as_tuple=False).size(0)
# print(pruned_params)

image = "Zentree_Labs\stop.jpg"
results = model(image)
annotated_image = results[0].plot()  
cv2.imwrite("Zentree_Labs\stop_res.jpg", annotated_image)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


# import torch
# import torch.nn.utils.prune as prune

# t = torch.randn(100, 100)
# torch.save(t, 'full.pth')

# p = prune.L1Unstructured(amount=0.9)
# pruned = p.prune(t)
# torch.save(pruned, 'pruned.pth')

# sparsified = pruned.to_sparse()
# torch.save(sparsified, 'sparsified.pth')