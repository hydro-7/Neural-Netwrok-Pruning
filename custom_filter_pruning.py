import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# --------------------------------------------- PRUNING --------------------------------------------------- #

@staticmethod
def PRUNE(layer, idx_to_rem):

    idx_to_rem = sorted(idx_to_rem) 
    idx_to_rem.reverse() # desc order

    weights = layer.weight.data

    print(f"\ninitial weights shape : {weights.shape}")

    for idx in idx_to_rem:
        weights = torch.cat([weights[:idx], weights[idx+1:]])

    print(f"final weights shape : {weights.shape}\n") # 3 channel hata di, as seen in shape[0]

    pruned_conv = nn.Conv2d( in_channels = weights.shape[1], out_channels = weights.shape[0], kernel_size = layer.kernel_size ) # its asking for kernel size in an error, so included

    pruned_conv.weight.data = weights # pruned weights -> new weights
    
    return pruned_conv

# --------------------------------------------- SELECTING PRUNABLE CHANNELS --------------------------------------------------- #

@staticmethod
def get_ch_idx(conv_layer, pruning_ratio = 0.2):
    
    channel_norms = torch.norm(conv_layer.weight.data.view(conv_layer.out_channels, -1), 
                                p=1, dim=1) 
    
    # l1 norm 
    # the first input resizes the tensor into (out_channels) filters of in * kx * ky size
    # [out, in, kx, ky] -> [out, in * kx * ky]
    # p = 1 -> calc norm across single dim
    # dim = 1 -> which dim to calc norm for 

    # print(conv_layer.weight.shape)

    sorted_indices = torch.argsort(channel_norms) 
    num_channels = conv_layer.out_channels
    num_prune = int(num_channels * pruning_ratio)
    
    return sorted_indices[:num_prune].tolist() # select the lowest l1 vals to prune


# --------------------------------------------- YOLOV8 PRUNING --------------------------------------------------- #


def prune_yolov8_model(model_path, pruning_ratio, save_path = None):
    
    yolo_model = YOLO(model_path)
    layer_to_prune = yolo_model.model.model[0].conv # picked first conv 2d layer 
    
    channels_to_remove = get_ch_idx( layer_to_prune, pruning_ratio = pruning_ratio )
    layer_after_pruning = PRUNE( layer_to_prune, channels_to_remove )
    
    yolo_model.model.model[0].conv = layer_after_pruning # replace w pruned layr
    
    print(f"Pruned {len(channels_to_remove)} filters from the given layer")
    print(f"Original number of filters: {layer_to_prune.out_channels}")
    print(f"Remaining number of filters: {layer_after_pruning.out_channels}")
    
    if save_path is not None:
        yolo_model.save(save_path)
    
    return yolo_model


# --------------------------------------------------------------- MAIN --------------------------------------------------------------- #



model_path = "yolov8n.pt"  
    
pruned_model = prune_yolov8_model(
    model_path = model_path, 
    pruning_ratio = 0.2, 
    save_path = "pruned_yolov8.pt"
)  

model = YOLO("pruned_yolov8.pt")

# image = "Zentree_Labs\stop.jpg"
# results = model(image)
# annotated_image = results[0].plot()  
# cv2.imwrite("Zentree_Labs\stop_res.jpg", annotated_image)

# plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

# yolo_model = YOLO(model_path)
# print(yolo_model.model.model[1].conv)