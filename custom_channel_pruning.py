import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# --------------------------------------------- PRUNING --------------------------------------------------- #

@staticmethod
def PRUNE(layer, idx_to_rem):

    weights = layer.weight.data
    
    # new_weights_tensor = torch.tensor([], dtype=torch.float32)
    new_weights_tensor = []

    # print(weights) # size of 16 x 3 x 3 x 3, we need to remove values from weight[1]
    print(f"\ninitial weights shape : {weights.shape}")

    for i in range(0, weights.shape[0]):
        idx = idx_to_rem[i] % 3
        updated_filter = torch.cat([weights[i][:idx], weights[i][idx+1:]])
        new_weights_tensor.append(updated_filter)

    new_weights_tensor = torch.stack(new_weights_tensor)

    print(f"final weights shape : {new_weights_tensor.shape}\n") # 3 channel hata di, as seen in shape[0]

    pruned_conv = nn.Conv2d( in_channels = new_weights_tensor.shape[1], out_channels = new_weights_tensor.shape[0], kernel_size = layer.kernel_size ) # its asking for kernel size in an error, so included

    pruned_conv.weight.data = new_weights_tensor # pruned weights -> new weights
    
    return pruned_conv

# --------------------------------------------- SELECTING PRUNABLE CHANNELS --------------------------------------------------- #

@staticmethod
def get_ch_idx(conv_layer, pruning_ratio = 0.2):
    
    smallest_indices_tensor = torch.tensor([], dtype=torch.long)

    for i in range(0, conv_layer.out_channels):

        channel_norms = torch.norm(conv_layer.weight[i].view(conv_layer.in_channels, -1), p = 1, dim = 1)
        sorted_indices = torch.argsort(channel_norms) 
        smallest_index_with_offset = sorted_indices[:1] + (i * 3)
    
        # Append the modified value to the tensor
        smallest_indices_tensor = torch.cat((smallest_indices_tensor, smallest_index_with_offset))
        # smallest_indices_tensor = torch.cat((smallest_indices_tensor, sorted_indices[:1]))


    return smallest_indices_tensor # tensor of 16 values, each telling which channel to remove from the ith filter 

    # filter_norms = torch.norm(conv_layer.weight.data.view(conv_layer.out_channels, -1), 
    #                             p=1, dim=1) # L1 norm values for the whole filter 
    
    # l1 norm 
    # the first input resizes the tensor into (out_channels) filters of in * kx * ky size
    # [out, in, kx, ky] -> [out, in * kx * ky]
    # p = 1 -> calc norm across single dim
    # dim = 1 -> which dim to calc norm for 

    
    # sorted_indices = torch.argsort(filter_norms) 
    # num_channels = conv_layer.out_channels
    # num_prune = int(num_channels * pruning_ratio)

    # return sorted_indices[:num_prune].tolist() # select the lowest l1 vals to prune


# --------------------------------------------- YOLOV8 PRUNING --------------------------------------------------- #


def prune_yolov8_model(model_path, pruning_ratio, save_path = None):
    
    yolo_model = YOLO(model_path)
    layer_to_prune = yolo_model.model.model[1].conv # picked first conv 2d layer 
    
    channels_to_remove = get_ch_idx( layer_to_prune, pruning_ratio = pruning_ratio )
    layer_after_pruning = PRUNE( layer_to_prune, channels_to_remove )
    
    yolo_model.model.model[1].conv = layer_after_pruning # replace w pruned layr
    
    print(f"Pruned {len(channels_to_remove)} channels from the given layer")
    print(f"Original number of channels: {layer_to_prune.in_channels * layer_to_prune.out_channels}")
    print(f"Remaining number of channels: {layer_after_pruning.in_channels * layer_to_prune.out_channels}")
    
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

image = "Zentree_Labs\stop.jpg"
results = model(image)
annotated_image = results[0].plot()  
cv2.imwrite("Zentree_Labs\stop_res.jpg", annotated_image)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

yolo_model = YOLO(model_path)
print(yolo_model.model.model[1].conv)