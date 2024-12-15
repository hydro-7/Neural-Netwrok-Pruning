import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# --------------------------------------------- PRUNING CHANNELS --------------------------------------------------- #

@staticmethod
def PRUNE_CHANNELS(layer, idx_to_rem):

    weights = layer.weight.data
    new_weights_tensor = []

    print(f"\ninitial weights shape : {weights.shape}")

    for i in range(0, weights.shape[0]):
        idx = idx_to_rem[i] % 3
        updated_filter = torch.cat([weights[i][:idx], weights[i][idx+1:]])
        new_weights_tensor.append(updated_filter)

    new_weights_tensor = torch.stack(new_weights_tensor)

    print(f"final weights shape : {new_weights_tensor.shape}\n") # 3 channel hata di, as seen in shape[0]

    pruned_conv = nn.Conv2d( in_channels = new_weights_tensor.shape[1], out_channels = new_weights_tensor.shape[0], kernel_size = layer.kernel_size, bias = False ) # its asking for kernel size in an error, so included

    pruned_conv.weight.data = new_weights_tensor # pruned weights -> new weights
    
    return pruned_conv


# --------------------------------------------- PRUNING FILTERS --------------------------------------------------- #

@staticmethod
def PRUNE_FILTERS(layer, idx_to_rem):

    idx_to_rem = sorted(idx_to_rem) 
    idx_to_rem.reverse() # desc order

    weights = layer.weight.data

    print(f"\ninitial weights shape : {weights.shape}")

    for idx in idx_to_rem:
        weights = torch.cat([weights[:idx], weights[idx+1:]])
        break

    print(f"final weights shape : {weights.shape}\n") # 3 channel hata di, as seen in shape[0]

    pruned_conv = nn.Conv2d( in_channels = weights.shape[1], out_channels = weights.shape[0], kernel_size = layer.kernel_size, bias=False ) # its asking for kernel size in an error, so included

    pruned_conv.weight.data = weights # pruned weights -> new weights
    
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

    return smallest_indices_tensor 


# --------------------------------------------- SELECTING PRUNABLE FILTERS --------------------------------------------------- #

@staticmethod
def get_filter_idx(conv_layer, pruning_ratio = 0.2):
    
    channel_norms = torch.norm(conv_layer.weight.data.view(conv_layer.out_channels, -1), 
                                p=1, dim=1) 
    
    # l1 norm 
    # the first input resizes the tensor into (out_channels) filters of in * kx * ky size
    # [out, in, kx, ky] -> [out, in * kx * ky]
    # p = 1 -> calc norm across single dim
    # dim = 1 -> which dim to calc norm for 

    # print(conv_layer.weight.shape)

    sorted_indices = torch.argsort(channel_norms) 
    
    return sorted_indices[:1].tolist() # select the lowest l1 vals to prune

# --------------------------------------------- YOLOV8 PRUNING --------------------------------------------------- #


def prune_yolov8_model(model_path, pruning_ratio, save_path = None):
    
    yolo_model = YOLO(model_path)
    layer_to_prune = yolo_model.model.model[1].conv # picked layer1's conv2d block
    
    channels_to_remove = get_ch_idx( layer_to_prune, pruning_ratio = pruning_ratio )
    layer_after_pruning = PRUNE_CHANNELS( layer_to_prune, channels_to_remove )
    
    yolo_model.model.model[1].conv = layer_after_pruning # replace w pruned layr
    
    print(f"Pruned {len(channels_to_remove)} channels from the given layer")
    print(f"Original number of channels: {layer_to_prune.in_channels * layer_to_prune.out_channels}")
    print(f"Remaining number of channels: {layer_after_pruning.in_channels * layer_to_prune.out_channels}")

    # now we have pruned all the channels out of the 1st layer, so we have to adjust the size of the previous block of (C - B - S)
    # to adjust the previous (C - B - S) block, remove the filter with the lowest L1 score out of the previous layer
    # also, within the batch normalization block of that particular layer, edit the running_mean and running_var values according to the previously calculated index value
    # remove these running_mean[idx] & running_var[idx] values

    previous_layer = yolo_model.model.model[0].conv # layer0's conv2d block

    filter_to_remove = get_filter_idx( previous_layer, pruning_ratio = pruning_ratio )
    previous_layer_after_pruning = PRUNE_FILTERS( previous_layer, filter_to_remove )

    yolo_model.model.model[0].conv = previous_layer_after_pruning

    print("Pruned 1 filter from the previous layer")
    print(f"Original number of filters : {previous_layer.out_channels}")
    print(f"Remaining number of filters : {previous_layer_after_pruning.out_channels}\n")


    filter_index = filter_to_remove[0]

    # ----------------------- PRUNING BN - RUNNING MEAN & VAR ---------------------------------------- #

    running_mean_of_prev_layer = yolo_model.model.model[0].bn.running_mean
    running_var_of_prev_layer = yolo_model.model.model[0].bn.running_var

    running_mean_of_prev_layer = torch.cat([running_mean_of_prev_layer[:filter_index], running_mean_of_prev_layer[filter_index + 1:]])
    running_var_of_prev_layer = torch.cat([running_var_of_prev_layer[:filter_index], running_var_of_prev_layer[filter_index + 1:]])
    
    # print(running_mean_of_prev_layer.shape)
    yolo_model.model.model[0].bn.running_mean = running_mean_of_prev_layer
    yolo_model.model.model[0].bn.running_var = running_var_of_prev_layer

    # ----------------------- PRUNING BN - BIAS ---------------------------------------- #

    print(f"Bias size before pruning: {yolo_model.model.model[0].bn.bias.shape[0]}")

    bias_of_prev_layer = yolo_model.model.model[0].bn.bias
    bias_of_prev_layer = torch.cat([bias_of_prev_layer[:filter_index], bias_of_prev_layer[filter_index + 1:]])
    yolo_model.model.model[0].bn.bias = nn.Parameter(bias_of_prev_layer)

    print(f"Bias size after pruning: {yolo_model.model.model[0].bn.bias.shape[0]}\n")

    # ----------------------- PRUNING BN - WEIGHT ---------------------------------------- #

    bnweight_of_prev_layer = yolo_model.model.model[0].bn.weight
    bnweight_of_prev_layer = torch.cat([bnweight_of_prev_layer[:filter_index], bnweight_of_prev_layer[filter_index + 1:]])
    yolo_model.model.model[0].bn.weight = nn.Parameter(bnweight_of_prev_layer)

    yolo_model.model.model[0].bn.num_features = 15
    
    # --------- CHANGING THE STRIDE, PADDING & BIAS PARAMETERS IN THE CREATED LAYERS ----------- #

    yolo_model.model.model[0].conv.stride = (2, 2)
    yolo_model.model.model[1].conv.stride = (2, 2)
    yolo_model.model.model[0].conv.padding = (1, 1)
    yolo_model.model.model[1].conv.padding = (1, 1)

    # ------------------------------------------------------------------------------------------- #

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

# print((model.model.model[0].conv.bias))

# print(model)

image = "Zentree_Labs\stop.jpg"
results = model(image)
annotated_image = results[0].plot()  
cv2.imwrite("Zentree_Labs\stop_res.jpg", annotated_image)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()