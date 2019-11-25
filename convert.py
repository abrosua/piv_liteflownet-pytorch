import os
import torch
from collections import OrderedDict

from src.models import LiteFlowNet


# Write layers name in csv
def layer_csv(title, param_dict, shape_only=True):
    with open(title, 'w') as f:
        f.write("layers,params_shape\n")
        for key in param_dict.keys():
            if shape_only:
                val = list(param_dict[key].size())
            else:
                val = list(param_dict[key])

            f.write("%s,%s\n"%(key, str(val)))


# Renaming the keys
def renameKeys(source: torch.nn.Module.state_dict, target: str) -> torch.nn.Module.state_dict:
    new_key = list(source)
    state = torch.load(target)
    new_state = OrderedDict()

    i = 0
    for key, value in state.items():
        new_state[new_key[i]] = value
        i += 1

    return new_state

# Call the functions
if __name__ == '__main__':

    ## DIRECTORY
    inputdir = './models/caffe'
    outputdir = './models/torch'
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    ## INPUT
    # param_in = os.path.join(inputdir, 'Caffe_Hui-LFN.paramOnly')
    param_in = os.path.join(inputdir, 'Caffe_PIV-LFN.paramOnly')

    ## OUTPUT
    # param_out = os.path.join(outputdir, 'Hui-LiteFlowNet.paramOnly')
    param_out = os.path.join(outputdir, 'PIV-LiteFlowNet-en.paramOnly')

    ## Converting using MANUAL converter
    # net_torch = LiteFlowNet()  # Hui-LiteFlowNet
    net_torch = LiteFlowNet(starting_scale=10.0, lowest_level=1)  # PIV-LiteFlowNet-en
    lfn_torch = renameKeys(net_torch.state_dict(), param_in)  # renaming the keys, according to the PyTorch model

    # Saving and verifying
    torch.save(lfn_torch, param_out)
    net_torch.load_state_dict(lfn_torch)

    print('DONE!')
