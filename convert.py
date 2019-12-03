import os
import re
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

    # Init.
    i = 0
    misc = {}

    # Start iterating over the target
    for key, value in state.items():
        if not (bool(re.search('weight', key)) or
                bool(re.search('bias', key)) or
                isinstance(value, torch.Tensor)):
            misc[key] = value
            continue

        new_state[new_key[i]] = value
        i += 1

    return new_state, misc

# Call the functions
if __name__ == '__main__':

    ## DIRECTORY
    inputdir = './models/pretrain_caffe'
    outputdir = './models/pretrain_torch'
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    ## INPUT
    param_in = os.path.join(inputdir, 'Caffe_Hui-LFN_aug.paramOnly')
    # param_in = os.path.join(inputdir, 'Caffe_PIV-LFN_aug.paramOnly')

    ## OUTPUT
    param_out = os.path.join(outputdir, 'Hui-LiteFlowNet.paramOnly')
    # param_out = os.path.join(outputdir, 'PIV-LiteFlowNet-en.paramOnly')

    ## Converting using MANUAL converter
    # net_torch = LiteFlowNet()  # Hui-LiteFlowNet
    net_torch = LiteFlowNet(starting_scale=10, lowest_level=1)  # PIV-LiteFlowNet-en
    lfn_torch, extra_param = renameKeys(net_torch.state_dict(), param_in)  # renaming the based on the PyTorch model
    print(f'Finish renaming {len(lfn_torch)} layers with {len(extra_param)} extra params!')
    print(f'\nThe extra params: {extra_param}')

    # Saving and verifying
    torch.save(lfn_torch, param_out)
    net_torch.load_state_dict(lfn_torch)

    print('DONE!')
