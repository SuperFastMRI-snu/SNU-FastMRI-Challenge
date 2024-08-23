import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.feature_varnet import FIVarNet_n_att

def test(args, model1, model2, model3, model4, data_loader):
    # ensemble 적용
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # calculating acc
            indices_of_ones = torch.where(mask.flatten() == 1)[0]
            acceleration = int(indices_of_ones[1]-indices_of_ones[0])

            if acceleration < 8:
                output = (model1(kspace, mask)+model2(kspace, mask))/2
            elif acceleration < 12:
                output = (model1(kspace, mask)+model2(kspace, mask)+model3(kspace, mask)+model4(kspace, mask))/4
            else:
                output = (model3(kspace, mask)+model4(kspace, mask))/2
                
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model1 = FIVarNet_n_att(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   unet_chans=args.unet_chans)
    model2 = FIVarNet_n_att(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   unet_chans=args.unet_chans)
    model3 = FIVarNet_n_att(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   unet_chans=args.unet_chans)
    model4 = FIVarNet_n_att(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   unet_chans=args.unet_chans)
                
    model1.to(device=device)
    model2.to(device=device)
    model3.to(device=device)
    model4.to(device=device)


    checkpoint1 = torch.load(args.exp_dir_acc45 / 'model24_acc45.pt', map_location='cpu')
    checkpoint2 = torch.load(args.exp_dir_acc45 / 'model25_acc45.pt', map_location='cpu')
    checkpoint3 = torch.load(args.exp_dir_acc89 / 'model23_acc89.pt', map_location='cpu')
    checkpoint4 = torch.load(args.exp_dir_acc89 / 'model25_acc89.pt', map_location='cpu')


    model1.load_state_dict(checkpoint1['model'])
    model2.load_state_dict(checkpoint2['model'])
    model3.load_state_dict(checkpoint3['model'])
    model4.load_state_dict(checkpoint4['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model1, model2, model3, model4, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)