import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
# FIVarNet without block attention
from utils.model.feature_varnet import FIVarNet_n_att
from utils.model.att_unet import Att_NormUnet

def test(args, model1, model2, data_loader):
    # ensemble 적용
    model1.eval()
    model2.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # calculating acc
            indices_of_ones = torch.where(mask.flatten() == 1)[0]
            acceleration = int(indices_of_ones[1]-indices_of_ones[0])
            #print("acceleration: "+str(acceleration))

            # acc에 따른 model 선정
            if abs(acceleration - 4.5) < abs(acceleration - 8.5):
                output = model2(model1(kspace, mask))
                #print("model1")
            else:
                output = model2(model1(kspace, mask))
                #print("model2")

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
    model2 = Att_NormUnet(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pools=4,
        use_attention=True
    )
    model1.to(device=device)
    model2.to(device=device)
    
    checkpoint1 = torch.load(args.exp_dir / 'model13_89.pt', map_location='cpu')
    checkpoint2 = torch.load(args.exp_dir2 / 'best_model22.pt', map_location='cpu')
    print(checkpoint1['epoch'], checkpoint1['best_val_loss'].item())
    print(checkpoint2['epoch'], checkpoint2['best_val_loss'].item())
    model1.load_state_dict(checkpoint1['model'])
    model2.load_state_dict(checkpoint2['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model1, model2, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)