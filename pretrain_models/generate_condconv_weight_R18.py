"""
Generate condconv weight from Res50 ImageNet pretrained weight.
Only the last stage uses condconv (according to condconv paper).
We simply copy the weight n times (n stands for number of experts)
"""
import argparse
import torch
import logging

def generate_condconv_weight(weight, n=2):
    torch.manual_seed(0)
    new_weight = {}
    for key in weight.keys():
        if ("res3" in key or "res2" in key or "stem" in key) and ("conv" in key or "shortcut" in key) and "weight" in key and not "norm" in key:
            c_in, c_out, h, w = weight[key].shape
            x = weight[key].repeat(n,1,1,1,1)
            noise = torch.randn(n, c_in, c_out, h, w) * weight[key].std() * 1e-2
            print(weight[key].abs().mean(), noise.abs().mean())
            # noise[0] = 0  # can leave one of them alone. TODO: feeling this will cause trouble. try it later.
            new_weight[key] = x + noise
            print("extending {0} from {1} into {2}".format(key, weight[key].shape, x.shape))
        else:
            new_weight[key] = weight[key]
    return new_weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('new_model', type=str)   

    args = parser.parse_args()
    
    weight = torch.load(args.model)
    with torch.no_grad():
        weight = generate_condconv_weight(weight)
    torch.save(weight, args.new_model)
