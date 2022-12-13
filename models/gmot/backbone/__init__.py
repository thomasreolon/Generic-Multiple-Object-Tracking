import torch
from .revbifpn import BackboneRevbifpn


def get_backbone(args):

    if args.nn_backbone == 'revbifpn':
        net = BackboneRevbifpn(args.nn_initial_downsample, args.nn_n_feat_levels)
    else:
        raise Exception(f'{args.nn_backbone} Not Implemented Yet')

    return torch.nn.Sequential(net, ListFeatNormalizer())

class ListFeatNormalizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        new_x = []
        for x_i in x:
            x_i = torch.sqrt(x_i)
            x_i = (2*x_i / x_i.sum(dim=1).unsqueeze(1))  -1
            new_x.append(x_i)
        return new_x

