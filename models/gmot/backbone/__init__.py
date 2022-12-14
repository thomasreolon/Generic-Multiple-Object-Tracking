import torch
from .revbifpn import BackboneRevbifpn


def build_backbone(args):

    if True or args.nn_backbone == 'revbifpn':
        net = BackboneRevbifpn(4, args.num_feature_levels)
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

