import torch
import torch.nn as nn
import numpy as np


class PixelExtractor(nn.Module):
    """maybe take avg of neighbours instead of single point (?)"""
    def __init__(self) -> None:
        super().__init__()
    
    def get_points_hw(self, r_hw, center, n_points):
        points = []
        step = 6.28318 / n_points
        for i in range(n_points):
            p = [center[0]-np.cos(i*step)*r_hw[0], center[1]-np.sin(i*step)*r_hw[1]]
            points.append(np.round(p).astype(int).tolist())
        return points

    def forward(self, esrcs):
        """the number of queries is always squarable (1,4,9,16)"""
        # set exemplar as first pixel
        queries = []
        for lvl, src in enumerate(esrcs):
            _,_,H,W = src.shape
            lvl_queries = []

            # always get central pixel     # +1 
            lvl_queries.append( src[:,:,H//2,W//2] )
            # always get global pooling     # +1 
            lvl_queries.append( src.mean(dim=(2,3)) )

            # pixels in a oval
            num_pix = 2*(len(esrcs)-lvl-1) # 4, 2, 0
            if num_pix:
                for h,w in self.get_points_hw((H//4, W//4), (H//2,W//2), num_pix):
                    lvl_queries.append( src[:,:,h,w] )
            queries.append(torch.stack(lvl_queries, dim=2))
        return queries

class QueryExtractor(nn.Module):

    def __init__(self, emb_size=256, num_queries=150, n_flvl=3):
        super().__init__()

        self.num_queries = num_queries
        self.num_features = n_flvl
        self.extractor = PixelExtractor()
        self.similarity = nn.Parameter(torch.eye(emb_size), requires_grad=True)
        self.attnpooling = nn.ParameterList([
            nn.Parameter(torch.ones(2*(i+1))/(2*(i+1)), requires_grad=True)
            for i in range(n_flvl)
        ])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.proposer =nn.Sequential(
            nn.BatchNorm2d(n_flvl),
            nn.Conv2d(n_flvl,n_flvl*2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(n_flvl*2,n_flvl*2,kernel_size=7,padding=3),
            nn.ReLU(),
            nn.Conv2d(n_flvl*2,3, kernel_size=1),
        )

        self.lvl_importance = nn.Parameter(torch.ones(n_flvl, 1), requires_grad=True)
        
    
    def forward(self, img_feat, exe_feat, mask=None):
        """ 
        - img_feat: F xBxCxHxW
        - exe_feat: F xBxCxhxw
        - mask    : BxHxW   1 if pixel should be ignored

        with C=embed_size
        """

        exe_feat = self.extractor(exe_feat)  #  BxCxN
        similarities_maps = []
        for lvl, (src, esrc) in enumerate(zip(img_feat, exe_feat)):
            B,C,H,W = src.shape
            src = src.view(B,C,H*W).transpose(1,2)   #  BxHWxC

            similarity = src @ self.similarity @ esrc  # BxHWxN
            similarity = (similarity @ self.attnpooling[lvl].view(1,-1,1)).view(B,1,H,W)  # BxHW

            for _ in range(lvl):
                similarity = self.up(similarity)
            similarities_maps.append(similarity)
        similarities_maps = torch.cat(similarities_maps, dim=1)

        interest = self.proposer(similarities_maps)

        return self.get_locations(interest, img_feat[0].shape[-2:]), interest

    @torch.no_grad()
    def get_locations(self, interest, H,W):
        # NMS in interest
        good_pixels1 = None

        # thresholding
        good_pixels2 = interest[:,0] > 0.6

        # getting indices
        good_pixels = (good_pixels1 & good_pixels2).nonzero(as_tuple=True)  
        bb = interest[good_pixels].sigmoid()    [0]  # N, 2   TODO: [0] imposes to use BS=1
        yx = torch.stack(good_pixels[-2:], dim=1).float()  # N, 2 #TODO: crashes if BS>1
        yx = yx / torch.tensor([H,W],device=yx.device).view(1,1,2)
        return torch.cat((yx,bb),dim=2)  # Nx4

    def get_queries(self, srcs, ref_pts):
        """not in the forward cause you could further process srcs between the 2 methods"""
        queries = []
        for src in srcs:
            lvl_q = []
            B,C,H,W = src.shape
            for pt in ref_pts:
                y,x = int(pt[0]*H), int(pt[1]*W)
                lvl_q.append(src[0,:,y,x])
            queries.append(torch.stack(lvl_q))
        queries = torch.stack(queries, dim=2) @ self.lvl_importance / self.lvl_importance.sum()
        
