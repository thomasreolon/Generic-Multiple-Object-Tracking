import torch
import torch.nn as nn



class ImgExemplarSelfAttn(nn.Module):

    def __init__(self, emb_size=256, num_queries=150):
        super().__init__()

        self.num_queries = num_queries
        self.similarity = nn.Parameter(torch.eye(emb_size), requires_grad=True)
        self.v = nn.Linear(emb_size, emb_size, bias=False)

    
    def forward(self, img_feat, exe_feat, mask=None):
        """ 
        - img_feat: BxCxHxW
        - exe_feat: BxC
        - mask    : BxHxW   0 if pixel should be ignored

        with C=embed_size
        """
        # unroll image
        B,C,H,W = img_feat.shape
        unrolled = img_feat.view(B,C,H*W)              #  BxCxHW


        #TODO: add pooling if exefeat is BChw
        exe_feat = exe_feat*1.2

        exe_feat = exe_feat.unsqueeze(2)            #  BxCx1

        # compute values
        concat = torch.cat((unrolled, exe_feat), dim=2) #  BxCx(HW+1)
        values = self.v(concat.permute(0,2,1))          #  Bx(HW+1)xC
        attn_matrix = unrolled.permute(0,2,1) @ self.similarity @ concat  # BxHWx(HW+1)

        # select best pixels for queries
        q_ids = self.get_queries(attn_matrix, img_feat.shape)

        queries=[] # maybe there is a better way
        for img, id in zip(values[:,:-1], q_ids):
            queries.append(img[id])
        queries = torch.stack(queries)     #  Bx150xC

        if mask is not None:
            with torch.no_grad():
                mask = mask.view(B,H*W,1) @ mask.view(B,1,H*W)
                mask = torch.cat((mask,torch.ones((B,H*W,1), device=mask.device)), dim=2)
                mask = (mask!=1).float()*1e9 # negative where "pixel not interesting"
            attn_matrix = attn_matrix - mask
        attn_matrix = attn_matrix.softmax(dim=2)

        new_img_feat = attn_matrix @ values #  BxHWxC
        new_img_feat = new_img_feat.permute(0,2,1).view(B,C,H,W)

        return new_img_feat+img_feat, queries

    @torch.no_grad() 
    def get_queries(self, attn_matrix, shapes):
        B,_,H,W = shapes

        # similarity between each pixel and the exemplar
        simil = attn_matrix[:,:,-1].detach().clone().cpu().view(B,H,W)  # BxHW

        # NMS like
        c1 = simil[:, :-1] >= simil[:, 1:]  # bigger than up
        c2 = simil[:, 1:] > simil[:, :-1]  # bigger than down
        c3 = simil[:, :,:-1] >= simil[:, :,1:]  # bigger than left
        c4 = simil[:, :,1:] > simil[:, :,:-1]  # bigger than right
        max_ = c1[:, 1:, 1:-1] & c2[:, :-1, 1:-1] & c3[:, 1:-1, 1:] & c4[:, 1:-1, :-1]

        # kills all points with a bigger neighbour nearby
        simil[:, 1:-1, 1:-1][~max_] -= 1e9

        _, q_ids = simil.view(B,H*W).topk(self.num_queries)  
        return q_ids


if __name__ == '__main__':

    img_feat, exe_feat = torch.rand(2,256,32,48), torch.rand(2,256)
    net = ImgExemplarSelfAttn()

    new_img_feat, queries = net(img_feat, exe_feat, torch.ones_like(img_feat)[:,0])

    print(new_img_feat.shape, queries.shape)




