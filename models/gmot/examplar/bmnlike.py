import torch
import torch.nn as nn



class ImgExemplarSelfAttn(nn.Module):

    def __init__(self, emb_size=256):
        super().__init__()

        self.similarity = nn.Parameter(torch.eye(emb_size), requires_grad=True)
        self.v = nn.Linear(emb_size, emb_size, bias=False)

    
    def forward(self, img_feat, exe_feat):
        """ 
        - img_feat: BxCxHxW
        - exe_feat: BxC

        with C=embed_size
        """
        # unroll image
        B,C,H,W = img_feat.shape
        unrolled = img_feat.reshape(B,C,H*W)              #  BxCxHW


        #TODO: add pooling if exefeat is BChw
        exe_feat = exe_feat*1.2

        exe_feat = exe_feat.unsqueeze(2)            #  BxCx1

        # compute values
        concat = torch.cat((unrolled, exe_feat), dim=2) #  BxCx(HW+1)
        values = self.v(concat.permute(0,2,1))          #  Bx(HW+1)xC
        attn_matrix = unrolled.permute(0,2,1) @ self.similarity @ concat  # BxHWx(HW+1)

        # get topk queries:
        similarity_w_exemplar = attn_matrix[:,:,-1]
        _, q_ids = similarity_w_exemplar.topk(150)       #TODO: avoid nearby pixels to be selected


        queries=[] # maybe there is a better way
        for img, id in zip(values[:,:-1], q_ids):
            queries.append(img[id])
        queries = torch.stack(queries) #  Bx150xC


        # TODO: maybe subtract mask from attn
        attn_matrix = attn_matrix.softmax(dim=2)

        new_img_feat = attn_matrix @ values #  BxHWxC
        new_img_feat.permute(0,2,1).view(B,C,H,W)       # or use RESHAPE??TODO

        return new_img_feat, queries


if __name__ == '__main__':

    img_feat, exe_feat = torch.rand(2,256,32,48), torch.rand(2,256)
    net = ImgExemplarSelfAttn()

    new_img_feat, queries = net(img_feat, exe_feat)

    print(new_img_feat.shape, queries.shape)




