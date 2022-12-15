from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
from torchvision.transforms import transforms


# should return {
#  imgs -> list of tensors
#  gt_instances -> list Instance :   Instance=(  boxes=tensor(Nx4), labels=[00000], obj_ids=[abcde])    : a = v_id*1000000 + obj_id
#  proposals -> tensor0x5   (yolodetections)
# }


class FSCDataset(Dataset):
    selected = ['23.jpg', '60.jpg', '76.jpg', '97.jpg', '98.jpg', '367.jpg', '417.jpg', '487.jpg', '783.jpg', '791.jpg', '881.jpg', '2103.jpg', '2130.jpg', '2415.jpg', '2423.jpg', '2435.jpg', '2441.jpg', '2458.jpg', '2461.jpg', '2477.jpg', '2478.jpg', '2482.jpg', '2491.jpg', '2499.jpg', '2517.jpg', '2520.jpg', '2529.jpg', '2538.jpg', '2558.jpg', '2664.jpg', '2698.jpg', '2788.jpg', '2793.jpg', '2800.jpg','3084.jpg', '3085.jpg', '3091.jpg', '3149.jpg', '3184.jpg', '3193.jpg', '3194.jpg', '3196.jpg', '3197.jpg', '3198.jpg', '3207.jpg', '3215.jpg', '3255.jpg', '3294.jpg', '3298.jpg', '3303.jpg', '3304.jpg', '3305.jpg', '3364.jpg', '3368.jpg', '3371.jpg', '3372.jpg', '3384.jpg', '3387.jpg', '3408.jpg', '3445.jpg', '3607.jpg', '3633.jpg', '3681.jpg', '3684.jpg', '3692.jpg', '3703.jpg', '3708.jpg', '3721.jpg', '3734.jpg', '3740.jpg', '3798.jpg', '3814.jpg', '3826.jpg', '3828.jpg', '3829.jpg', '3882.jpg', '3897.jpg', '3949.jpg', '3952.jpg', '3960.jpg', '3964.jpg', '3973.jpg', '4127.jpg', '4136.jpg', '4182.jpg', '4187.jpg', '4190.jpg', '4203.jpg', '4207.jpg', '4209.jpg', '4210.jpg', '4212.jpg', '4217.jpg', '4232.jpg', '4238.jpg', '4243.jpg', '4247.jpg', '4252.jpg', '4253.jpg', '4254.jpg', '4256.jpg', '4258.jpg', '4260.jpg', '4262.jpg', '4269.jpg', '4271.jpg', '4274.jpg', '4275.jpg', '4276.jpg', '4278.jpg', '4306.jpg', '4333.jpg', '4337.jpg', '4353.jpg', '4372.jpg', '4430.jpg', '4431.jpg', '4436.jpg', '4479.jpg', '4480.jpg', '4621.jpg', '4628.jpg', '4666.jpg', '4782.jpg', '4864.jpg', '4973.jpg', '4978.jpg', '4980.jpg', '4981.jpg', '4984.jpg', '5030.jpg', '5033.jpg', '5041.jpg', '5125.jpg', '5130.jpg', '5138.jpg', '5159.jpg', '5162.jpg', '5182.jpg', '5243.jpg', '5244.jpg', '5245.jpg', '5248.jpg', '5253.jpg', '5271.jpg', '5307.jpg', '5348.jpg', '5354.jpg', '5412.jpg', '5421.jpg', '5423.jpg', '5425.jpg', '5427.jpg', '5430.jpg', '5447.jpg', '5453.jpg', '5456.jpg', '5457.jpg', '5536.jpg', '5603.jpg', '5619.jpg', '5625.jpg', '5632.jpg', '5709.jpg', '5713.jpg', '5717.jpg', '5732.jpg', '5751.jpg', '5779.jpg', '5780.jpg', '5959.jpg', '5961.jpg', '5963.jpg', '5967.jpg', '5984.jpg', '5988.jpg', '5994.jpg', '6008.jpg', '6022.jpg', '6028.jpg', '6030.jpg', '6177.jpg', '6198.jpg', '6205.jpg', '6212.jpg', '6222.jpg', '6230.jpg', '6375.jpg', '6378.jpg', '6387.jpg', '6390.jpg', '6391.jpg', '6407.jpg', '6420.jpg', '6425.jpg', '6438.jpg', '6445.jpg', '6448.jpg', '6456.jpg', '6473.jpg', '6493.jpg', '6495.jpg', '6504.jpg', '6529.jpg', '6558.jpg', '6587.jpg', '6683.jpg', '6687.jpg', '7177.jpg', '6972.jpg', '7638.jpg', '7158.jpg', '6975.jpg', '7170.jpg', '7227.jpg', '7166.jpg', '7141.jpg', '237.jpg', '591.jpg', '984.jpg',  '1910.jpg', '1972.jpg', '1982.jpg', '2815.jpg', '2825.jpg', '3266.jpg', '3427.jpg', '3432.jpg', '3438.jpg', '3478.jpg', '3518.jpg', '3546.jpg', '3567.jpg', '3660.jpg', '3661.jpg', '3669.jpg', '3675.jpg', '3762.jpg', '3764.jpg', '3768.jpg', '3770.jpg', '3771.jpg', '3780.jpg', '3782.jpg', '4585.jpg', '4754.jpg', '5048.jpg', '5059.jpg', '5062.jpg', '5096.jpg', '5152.jpg', '5155.jpg', '5202.jpg', '5210.jpg', '5223.jpg', '5664.jpg', '5668.jpg', '5670.jpg', '5900.jpg', '6101.jpg', '6107.jpg', '6120.jpg', '7403.jpg', '7487.jpg', '7504.jpg', '7049.jpg', '3.jpg', '311.jpg', '996.jpg', '2032.jpg', '2143.jpg', '2181.jpg', '2213.jpg', '2227.jpg', '2269.jpg', '2288.jpg', '2908.jpg', '2912.jpg', '2918.jpg', '2924.jpg', '2942.jpg', '2945.jpg', '4063.jpg', '4073.jpg', '4086.jpg', '4120.jpg', '4287.jpg', '4294.jpg', '4303.jpg', '4935.jpg', '5380.jpg', '5387.jpg', '5463.jpg', '5464.jpg', '5512.jpg', '5575.jpg', '5917.jpg', '5923.jpg', '5927.jpg', '6042.jpg', '7693.jpg', '7273.jpg', '7263.jpg', '7384.jpg', '7709.jpg', '7080.jpg', '7683.jpg', '7581.jpg', '7666.jpg', '7282.jpg', '6856.jpg', '7506.jpg']

    def __init__(self, root, split) -> None:
        super().__init__()

        # gt = root+'gt_density_map_adaptive_384_VarV2/'
        path_imgs = root+'/images_384_VarV2/'
        path_ann = root+'/annotation_FSC147_384.json'

        n_imgs = len(self.selected)
        if split == 'train' : self.imgs = self.selected[              : n_imgs*3//4    ]  
        elif split == 'val' : self.imgs = self.selected[n_imgs*3//4-30: n_imgs*3//4+30 ]
        elif split == 'test': self.imgs = self.selected[n_imgs*3//4+30:                ]




"""
todo list:

0) visualize results on MOT of pretrained MOT - gioMatt

1) import GMOT - gioPome
2) make BMN module - gioPome
3) train MOTR-BMN on GMOT - venMatt
4) visualize results - venSera







"""






def get_image_classes(class_file):
    class_dict = dict()
    with open(class_file, 'r') as f:
        classes = [line.split('\t') for line in f.readlines()]
    
    for entry in classes:
        class_dict[entry[0]] = entry[1]
    
    return class_dict 

def batch_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0], scale_embedding, batch[2] = batch_padding(batch[0], batch[2])
    patches = torch.stack(batch[1], dim=0)
    batch[1] = {'patches': patches, 'scale_embedding': scale_embedding.long()}
    return tuple(batch)

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def batch_padding(tensor_list, target_dict):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        density_shape = [len(tensor_list)] + [1, max_size[1], max_size[2]]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        density_map = torch.zeros(density_shape, dtype=dtype, device=device)
        pt_map = torch.zeros(density_shape, dtype=dtype, device=device)
        gtcount = []
        scale_embedding = []
        for idx, package  in enumerate(zip(tensor_list, tensor, density_map, pt_map)):
            img, pad_img, pad_density, pad_pt_map = package
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_density[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['density_map'])
            pad_pt_map[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['pt_map'])
            gtcount.append(target_dict[idx]['gtcount']) 
            scale_embedding.append(target_dict[idx]['scale_embedding'])
        target = {'density_map': density_map,
                  'pt_map': pt_map,
                  'gtcount': torch.tensor(gtcount)}
    else:
        raise ValueError('not supported')
    return tensor, torch.stack(scale_embedding), target

class FSC147Dataset(Dataset):
    def __init__(self, data_dir, data_list, scaling, box_number=3, scale_number=20, min_size=384, max_size=1584, preload=True, main_transform=None, query_transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.scaling = scaling
        self.box_number = box_number
        self.scale_number = scale_number
        self.preload = preload
        self.main_transform = main_transform
        self.query_transform = query_transform
        self.min_size = min_size
        self.max_size = max_size 
        
        # load annotations for the entire dataset
        annotation_file = os.path.join(self.data_dir, 'annotation_FSC147_384.json')
        image_classes_file = os.path.join(self.data_dir, 'ImageClasses_FSC147.txt')
                
        self.image_classes = get_image_classes(image_classes_file)
        with open(annotation_file) as f:
            self.annotations = json.load(f)
    
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.patches = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx][0]

        if file_name in self.images:
            img = self.images[file_name]
            target = self.targets[file_name]
            patches = self.patches[file_name]
            
        else:
            image_path = os.path.join(self.data_dir, 'images_384_VarV2/' + file_name)
            density_path = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2/' + file_name.replace('jpg', 'npy'))
            
            img_info = self.annotations[file_name]
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            # resize the image
            r = 1.0
            if h > self.max_size or w > self.max_size:
                r = self.max_size / max(h, w)
            if r * h < self.min_size or w*r < self.min_size:
                r = self.min_size / min(h, w)
            nh, nw = int(r*h), int(r*w)
            img = img.resize((nw, nh), resample=Image.BICUBIC)
        
            #target = np.zeros((nh, nw), dtype=np.float32)
            density_map = np.load(density_path).astype(np.float32)
            pt_map = np.zeros((nh, nw), dtype=np.int32)
            points = (np.array(img_info['points']) * r).astype(np.int32)
            boxes = np.array(img_info['box_examples_coordinates']) * r   
            boxes = boxes[:self.box_number, :, :]
            gtcount = points.shape[0]
            
            # crop patches and data transformation
            target = dict()
            patches = []
            scale_embedding = []
            
            #print('boxes:', boxes.shape[0])
            if points.shape[0] > 0:     
                points[:,0] = np.clip(points[:,0], 0, nw-1)
                points[:,1] = np.clip(points[:,1], 0, nh-1)
                pt_map[points[:, 1], points[:, 0]] = 1 
                for box in boxes:
                    x1, y1 = box[0].astype(np.int32)
                    x2, y2 = box[2].astype(np.int32)
                    patch = img.crop((x1, y1, x2, y2))
                    patches.append(self.query_transform(patch))
                    # calculate scale
                    scale = (x2 - x1) / nw * 0.5 + (y2 -y1) / nh * 0.5 
                    scale = scale // (0.5 / self.scale_number)
                    scale = scale if scale < self.scale_number - 1 else self.scale_number - 1
                    scale_embedding.append(scale)
            
            target['density_map'] = density_map * self.scaling
            target['pt_map'] = pt_map
            target['gtcount'] = gtcount
            target['scale_embedding'] = torch.tensor(scale_embedding)
            
            img, target = self.main_transform(img, target)
            patches = torch.stack(patches, dim=0)
           
            if self.preload:
                self.images.update({file_name: img})
                self.patches.update({file_name: patches})
                self.targets.update({file_name: target})
            
        return img, patches, target

def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize-h%psize),(psize-w%psize)
    # print(ph,pw)

    (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)   
    (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
    if (ph!=psize) or (pw!=psize):
        tmp_pad = [pl, pr, pt, pb]
        # print(tmp_pad)
        inputs = torch.nn.functional.pad(inputs, tmp_pad)
    
    return inputs
    

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img, target):
        img = self.img_trans(img)
        density_map = target['density_map']
        pt_map = target['pt_map']
        pt_map = torch.from_numpy(pt_map).unsqueeze(0)
        density_map = torch.from_numpy(density_map).unsqueeze(0)
        
        img = pad_to_constant(img, 32)
        density_map = pad_to_constant(density_map, 32)
        pt_map = pad_to_constant(pt_map, 32)
        target['density_map'] = density_map.float()
        target['pt_map'] = pt_map.float()
        
        return img, target


def get_query_transforms(is_train, exemplar_size):
    if is_train:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def build_dataset(cfg, is_train):
    main_transform = MainTransform()
    query_transform = get_query_transforms(is_train, cfg.DATASET.exemplar_size)
    if is_train: 
        data_list = cfg.DATASET.list_train
    else:
        if not cfg.VAL.evaluate_only:
            data_list = cfg.DATASET.list_val 
        else:
            data_list = cfg.DATASET.list_test
    
    dataset = FSC147Dataset(data_dir=cfg.DIR.dataset,
                            data_list=data_list,
                            scaling=1.0,
                            box_number=cfg.DATASET.exemplar_number,
                            scale_number=cfg.MODEL.ep_scale_number,
                            main_transform=main_transform,
                            query_transform=query_transform)
    
    return dataset
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_transform = MainTransform()
    query_transform = get_query_transforms(is_train=True, exemplar_size=(128, 128))
    
    dataset = FSC147Dataset(data_dir='D:/dataset/FSC147/',
                            data_list='D:/dataset/FSC147/train.txt',
                            scaling=1.0,
                            main_transform=main_transform,
                            query_transform=query_transform)
    
    data_loader = DataLoader(dataset, batch_size=5, collate_fn=batch_collate_fn)
    
    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        print(img.shape)
        print(patches.keys())
        print(targets.keys())
        break
    