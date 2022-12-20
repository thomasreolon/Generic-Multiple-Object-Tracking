
from copy import deepcopy
import json

import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from datasets.fsc147 import build as build_dataset
from util.misc import NestedTensor


def main():

    # info about code execution
    args = get_args()

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.to(args.device)

    # load dataset
    dataset = load_svdataset(args.ds_eval, args.ds_split)








def get_args():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--ds_split', default='test', type=str)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args

def load_svdataset(datasetname, split):
    assert datasetname in {'e2e_gmot', 'e2e_fscd'}, f'invalid dataset "{datasetname}"'
    assert split in {'train', 'val', 'test'}, f'invalid dataset "{split}"'
    
    if datasetname=='e2e_gmot':
        video_images_list = load_gmot(split)
    elif datasetname=='e2e_fscd':
        video_images_list = load_fscd(split)

def load_gmot(split): raise NotImplementedError()

def load_fscd(split): raise NotImplementedError



class ListImgDataset(Dataset):
    def __init__(self, base_path, img_list, frame1_bb_list) -> None:
        super().__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.img_list = frame1_bb_list

        '''
        common settings
        '''
        self.img_height = 704   # 800
        self.img_width = 1216   # 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, fpath_or_ndarray, bb):
        # bb as a box coordinates array [x1,y1,x2,y2]
        if isinstance(fpath_or_ndarray, str):
            cur_img = cv2.imread(os.path.join(self.mot_path, fpath_or_ndarray))
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        elif isinstance(fpath_or_ndarray, function):
            cur_img = fpath_or_ndarray
        assert cur_img is not None
        exemplar = cur_img[bb[1]:bb[3], bb[0]:bb[2]]
        return cur_img, exemplar

    def init_img(self, img, exemplar):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, exemplar

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, exemplar = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, exemplar)

class FSCDDataset(Dataset):
    def __init__(self, args, vid=8) -> None:
        super().__init__()
        args.sampler_lengths[0] = 100
        self.ds = build_dataset('val', args)

        self._select(vid)

        self.img_height = 736
        self.img_width = 1312

    def _select(self, vid):
        data = self.ds[vid]
        self.images = [img.clone().permute(1,2,0).numpy() for img in data['imgs']]
        self.exemplar = data['patches']


    def init_img(self, img):
        ori_img = img.copy()
        ori_img = ori_img/4+.4 # de_normalize
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).contiguous()
        return img, ori_img

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        img, ori_img = self.init_img(img)
        exemplar = self.exemplar[0].clone(), self.exemplar[1].clone()
        return img, ori_img, exemplar


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        # self.seq_num = os.path.basename(vid)
        # img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        # img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        # self.img_list = sorted(img_list)
        # self.img_len = len(self.img_list)

        # self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        # os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.08, area_threshold=30, vis=True):
        total_dts = 0
        total_occlusion_dts = 0

        if self.args.det_db and os.path.exists(self.args.det_db):
            with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
                det_db = json.load(f)
        else:
            det_db = None
        
        # loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        loader = DataLoader(FSCDDataset(self.args, self.vid), 1, num_workers=2)  # TODO: initialize dataset img_list in Detection.__init__ (instead of dataset)
       
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = data[0][0], data[1][0], data[2]

            proposals = NestedTensor(proposals[0][0], proposals[1][0])
            cur_img, proposals = cur_img.to(self.args.device), proposals.to(self.args.device)

            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), None, None, exemplar=proposals)  ####### track_instances is = None....   we arent'reusing queries from prev
            track_instances = res['track_instances']

            dt_instances = deepcopy(track_instances)

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()

            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            img = cur_img.clone().cpu()[0].permute(1,2,0).numpy()[:,:,::-1]
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))

                if vis:
                    x1, y1, x2, y2 = [int(a*800/1080) for a in xyxy]
                    tmp = img[ y1:y2, x1:x2].copy()
                    img[y1-3:y2+3, x1-3:x2+3] = (0,2.3,0)
                    img[y1:y2, x1:x2] = tmp
            if vis:
                cv2.imshow('preds', img/4+.4)
                cv2.waitKey(40)
            

        # with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
        #     f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.to(args.device)

    # # '''for GMOT''' 
    vids = ['boat-3', 'boat-3', 'stock-1', 'airplane-1']

    # # '''for FSCD147''' 
    vids = [1,5,9]


    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    args.mot_path = '/home/intern/Desktop/datasets/GMOT/GenericMOT_JPEG_Sequence/'

    for vid in vids:#vids:
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold, vis=True)
