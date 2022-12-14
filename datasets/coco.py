# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from util.misc import get_local_rank, get_local_size
import datasets.transforms as T

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


class TvCocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset




"""

object example in coco
{
    "segmentation": [[382.41,450.5,363.23,435.5,349.05,432.16,333.21,445.5,344.89,462.18,352.39,468.85,347.39,489.69,355.72,516.37,359.89,535.55,364.06,535.55,368.23,532.22,370.73,538.89,370.73,543.89,373.23,544.72,390.74,552.23,412.42,555.56,419.09,557.23,429.1,560.56,425.76,568.9,422.43,563.9,422.43,571.4,407.42,564.73,403.25,562.23,403.25,567.23,411.59,575.57,419.09,577.24,412.42,583.96,415.76,587.29,427.43,581.46,433.27,577.29,442.44,582.29,446.61,578.12,435.77,573.95,435.77,571.45,439.1,555.66,446.61,548.15,454.94,557.32,459.11,556.54,468.29,549.03,474.12,549.03,475.79,549.03,480.0,536.53,479.12,524.85,477.46,519.02,479.12,482.42,476.62,480.76,467.45,478.25,455.78,473.25,442.44,464.08,435.77,451.62,424.93,449.95,423.26,449.12,413.26,444.12,408.25,444.12,392.41,446.62,386.57,453.29]],
    "area": 13048.958299999998,
    "iscrowd": 0,
    "image_id": 123585,
    "bbox": [333.21,432.16,146.79,155.13],
    "category_id": 16,
    "id": 43705
}

"categories": [
    {"supercategory": "person","id": 1,"name": "person"},
    {"supercategory": "vehicle","id": 2,"name": "bicycle"},
    {"supercategory": "vehicle","id": 3,"name": "car"},
    {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
    {"supercategory": "vehicle","id": 5,"name": "airplane"},
    {"supercategory": "vehicle","id": 6,"name": "bus"},
    {"supercategory": "vehicle","id": 7,"name": "train"},
    {"supercategory": "vehicle","id": 8,"name": "truck"},
    {"supercategory": "vehicle","id": 9,"name": "boat"},
    {"supercategory": "outdoor","id": 10,"name": "traffic light"},
    {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
    {"supercategory": "outdoor","id": 13,"name": "stop sign"},
    {"supercategory": "outdoor","id": 14,"name": "parking meter"},
    {"supercategory": "outdoor","id": 15,"name": "bench"},
    {"supercategory": "animal","id": 16,"name": "bird"},
    {"supercategory": "animal","id": 17,"name": "cat"},
    {"supercategory": "animal","id": 18,"name": "dog"},
    {"supercategory": "animal","id": 19,"name": "horse"},
    {"supercategory": "animal","id": 20,"name": "sheep"},
    {"supercategory": "animal","id": 21,"name": "cow"},
    {"supercategory": "animal","id": 22,"name": "elephant"},
    {"supercategory": "animal","id": 23,"name": "bear"},
    {"supercategory": "animal","id": 24,"name": "zebra"},
    {"supercategory": "animal","id": 25,"name": "giraffe"},
    {"supercategory": "accessory","id": 27,"name": "backpack"},
    {"supercategory": "accessory","id": 28,"name": "umbrella"},
    {"supercategory": "accessory","id": 31,"name": "handbag"},
    {"supercategory": "accessory","id": 32,"name": "tie"},
    {"supercategory": "accessory","id": 33,"name": "suitcase"},
    {"supercategory": "sports","id": 34,"name": "frisbee"},
    {"supercategory": "sports","id": 35,"name": "skis"},
    {"supercategory": "sports","id": 36,"name": "snowboard"},
    {"supercategory": "sports","id": 37,"name": "sports ball"},
    {"supercategory": "sports","id": 38,"name": "kite"},
    {"supercategory": "sports","id": 39,"name": "baseball bat"},
    {"supercategory": "sports","id": 40,"name": "baseball glove"},
    {"supercategory": "sports","id": 41,"name": "skateboard"},
    {"supercategory": "sports","id": 42,"name": "surfboard"},
    {"supercategory": "sports","id": 43,"name": "tennis racket"},
    {"supercategory": "kitchen","id": 44,"name": "bottle"},
    {"supercategory": "kitchen","id": 46,"name": "wine glass"},
    {"supercategory": "kitchen","id": 47,"name": "cup"},
    {"supercategory": "kitchen","id": 48,"name": "fork"},
    {"supercategory": "kitchen","id": 49,"name": "knife"},
    {"supercategory": "kitchen","id": 50,"name": "spoon"},
    {"supercategory": "kitchen","id": 51,"name": "bowl"},
    {"supercategory": "food","id": 52,"name": "banana"},
    {"supercategory": "food","id": 53,"name": "apple"},
    {"supercategory": "food","id": 54,"name": "sandwich"},
    {"supercategory": "food","id": 55,"name": "orange"},
    {"supercategory": "food","id": 56,"name": "broccoli"},
    {"supercategory": "food","id": 57,"name": "carrot"},
    {"supercategory": "food","id": 58,"name": "hot dog"},
    {"supercategory": "food","id": 59,"name": "pizza"},
    {"supercategory": "food","id": 60,"name": "donut"},
    {"supercategory": "food","id": 61,"name": "cake"},
    {"supercategory": "furniture","id": 62,"name": "chair"},
    {"supercategory": "furniture","id": 63,"name": "couch"},
    {"supercategory": "furniture","id": 64,"name": "potted plant"},
    {"supercategory": "furniture","id": 65,"name": "bed"},
    {"supercategory": "furniture","id": 67,"name": "dining table"},
    {"supercategory": "furniture","id": 70,"name": "toilet"},
    {"supercategory": "electronic","id": 72,"name": "tv"},
    {"supercategory": "electronic","id": 73,"name": "laptop"},
    {"supercategory": "electronic","id": 74,"name": "mouse"},
    {"supercategory": "electronic","id": 75,"name": "remote"},
    {"supercategory": "electronic","id": 76,"name": "keyboard"},
    {"supercategory": "electronic","id": 77,"name": "cell phone"},
    {"supercategory": "appliance","id": 78,"name": "microwave"},
    {"supercategory": "appliance","id": 79,"name": "oven"},
    {"supercategory": "appliance","id": 80,"name": "toaster"},
    {"supercategory": "appliance","id": 81,"name": "sink"},
    {"supercategory": "appliance","id": 82,"name": "refrigerator"},
    {"supercategory": "indoor","id": 84,"name": "book"},
    {"supercategory": "indoor","id": 85,"name": "clock"},
    {"supercategory": "indoor","id": 86,"name": "vase"},
    {"supercategory": "indoor","id": 87,"name": "scissors"},
    {"supercategory": "indoor","id": 88,"name": "teddy bear"},
    {"supercategory": "indoor","id": 89,"name": "hair drier"},
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}]



"""