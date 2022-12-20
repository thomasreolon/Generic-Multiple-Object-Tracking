# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import gc
import sys
from typing import Iterable

import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda

import cv2

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        gc.collect(); torch.cuda.empty_cache()

        # # images are a sequence of 5 frames from the same video (show GT for debugging)
        # import cv2, numpy as np
        # imgs = data_dict['imgs']
        # concat = torch.cat(imgs, dim=1)
        # concat = np.ascontiguousarray(concat.clone().permute(1,2,0).numpy() [:,:,::-1])
        # # concat = (((concat * 0.22) + 0.5) * 255)

        # for i in range(len(imgs)):
        #     for box in data_dict['gt_instances'][i].boxes:
        #         box = (box.view(2,2) * torch.tensor([imgs[0].shape[2], imgs[0].shape[1]]).view(1,2)).int()
        #         x1,x2 = box[0,0] - box[1,0]//2, box[0,0] + box[1,0]//2
        #         y1,y2 = box[0,1] - box[1,1]//2, box[0,1] + box[1,1]//2
        #         y1, y2 = y1+imgs[0].shape[1]*i, y2+imgs[0].shape[1]*i
        #         tmp = concat[y1:y2, x1:x2].copy()
        #         concat[y1-2:y2+2, x1-2:x2+2] = 1
        #         concat[y1:y2, x1:x2] = tmp

        # concat = cv2.resize(concat, (400, 1300))
        # cv2.imshow('batch', concat/4+ .3) 
        # cv2.waitKey()


        data_dict = data_dict_to_cuda(data_dict, device)
        # try:
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # except RuntimeError as e:
        #     del e
        #     torch.cuda.empty_cache()
        #     continue
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()


        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes



        ######### preds visualization
        if False:
            dt_instances = model.module.post_process(outputs['track_instances'], data_dict['imgs'][0].shape[-2:])

            keep = dt_instances.scores > .1
            keep &= dt_instances.obj_idxes >= 0
            dt_instances = dt_instances[keep]

            wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
            areas = wh[:, 0] * wh[:, 1]
            keep = areas > 100
            dt_instances = dt_instances[keep]

            if len(dt_instances)==0:
                print('nothing found')
            else:
                print('ok')
                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()

                img = data_dict['imgs'][-1].clone().cpu().permute(1,2,0).numpy()[:,:,::-1]
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = [int(a) for a in xyxy]

                    tmp = img[ y1:y2, x1:x2].copy()
                    img[y1-3:y2+3, x1-3:x2+3] = (0,2.3,0)
                    img[y1:y2, x1:x2] = tmp
                cv2.imshow('preds', img/4+.4)
                cv2.waitKey()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
