# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .motr import build as build_motr
from .gmot.v_oq import build as build_v_oq
from .gmot.v_mix import build as build_v_mix
from .gmot.v_da_old import build as build_v_da_old
from .gmot.v_da_new import build as build_v_da_new


def build_model(args):
    arch_catalog = {
        'motr': build_motr,
        'v_oq': build_v_oq,
        'v_mix': build_v_mix,
        'v_da_old': build_v_da_old,
        'v_da_new': build_v_da_new,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)

