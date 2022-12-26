# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .motr import build as build_motr
from .gmot import build as build_gmot
from .gmot.v_da2 import build as build_gmot2
from .gmot.v3 import build as build_gmot3


def build_model(args):
    arch_catalog = {
        'motr': build_motr,
        'gmot': build_gmot,
        'gmot2': build_gmot2,
        'gmot3': build_gmot3,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)

