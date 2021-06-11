import numpy as np
import re
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import build_model

sensitive = False
batch_max_length = 300

norm_cfg = dict(type='BN')
num_class = 245 # vocab 242 + token 3
base_channel = 16

cstr_structure = dict(
        type='GModel',
        need_text=False,
        body=dict(
            type='GBody',
            pipelines=[
                dict(
                    type='FeatureExtractorComponent',
                    from_layer='input',
                    to_layer='cnn_feat',
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type='GBackbone',
                                layers=[
                                    dict(type='ConvModule', in_channels=1, out_channels=32 + base_channel,
                                         kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg),  # 48, 192
                                    dict(type='ConvModule', in_channels=32 + base_channel,
                                         out_channels=64 + base_channel * 2, kernel_size=3,
                                         stride=1, padding=1, norm_cfg=norm_cfg),  # 48, 192 # c0
                                    dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0),  # 24, 96
                                    dict(type='BasicBlocks', inplanes=64 + base_channel * 2,
                                         planes=128 + base_channel * 4, blocks=1, stride=1, norm_cfg=norm_cfg,
                                         plug_cfg=dict(type='CBAM', gate_channels=128 + base_channel * 4,
                                                       reduction_ratio=16,
                                                       norm_cfg=dict(type='BN', momentum=0.01))),
                                    # 24, 96
                                    dict(type='ConvModule', in_channels=128 + base_channel * 4,
                                         out_channels=128 + base_channel * 4, kernel_size=3,
                                         stride=1, padding=1, norm_cfg=norm_cfg),  # 24, 96

                                    dict(type='NonLocal2d', in_channels=128 + base_channel * 4, sub_sample=True),  # c1
                                    dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0),  # 12, 49
                                    dict(type='BasicBlocks', inplanes=128 + base_channel * 4,
                                         planes=256 + base_channel * 8, blocks=4, stride=1, norm_cfg=norm_cfg,
                                         plug_cfg=dict(type='CBAM', gate_channels=256 + base_channel * 8,
                                                       reduction_ratio=16,
                                                       norm_cfg=dict(type='BN', momentum=0.01))),
                                    # 12, 48
                                    dict(type='ConvModule', in_channels=256 + base_channel * 8,
                                         out_channels=256 + base_channel * 8, kernel_size=3,
                                         stride=1, padding=1, norm_cfg=norm_cfg),  # 12, 48
                                    dict(type='NonLocal2d', in_channels=256 + base_channel * 8, sub_sample=True),  # c2
                                    dict(type='MaxPool2d', kernel_size=2, stride=(2, 2)),
                                    # 6, 24
                                    dict(type='BasicBlocks', inplanes=256 + base_channel * 8,
                                         planes=512 + base_channel * 16, blocks=7, stride=1, norm_cfg=norm_cfg,
                                         plug_cfg=dict(type='CBAM', gate_channels=512 + base_channel * 16,
                                                       reduction_ratio=16,
                                                       norm_cfg=dict(type='BN', momentum=0.01))),
                                    # 6, 24

                                    dict(type='ConvModule', in_channels=512 + base_channel * 16,
                                         out_channels=512 + base_channel * 16, kernel_size=3,
                                         stride=1, padding=1, norm_cfg=norm_cfg),  # 6, 24
                                    dict(type='BasicBlocks', inplanes=512 + base_channel * 16,
                                         planes=512 + base_channel * 16, blocks=5, stride=1, norm_cfg=norm_cfg,
                                         plug_cfg=dict(type='CBAM', gate_channels=512 + base_channel * 16,
                                                       reduction_ratio=16,
                                                       norm_cfg=dict(type='BN', momentum=0.01))),
                                    # 6, 24
                                    dict(type='NonLocal2d', in_channels=512 + base_channel * 16, sub_sample=True),  # c3
                                    dict(type='ConvModule', in_channels=512 + base_channel * 16,
                                         out_channels=512 + base_channel * 16, kernel_size=2,
                                         stride=(2, 1), norm_cfg=norm_cfg),  # 3, 23
                                    dict(type='BasicBlocks', inplanes=512 + base_channel * 16,
                                         planes=512 + base_channel * 16,
                                         blocks=3, stride=1, norm_cfg=norm_cfg,
                                         plug_cfg=dict(type='CBAM', gate_channels=512 + base_channel * 16,
                                                       reduction_ratio=16,
                                                       norm_cfg=dict(type='BN', momentum=0.01))),
                                    dict(type='ConvModule', in_channels=512 + base_channel * 16,
                                         out_channels=512 + base_channel * 16, kernel_size=2,
                                         stride=1, padding=0, norm_cfg=norm_cfg),  # 2, 24  # c4
                                ],
                            ),
                        ),
                        decoder=dict(
                            type='GFPN',
                            neck=[
                                dict(
                                    type='JunctionBlock',
                                    top_down=None,
                                    lateral=dict(
                                        from_layer='c4',
                                        type='ConvModule',
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation='relu',
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer='p5',
                                ),  # 32
                                # model/decoder/blocks/block2
                                dict(
                                    type='JunctionBlock',
                                    fusion_method='add',
                                    top_down=dict(
                                        from_layer='p5',
                                        upsample=dict(
                                            type='Upsample',
                                            # scale_factor=2,
                                            size=(6, 24),
                                            scale_bias=0,
                                            mode='bilinear',
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer='c3',
                                        type='ConvModule',
                                        in_channels=512 + base_channel * 16,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation='relu',
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer='p4',
                                ),  # 16
                                # model/decoder/blocks/block3
                                dict(
                                    type='JunctionBlock',
                                    fusion_method='add',
                                    top_down=dict(
                                        from_layer='p4',
                                        upsample=dict(
                                            type='Upsample',
                                            size=(12, 48),
                                            # scale_factor=2,
                                            scale_bias=0,
                                            mode='bilinear',
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer='c2',
                                        type='ConvModule',
                                        in_channels=256 + base_channel * 8,
                                        out_channels=512,
                                        kernel_size=1,
                                        activation='relu',
                                        norm_cfg=norm_cfg,
                                    ),
                                    post=None,
                                    to_layer='p3',
                                ),  # 8
                                # fusion the features
                                dict(
                                    type='JunctionBlock',
                                    fusion_method=None,
                                    top_down=dict(
                                        from_layer='p5',
                                        trans=dict(
                                            type='ConvModule',
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type='Conv'),
                                            norm_cfg=norm_cfg,
                                            activation='relu',
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type='Upsample',
                                            size=(6, 24),
                                            scale_bias=0,
                                            mode='bilinear',
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=None,
                                    post=None,
                                    to_layer='p5_1',
                                ),  # 6, 24
                                dict(
                                    type='JunctionBlock',
                                    fusion_method=None,
                                    top_down=dict(
                                        from_layer='p5_1',
                                        trans=dict(
                                            type='ConvModule',
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type='Conv'),
                                            norm_cfg=norm_cfg,
                                            activation='relu',
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type='Upsample',
                                            size=(12, 48),
                                            scale_bias=0,
                                            mode='bilinear',
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=None,
                                    post=None,
                                    to_layer='p5_2',
                                ),  # 12, 48
                                dict(
                                    type='JunctionBlock',
                                    fusion_method='add',
                                    top_down=dict(
                                        from_layer='p4',
                                        trans=dict(
                                            type='ConvModule',
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type='Conv'),
                                            norm_cfg=norm_cfg,
                                            activation='relu',
                                            inplace=True,
                                        ),
                                        upsample=dict(
                                            type='Upsample',
                                            size=(12, 48),
                                            scale_bias=0,
                                            mode='bilinear',
                                            align_corners=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer='p5_2',
                                    ),
                                    post=None,
                                    to_layer='p4_1',
                                ),  # 12, 48
                                dict(
                                    type='JunctionBlock',
                                    fusion_method='add',
                                    top_down=dict(
                                        from_layer='p3',
                                        trans=dict(
                                            type='ConvModule',
                                            in_channels=512,
                                            out_channels=512,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=dict(type='Conv'),
                                            norm_cfg=norm_cfg,
                                            activation='relu',
                                            inplace=True,
                                        ),
                                    ),
                                    lateral=dict(
                                        from_layer='p4_1',
                                    ),
                                    post=None,
                                    to_layer='p3_1',
                                ),  # 12, 48
                            ],
                        ),
                        collect=dict(type='CollectBlock', from_layer='p3_1'),
                    ),
                ),
                dict(
                    type='FeatureExtractorComponent',
                    from_layer='cnn_feat',
                    to_layer='non_local_feat',
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type='GBackbone',
                                layers=[
                                    dict(type='NonLocal2d', in_channels=512, sub_sample=True, )  # c0
                                ]),
                        ),
                        collect=dict(type='CollectBlock', from_layer='c0'),
                    ),
                ),
            ],
        ),
        head=dict(
            type='MultiHead',
            in_channels=512,
            num_class=num_class,
            from_layer='non_local_feat',
            batch_max_length=batch_max_length,
            pool=dict(
                type='AdaptiveAvgPool2d',
                output_size=1,
            ),
        ))


class CSTR(nn.Module):
    def __init__(
        self,
        FLAGS,
        tokenizer
    ):
        super(CSTR, self).__init__()       
        self.model = self._build_model(cstr_structure)

    

    def _build_model(self, cfg):

        model = build_model(cfg)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        return model


    # CSTR은 teacher_forcing_ratio 필요 없지만 format을 다른 것들과 맞추기 위해 그냥 arguement에만 있고 사용 안함
    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        output = self.model((input, expected))
        return output