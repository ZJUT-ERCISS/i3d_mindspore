# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" I3D network."""


from src.models.base import BaseClassifier
from src.utils.model_urls import model_urls
from src.utils.load_pretrained_model import LoadPretrainedModel
from src.models.pooling3d import AvgPooling3D
from src.models.i3d_backbone import InceptionI3d
from src.models.i3d_head import I3dHead
from src.models.inception3d import Inception3dModule

__all__ = ['i3d_rgb']


def i3d_rgb(in_channel: int = 3,
            num_classes: int = 400,
            keep_prob: float = 0.5,
            pretrained: bool = False,
            ) -> InceptionI3d:
    """
    TODO: introduction i3d network.

    Args:
        in_channel(int): Number of channel of input data. Default: 3.
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers. Default: 0.5.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a i3d model with uniform initialization for weight and bias. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from src.msvideo.models import i3d
        >>>
        >>> net = i3d()
        >>> x = ms.Tensor(np.ones([1, 3, 32, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)

    About i3d:

    TODO: i3d introduction.

    Citation:

    .. code-block::

        TODO: i3d Citation.
    """

    backbone_output_channel = 1024
    backbone = InceptionI3d(in_channels=in_channel)
    neck = AvgPooling3D(kernel_size=(2, 7, 7))
    head = I3dHead(in_channels=backbone_output_channel,
                   num_classes=num_classes,
                   dropout_keep_prob=keep_prob)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "I3D_kinetic_rgb"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
