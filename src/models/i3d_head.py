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
"""I3D Head."""

from mindspore import nn
from mindspore import ops
from src.models.layers.unit3d import Unit3D
from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.HEAD)
class I3dHead(nn.Cell):
    """
    I3dHead definition

    Args:
        in_channels: Input channel.
        num_classes (int): The number of classes .
        dropout_keep_prob (float): A float value of prob.

    Returns:
        Tensor, output tensor.

    Examples:
        I3dHead(in_channels=2048, num_classes=400, dropout_keep_prob=0.5)
    """

    def __init__(self, in_channels, num_classes=400, dropout_keep_prob=0.5):
        super(I3dHead, self).__init__()
        self._num_classes = num_classes
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=in_channels,
            out_channels=self._num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            norm=None,
            has_bias=True)
        self.mean_op = ops.ReduceMean()
        self.squeeze = ops.Squeeze(3)

    def construct(self, x):
        x = self.logits(self.dropout(x))
        x = self.squeeze(self.squeeze(x))
        x = self.mean_op(x, 2)
        return x
