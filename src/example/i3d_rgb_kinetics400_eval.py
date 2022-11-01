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
""" I3D eval script. """

import argparse

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model

from src.data import transforms
from src.data import Kinetic400
from src.models import i3d_rgb
from src.utils.callbacks import EvalLossMonitor

def i3d_rgb_eval(args_opt):
    """I3D eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

    # Data Pipeline.
    dataset_eval = Kinetic400(path=args_opt.dataset_path,
                              split='test',
                              seq=args.seq,
                              seq_mode='discrete',
                              batch_size=args_opt.batch_size,
                              shuffle=False,
                              num_parallel_workers=args_opt.num_parallel_workers,
                              frame_interval=1
                              )
    dataset_transforms = [transforms.VideoShortEdgeResize(size=256, interpolation='linear'),
                          transforms.VideoCenterCrop([224, 224]),
                          transforms.VideoToTensor()]
    dataset_eval.transform = dataset_transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    network = i3d_rgb()

    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)
    print_cb = EvalLossMonitor(model)
    # Begin to train.
    print('[Start eval `{}`]'.format(args_opt.model_name))
    result = model.eval(dataset_eval,callbacks=[print_cb], dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I3d_grb Eval.")
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--seq', type=int, default=64, help='Number of frames of captured video.')
    parser.add_argument("--dataset_path", type=str, default="/home/publicfile/kinetics-400")
    parser.add_argument("--model_name", type=str, default="i3d_rgb")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of batch size.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="/home/zhengs/i3d_mindspore-main/script/i3d_rgb_kinetics400.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--dataset_sink_mode", default=False, help="The dataset sink mode.")
    parser.add_argument("--num_parallel_workers", type=int, default=1, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]

    i3d_rgb_eval(args)
