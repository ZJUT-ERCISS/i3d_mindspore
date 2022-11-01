# PYTHON_PATH="/home/zhengs/i3d_mindspore-main"
# export PYTHONPATH=$PYTHON_PATH
# python /home/zhengs/i3d_mindspore-main/src/example/i3d_rgb_kinetics400_eval.py --data_url /home/publicfile/kinetics-400

PYTHON_PATH="/home/zhengs/i3d_mindspore-main"
export PYTHONPATH=$PYTHON_PATH
python /home/zhengs/i3d_mindspore-main/src/example/i3d_rgb_kinetics400_train.py --data_url /home/publicfile/kinetics-400 \
   --pretrained_model ./i3d_rgb_kinetics400.ckpt > ./train_result.log 2>&1