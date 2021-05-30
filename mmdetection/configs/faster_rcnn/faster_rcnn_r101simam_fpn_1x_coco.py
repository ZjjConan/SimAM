_base_ = './faster_rcnn_r50simam_fpn_1x_coco.py'
model = dict(pretrained='checkpoints/simam-net/resnet101.pth.tar', backbone=dict(depth=101))
