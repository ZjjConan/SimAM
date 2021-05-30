_base_ = [
    '../_base_/models/faster_rcnn_r50simam_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x_lr0.01.py', '../_base_/default_runtime.py'
]

find_unused_parameters=True
