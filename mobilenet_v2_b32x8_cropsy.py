_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
    init_cfg = dict(
        type='Pretrained', 
        checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth', 
        prefix='backbone')
    ),
    head=dict(
        num_classes=3,
        topk=(1, 2, 3),
    ))

dataset_type = 'ImageNet'
img_norm_cfg = dict(
     mean=[124.508, 116.050, 106.438],
     std=[58.577, 57.310, 57.437],
     to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# dataset settings
data = dict(
    samples_per_gpu=64, #32,
    workers_per_gpu=2,
    train=dict(
        #type=dataset_type,
        data_prefix='../formatted_dataset/train',
        classes='../formatted_dataset/classes.txt'#,
        #pipeline=train_pipeline
        ),
    val=dict(
        #type=dataset_type,
        data_prefix='../formatted_dataset/val',
        ann_file='../formatted_dataset/val.txt',
        classes='../formatted_dataset/classes.txt'#,
        #pipeline=test_pipeline
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        #type=dataset_type,
        data_prefix='../formatted_dataset/test',
        ann_file='../formatted_dataset/test.txt',
        classes='../formatted_dataset/classes.txt'#,
        #pipeline=test_pipeline
        ))

evaluation = dict(metric_options={'topk':(1, 2, 3)})

# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=8)

#load_from = 'checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'