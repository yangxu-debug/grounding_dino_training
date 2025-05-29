_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = 'data/cc5k_coco_orisize/'

class_name = ("Space Bath","Space DraughtLobby","Space UserDefined","Space Undefined","Space Storage Oil","Space Storage","Space Closet WalkIn","Space Bath Shower","Space DressingRoom","Space Entry Lobby",
"Space Garage","Space Outdoor","Wall External","Door Swing Beside","Window Regular","Wall","Railing","Space Kitchen","Space Room","Space LivingRoom","Space Outdoor Balcony","Column FreeShape","Column Rectangle",
"SelectionControls","Space Storage Fuel","Space Bedroom","Space Hall","Door None Beside","Space Outdoor Patio Glazed","Space Outdoor Terrace","Space Outdoor CoveredArea","Space Entry","Door Slide Beside","Space ExerciseRoom Gym",
"Space Outdoor Porch","Space TechnicalRoom","Space Dining","Door Zfold Beside","Space Kitchen Kitchenette","Door Swing Opposite","Space RecreationRoom","Door RollUp Beside","Space Utility Laundry","Space Room Cold",
"Space Kitchen Open","Space CarPort","Space Den Fireplace","Space Storage Shed","Space Office","Space Alcove","Space Sauna","Space Outdoor Terrace Covered","Space Hall Corridor","Space Outdoor Garden",
"Space Outdoor Balcony Glazed","Space TechnicalRoom Boiler","Space Kitchen Scullery","Window Sauna","Space Elevated","Space Bedroom Guest","Door ParallelSlide Beside","Space Outdoor Patio","Space Attic",
"Space Storage Cold","Space Storage Wood","Space Utility Drying","Space Room HighCeiling","Space Outdoor Veranda Glazed","Space Outdoor Terrace Covered Open","Space Outdoor Terrace Roof","Space Library",
"Space Elevator","Space Outdoor Veranda","Space RetailSpace","Space SwimmingPool","Space Basement","Door Fold Beside","Space Storage Bike","Space OpenToBelow","Space Outdoor Pergola","Space Library Archive",
"Space Outdoor Kitchen","Space Bar","Window Full","Space Outdoor Balcony Covered","Space Garbage")

num_classes = len(class_name)
palette = [(220, 20, 60), (0, 80, 100), (119, 11, 32)]
metainfo = dict(classes=class_name, palette=palette)

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size = 16,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    batch_size = 4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

max_epoch = 80

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
