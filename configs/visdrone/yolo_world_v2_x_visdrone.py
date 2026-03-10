# In your custom_visdrone_config.py
dataset_type = 'MultiModalDataset'
data_root = 'data/VisDrone/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file='annotations/train_coco.json',
            data_prefix=dict(img='train/images/')),
        class_text_path='data/texts/visdrone_texts.json',
        pipeline=train_pipeline))
