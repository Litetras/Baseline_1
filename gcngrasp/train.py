import argparse
import os
import copy
import pytorch_lightning as pl
import torch
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults

def get_timestamp():
    import datetime
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    day_month_year = '{}-{}-{}-{}-{}'.format(year, month, day, hour, minute)
    return day_month_year


def main(cfg):
    
    model = GraspGPT_plain(cfg)

    if cfg.pretraining_mode == 1:
        weight_file = os.path.join(cfg.log_dir, cfg.pretrained_weight_file)
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                'Unable to find pre-trained pointnet file {}'.format(weight_file))

        pretrained_dict = torch.load(weight_file)['state_dict']
        model_dict = model.state_dict()

        # 复制 PointNet 层的权重
        layers_updated = []
        for k in model_dict.keys():
            if k.find('SA_modules') >= 0 and (
                    k.find('bias') >= 0 or k.find('weight') >= 0):
                model_dict[k] = copy.deepcopy(pretrained_dict[k])
                layers_updated.append(k)

        model.load_state_dict(model_dict)

        print('Updated {} out of {} layers with pretrained weights'.format(
            len(layers_updated), len(list(model_dict.keys()))))

    exp_name = "{}_{}".format(cfg.name, get_timestamp())
    log_dir = os.path.join(cfg.log_dir, exp_name)

    # 1. 配置早停回调
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_acc",  # 确保你的模型内部 log 了这个名称
        patience=cfg.patience,
        mode="max"
    )

    # 2. 配置模型检查点回调
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=os.path.join(log_dir, 'weights'),
        filename="best",
        verbose=True,
    )

    # 3. GPU 设备设置
    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])

    # 4. 初始化 Trainer (针对 PyTorch Lightning 2.0+ 进行了修正)
    trainer = pl.Trainer(
        accelerator="gpu",                 # 指定使用 GPU
        devices=all_gpus,                  # 指定具体的设备列表，例如 [0]
        max_epochs=cfg.epochs,
        callbacks=[early_stop_callback, checkpoint_callback], # 所有回调放入列表
        default_root_dir=log_dir           # 替代原来的 default_save_path
    )
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument(
        '--cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise FileNotFoundError(args.cfg_file)

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    if torch.cuda.is_available():
        print('Cuda is available, make sure you are training on GPU')

    if args.gpus == -1:
        args.gpus = [0, ]
    cfg.gpus = args.gpus

    if args.batch_size > -1:
        cfg.batch_size = args.batch_size

    cfg.freeze()
    print(cfg)
    main(cfg)