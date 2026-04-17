import argparse
import os
import sys
# 强行将 Pointnet2_PyTorch 目录加入系统路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Pointnet2_PyTorch'))
import copy
import pytorch_lightning as pl
import torch
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults
# 在 train.py 顶部添加：
from ZYP_custom_loader import ZYPGraspGPTDatasetWrapper
from gcngrasp.data_specification import TASKS
from torch.utils.data import DataLoader, random_split

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
    
    # ==== 删除了这里原本错误的 dataset 实例化 ====

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
        monitor="val_acc",  
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

# ---------------- 核心修复点 ---------------- 
    # 手动触发数据准备，否则 train_dset 属性还没生成
    print("正在初始化原始数据集...")
    model.prepare_data()

    # ---------------- 核心修改区开始 ---------------- 
    # 1. 必须先获取原始数据加载器，把它作为“特征源”
    orig_train_loader = model.train_dataloader()
    orig_dataset = orig_train_loader.dataset

    # 2. 导入官方任务列表
    from gcngrasp.data_specification import TASKS
    
    # 3. 传入完整的 3 个参数！
    index_path = "/home/zyp/Desktop/zyp_dataset7/graspgpt_format/dataset_index.json"
    custom_dataset = ZYPGraspGPTDatasetWrapper(index_path, orig_dataset, TASKS)
    
    # 4. 将你的数据划分为训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # 5. 生成你专属的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    # ---------------- 核心修改区结束 ----------------


    # 初始化 Trainer
    trainer = pl.Trainer(
        accelerator="gpu",                 
        devices=all_gpus,                  
        max_epochs=cfg.epochs,
        callbacks=[early_stop_callback, checkpoint_callback], 
        default_root_dir=log_dir           
    )
    
    # 传入你自己的 Dataloader！
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

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