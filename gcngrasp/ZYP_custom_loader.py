import os, sys
# 确保能够找到上级目录的 visualize.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from visualize import get_gripper_control_points

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class ZYPGraspGPTDatasetWrapper(Dataset):
    def __init__(self, index_file, orig_dataset, task_list):
        with open(index_file, 'r') as f:
            self.data = json.load(f)
        self.orig_ds = orig_dataset
        self.task_list = task_list
        
        # 尝试推断原数据集的类别字典
        self.class_list = getattr(orig_dataset, 'classes', None)
        if self.class_list is None:
            self.class_list = getattr(orig_dataset, 'obj_classes', None)
            
        if isinstance(self.class_list, dict):
            self.class_to_id = self.class_list
        elif isinstance(self.class_list, list):
            self.class_to_id = {c: i for i, c in enumerate(self.class_list)}
        else:
            self.class_to_id = {}

        # 构建缓存，"偷取"原版数据集的语言大模型特征 (NLP Embeddings)
        self.nlp_cache = {}
        print("正在扫描原始数据集以提取语言特征 (NLP Embeddings)...")
        
        # 扫描前2000条即可收集全常用物体的语言特征
        for i in tqdm(range(min(2000, len(orig_dataset))), desc="提取语言特征缓存"):
            try:
                sample = orig_dataset[i]
                task_id = int(sample[2])
                class_id = int(sample[3])
                if (task_id, class_id) not in self.nlp_cache:
                    # 截取最后6个语言元素 (obj_desc, mask, task_desc, mask, task_ins, mask)
                    self.nlp_cache[(task_id, class_id)] = sample[7:] 
            except Exception:
                continue
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task_str = item['task']
        obj_str = item['obj_class']
        
        # 获取整数 ID
        task_id = self.task_list.index(task_str) if task_str in self.task_list else 0
        
        class_id = 0
        for k, v in self.class_to_id.items():
            if obj_str in k.lower() or k.lower() in obj_str:
                class_id = v
                break
                
        # 加载点云和抓取
        pc = np.load(item['pc_path']).astype(np.float32)
        grasp = np.load(item['grasp_path']).astype(np.float32)
        label = np.array(item['label'], dtype=np.float32)
        
        # 💡 严格控制物体点云在 4096 个点 (保持在原有坐标系，不进行局部坐标系变换)
        if pc.shape[0] > 4096:
            indices = np.random.choice(pc.shape[0], 4096, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < 4096:
            pad_size = 4096 - pc.shape[0]
            pad_points = pc[np.random.choice(pc.shape[0], pad_size, replace=True)]
            pc = np.vstack((pc, pad_points))

        # =======================================================
        # 🚀 核心修复步骤：官方 4103 拼接法 (4096 物体点 + 7 夹爪点)
        # =======================================================
        
        # 1. 获取官方夹爪的 7 个控制点 (齐次坐标形状为 [7, 4])
        grasp_pc_homo = get_gripper_control_points()
        
        # 2. 利用抓取位姿 grasp 将夹爪点变换到与物体相同的坐标系中
        # 提取前三列 XYZ 坐标，结果形状为 [7, 3]
        grasp_pc = np.matmul(grasp, grasp_pc_homo.T).T[:, :3]
        
        # 3. 拼接 XYZ 坐标 (4096 + 7 = 4103个点) -> 形状 [4103, 3]
        pc_combined = np.concatenate([pc, grasp_pc], axis=0)
        
        # 4. 构造 Latent Indicator 掩码通道 (物体点设为0，夹爪点设为1) -> 形状 [4103, 1]
        latent = np.concatenate([np.zeros(4096, dtype=np.float32), np.ones(7, dtype=np.float32)])
        latent = np.expand_dims(latent, axis=1)
        
        # 5. 组合成最终的 4 通道输入 (XYZ + Latent) -> 形状 [4103, 4]
        pc_4d = np.concatenate([pc_combined, latent], axis=1)

        # 转为 Tensor
        pc_tensor = torch.tensor(pc_4d, dtype=torch.float32)
        grasp_tensor = torch.tensor(grasp, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # 获取拼接语言特征 (如果字典找不到对应物体，随便拿一个兜底，避免崩库)
        nlp_tensors = self.nlp_cache.get((task_id, class_id))
        if nlp_tensors is None and len(self.nlp_cache) > 0:
            nlp_tensors = list(self.nlp_cache.values())[0]
        
        # 严格返回 13 个元素的 Tuple，匹配 GraspGPT_plain 的 unpack 需求
        return (
            pc_tensor, 
            grasp_tensor, 
            torch.tensor(task_id, dtype=torch.long), 
            torch.tensor(class_id, dtype=torch.long), 
            torch.tensor(0, dtype=torch.long), # instance_id 占位
            torch.tensor(0, dtype=torch.long), # grasp_id 占位
            label_tensor, 
            *nlp_tensors
        )