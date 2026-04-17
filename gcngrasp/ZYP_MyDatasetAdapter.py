import torch
import numpy as np
import random
import os
import sys
from omegaconf import OmegaConf








# ==================== 路径强行注入区域 ====================

# 1. 动态添加 GraspGPT 的根目录，避免触发原作者有 Bug 的 __init__.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# 2. 【关键新增】强行添加你的 GraspGen 项目根目录
# 这样 Python 就能跨文件夹识别到 "grasp_gen.dataset.dataset" 了
sys.path.append('/home/zyp/GraspGen')

# ==========================================================

# 3. 从 GraspGPT 根目录导入控制点获取函数
from visualize import get_gripper_control_points

# 4. 导入你的 GraspGen 数据集类（现在绝不会报 ModuleNotFoundError 了）
from grasp_gen.dataset.dataset import ObjectPickDataset

# 5. 把原版 SGNLoader.py 里的点云归一化函数原封不动地搬过来
def pc_normalize(pc, grasp, pc_scaling=True):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    grasp[:3, 3] -= centroid

    if pc_scaling:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = np.concatenate([pc, np.ones([pc.shape[0], 1])], axis=1)
        scale_transform = np.diag([1 / m, 1 / m, 1 / m, 1])
        pc = np.matmul(scale_transform, pc.T).T
        pc = pc[:, :3]
        grasp = np.matmul(scale_transform, grasp)
    return pc, grasp




class GraspGPTAdapterDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split='train'):
        self.split = split
        
        # ================== 核心修改区域 ==================
        # 1. 强行读取你原来的 GraspGen 配置文件
        graspgen_cfg_path = "/home/zyp/GraspGen/scripts/config.yaml" 
        my_cfg = OmegaConf.load(graspgen_cfg_path)
        
        # 2. 提取参数字典
        dataset_args = ObjectPickDataset.from_config(my_cfg.data)
            
        # 3. 强制指定 split
        dataset_args["split"] = split 
        
        # === 新增/修改：强制指定版本、路径和缓存目录 ===
        
# 强行覆盖 yaml 里的 v1，让它走 JSON 读取逻辑
        dataset_args["dataset_version"] = "v2" 
        
        # 1. 极其关键：结尾绝对不能有斜杠 '/'
        dataset_args["root_dir"] = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
        dataset_args["object_root_dir"] = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
        dataset_args["grasp_root_dir"] = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
        dataset_args["cache_dir"] = "/home/zyp/Desktop/zyp_dataset7/tutorial/cache"
        


        # ==================================
            
        # 4. 正式实例化你的数据集
        self.my_dataset = ObjectPickDataset(**dataset_args)


        self.tasks = ["cut", "pass me", "pound", "pull the nail"]  
        self.classes = ["knife", "hammer", "object"]          

    @property
    def weights(self):
        return np.array([0.5, 0.5])


    def __len__(self):
        # 数据集长度与你原数据集保持一致
        return len(self.my_dataset)

    def __getitem__(self, idx):
        # 1. 从你的数据集获取数据（包含该物体的点云，以及 N 个正负抓取）
        data = self.my_dataset[idx]
        
        # 处理异常读取
        if data.get("invalid", False):
            # 如果你的数据加载失败，随机返回另一个（或者你之前怎么处理的就怎么处理）
            return self.__getitem__(random.randint(0, len(self.my_dataset) - 1))

# 2. 核心：从当前物体中，随机抽取【1个】样本喂给 GraspGPT
        pos_grasps = data.get("positive_grasps", [])
        neg_grasps = data.get("negative_grasps", [])
        
        # 50% 的概率抽负样本，50% 抽正样本
        if len(neg_grasps) > 0 and (len(pos_grasps) == 0 or random.random() < 0.5):
            sample_idx = random.randint(0, len(neg_grasps) - 1)
            grasp_matrix = neg_grasps[sample_idx]
            label = 0.0
        elif len(pos_grasps) > 0:
            sample_idx = random.randint(0, len(pos_grasps) - 1)
            grasp_matrix = pos_grasps[sample_idx]
            label = 1.0
        else:
            # 如果这个物体既没正样本也没负样本，直接换下一个物体
            return self.__getitem__(random.randint(0, len(self.my_dataset) - 1))
            
        # 统一转为 numpy 矩阵
        if hasattr(grasp_matrix, "cpu"):
            grasp_matrix = grasp_matrix.cpu().numpy()

        # 3. 映射 Task ID 和 Class ID
        # 根据你之前生成的 strict_text 或 natural_text 进行映射
        obj_name = os.path.basename(self.my_dataset.scenes[idx]).split('.')[0].lower()
        
        # 判断类别 ID
        if "knife" in obj_name:
            class_id = self.classes.index("knife")
        elif "hammer" in obj_name:
            class_id = self.classes.index("hammer")
        else:
            class_id = self.classes.index("object")
            
        # 判断任务 ID (这里你可以写一个简单的规则，或者直接从 data 里读取你存进去的 label)
        task_text = data.get("natural_text", "")
        if "cut" in task_text or "chop" in task_text:
            task_id = self.tasks.index("cut")
        elif "pass" in task_text or "hand" in task_text:
            task_id = self.tasks.index("pass me")
        elif "pound" in task_text or "pin" in task_text:
            task_id = self.tasks.index("pound")
        elif "pull" in task_text:
            task_id = self.tasks.index("pull the nail")
        else:
            task_id = 0 # 默认兜底
            
        instance_id = 0 # 你的任务如果不涉及同类物体的 instance 区分，全给0即可

# 4. 核心：复刻 GraspGPT 的 夹爪-物体 拼接逻辑
        pc = data["points"].cpu().numpy()
        
        # === 新增安全锁：如果多了一个维度，强制把它压平为 (N, 3) ===
        if pc.ndim > 2:
            pc = pc.reshape(-1, 3)
            
        # 尝试获取颜色，没有就给全白
        if "rgb" in data:
            pc_color = data["rgb"].cpu().numpy()
            if pc_color.ndim > 2:
                pc_color = pc_color.reshape(-1, 3)
        else:
            pc_color = np.ones_like(pc) * 255.0

        # 获取标准的夹爪控制点，并根据当前的 grasp_matrix 进行空间变换
        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp_matrix, grasp_pc.T).T[:, :3]

        # 构造 Latent 标识符：物体点为 0，夹爪点为 1
        latent = np.concatenate([np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)

        # 拼接 XYZ 坐标
        pc_combined = np.concatenate([pc, grasp_pc], axis=0)

        # GraspGPT 的数据预处理：将物体和夹爪整体进行中心化
        pc_combined, grasp_matrix = pc_normalize(pc_combined, grasp_matrix)

# 最终输入点云变成 4 维 (X, Y, Z, Latent)
        pc_combined = np.concatenate([pc_combined, latent], axis=1)
        
        # === 精度转换补丁：防止 PointNet++ C++ 底层崩溃 ===
        pc_combined = pc_combined.astype(np.float32)
        pc_color = pc_color.astype(np.float32)
        grasp_matrix = grasp_matrix.astype(np.float32)
        label = np.float32(label)
        # ============================================

# === 终极补丁：满足 GraspGPT 硬编码的 13 个变量拆包要求 ===
        # 完美模拟 NLP 的序列维度：特征为 (SeqLen, 768)，Mask 为 (SeqLen,)
        dummy_seq_len = 1  # 序列长度给 1 即可，反正全是 0 不影响结果
        dummy_feature_dim = 768
        
        # 1. 文本特征 (二维：长度 x 维度)
        obj_desc = np.zeros((dummy_seq_len, dummy_feature_dim), dtype=np.float32)
        task_desc = np.zeros((dummy_seq_len, dummy_feature_dim), dtype=np.float32)
        task_ins = np.zeros((dummy_seq_len, dummy_feature_dim), dtype=np.float32)
        
        # 2. Attention Mask (一维：长度)
        obj_desc_mask = np.zeros((dummy_seq_len,), dtype=np.float32)
        task_desc_mask = np.zeros((dummy_seq_len,), dtype=np.float32)
        task_ins_mask = np.zeros((dummy_seq_len,), dtype=np.float32)

        # 返回 13 个参数
        return (
            pc_combined, pc_color, task_id, class_id, instance_id, grasp_matrix, label,
            obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask
        )