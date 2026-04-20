import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from visualize import get_gripper_control_points

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()

def encode_text(text, tokenizer, model, type=None):
    if type == 'od':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=300)
    elif type == 'td':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=200)
    elif type == 'li':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=21)
    else:
         raise ValueError(f'No such language embedding type: {type}')

    with torch.no_grad():
        output = model(**encoded_input)
        word_embedding = output[0]

    return word_embedding[0].cpu(), encoded_input['attention_mask'][0].cpu()

class ZYPGraspGPTDatasetWrapper(Dataset):
    def __init__(self, index_file, orig_dataset, task_list, transforms=None): 
        with open(index_file, 'r') as f:
            self.data = json.load(f)
        self.task_list = task_list
        self.transforms = transforms 
        
        self.class_list = getattr(orig_dataset, 'classes', None)
        if self.class_list is None:
            self.class_list = getattr(orig_dataset, 'obj_classes', None)
            
        if isinstance(self.class_list, dict):
            self.class_to_id = self.class_list
        elif isinstance(self.class_list, list):
            self.class_to_id = {c: i for i, c in enumerate(self.class_list)}
        else:
            self.class_to_id = {}

        # =======================================================
        # 🚀 提取官方数据库 NLP 特征
        # =======================================================
        self.nlp_cache = {}
        print(">>> [Dataset] 正在加载 BERT 模型以生成专属语言特征...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained("bert-base-uncased").eval()

        DB_BASE_DIR = "/home/zyp/pan1/GraspGPT_public/data/taskgrasp"

        unique_pairs = set([(item['task'], item['obj_class']) for item in self.data])
        
        print(f">>> [Dataset] 发现 {len(unique_pairs)} 种独特的 [任务-物体] 组合，正在生成 BERT 编码...")
        for task_str, obj_str in tqdm(unique_pairs, desc="生成文本特征"):
            mapped_obj = obj_str.replace('kitchen_knife', 'knife') 
            obj_db_path = os.path.join(DB_BASE_DIR, "obj_gpt_v2", mapped_obj, "descriptions", "0")
            if os.path.exists(obj_db_path):
                with open(os.path.join(obj_db_path, "all.txt"), 'r') as f:
                    obj_desc_txt = f.read().strip()
            else:
                obj_desc_txt = f"This is a {obj_str.replace('_', ' ')}. It is a common object."

            task_db_path = os.path.join(DB_BASE_DIR, "task_gpt_v2", task_str, "descriptions", "0")
            if os.path.exists(task_db_path):
                with open(os.path.join(task_db_path, "all.txt"), 'r') as f:
                    task_desc_txt = f.read().strip()
            else:
                task_desc_txt = f"The task is to {task_str} using the object."

            task_ins_txt = f"grasp the {obj_str.replace('_', ' ')} to {task_str}"

            obj_desc, obj_desc_mask = encode_text(obj_desc_txt, tokenizer, bert_model, type='od')
            task_desc, task_desc_mask = encode_text(task_desc_txt, tokenizer, bert_model, type='td')
            task_ins, task_ins_mask = encode_text(task_ins_txt, tokenizer, bert_model, type='li')

            self.nlp_cache[(task_str, obj_str)] = (
                obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask
            )

        del tokenizer
        del bert_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> [Dataset] 官方长文本特征提取完毕，已释放大模型显存。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task_str = item['task']
        obj_str = item['obj_class']
        
        task_id = self.task_list.index(task_str) if task_str in self.task_list else 0
        class_id = 0
        for k, v in self.class_to_id.items():
            if obj_str in k.lower() or k.lower() in obj_str:
                class_id = v
                break
                
        pc = np.load(item['pc_path']).astype(np.float32)
        grasp = np.load(item['grasp_path']).astype(np.float32)
        label = np.array(item['label'], dtype=np.float32)
        
        # 1. 点云采样至 4096
        if pc.shape[0] > 4096:
            indices = np.random.choice(pc.shape[0], 4096, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < 4096:
            pad_size = 4096 - pc.shape[0]
            pad_points = pc[np.random.choice(pc.shape[0], pad_size, replace=True)]
            pc = np.vstack((pc, pad_points))

# 2. 生成夹爪点
        grasp_pc_homo = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc_homo.T).T[:, :3].astype(np.float32)
        pc_combined = np.concatenate([pc, grasp_pc], axis=0) # [4103, 3]

        # 3. 🚨 提前生成第 4 个通道 (Latent Indicator) 🚨
        latent = np.concatenate([np.zeros(4096, dtype=np.float32), np.ones(7, dtype=np.float32)])
        latent = np.expand_dims(latent, axis=1) # [4103, 1]
        
        # 拼成官方要求的 4 通道数组 [4103, 4]
        pc_4d = np.concatenate([pc_combined, latent], axis=1) 

        # 4. 空间居中与防爆缩放 (仅对前3维 XYZ 进行操作)
        pc_mean = pc_4d[:4096, :3].mean(axis=0)
        pc_4d[:, :3] -= pc_mean
        
        max_dist = np.max(np.sqrt(np.sum(pc_4d[:, :3]**2, axis=1)))
        if max_dist > 0:
            pc_4d[:, :3] /= max_dist

        # 5. 🚨 核心修复：送入官方的数据增强 🚨
        if self.transforms is not None:
            # 官方接收 [4103, 4] 并返回 Tensor
            pc_tensor = self.transforms(pc_4d) 
        else:
            pc_tensor = torch.from_numpy(pc_4d).float()

        # 6. 提取与封装
        grasp_tensor = torch.from_numpy(grasp).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        nlp_tensors = self.nlp_cache[(task_str, obj_str)]
        
        return (
            pc_tensor, 
            grasp_tensor, 
            torch.tensor(task_id, dtype=torch.long), 
            torch.tensor(class_id, dtype=torch.long), 
            torch.tensor(0, dtype=torch.long), 
            torch.tensor(0, dtype=torch.long), 
            label_tensor, 
            *nlp_tensors
        )