import sys
import os
import open3d as o3d  # 必须在最前面导入
import torch
import numpy as np
import copy
import json
import trimesh

# 1. 导入官方可视化工具
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(CODE_DIR)
from visualize import draw_scene, get_gripper_control_points

# 2. 导入模型和工具
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults
from ZYP_custom_loader import ZYPGraspGPTDatasetWrapper
from gcngrasp.data_specification import TASKS

DEVICE = "cuda"

def main():
    # ==========================================
    # 1. 交互式自然语言输入
    # ==========================================
    print("-" * 50)
    task_ins_txt = input('Please input a natural language instruction (e.g., grasp the knife to cut): ').lower()
    
    task_name, obj_class = "unknown", "unknown"
    if "cut" in task_ins_txt and "knife" in task_ins_txt:
        task_name, obj_class = "cut", "kitchen_knife"
    elif "hammer" in task_ins_txt or "pound" in task_ins_txt:
        task_name, obj_class = "hammer", "hammer"
    else:
        print("警告：无法匹配任务，默认使用 hammer")
        task_name = "hammer"
    print(f"解析结果 -> 任务: {task_name}")

    # 测试路径
    OBJ_PATH = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset/kitchen_knife_6_up_handle.obj"
    JSON_PATH = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset/kitchen_knife_6_up_handle_grasps.json"
    CKPT_PATH = "/home/zyp/pan1/GraspGPT_public/checkpoints/gcngrasp_split_mode_o_split_idx_0__2026-04-17-17-31/weights/best.ckpt"

    # ==========================================
    # 2. 模型与配置加载
    # ==========================================
    cfg = get_cfg_defaults()
    cfg.merge_from_file("/home/zyp/pan1/GraspGPT_public/cfg/train/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml")
    cfg.defrost()
    cfg.base_dir = os.path.join(CODE_DIR, 'data')
    cfg.freeze()

    model = GraspGPT_plain.load_from_checkpoint(CKPT_PATH, cfg=cfg)
    model.to(DEVICE).eval()
    model.prepare_data()

    orig_dataset = model.train_dataloader().dataset
    wrapper = ZYPGraspGPTDatasetWrapper("/home/zyp/Desktop/zyp_dataset7/graspgpt_format/dataset_index.json", orig_dataset, TASKS)
    _, _, _, _, _, _, _, *nlp_features = wrapper[0]
    nlp_tensors = [torch.as_tensor(t).float().unsqueeze(0).cuda() for t in nlp_features]

    # ==========================================
    # 3. 处理物体点云 (⚠️ 移除去中心化，保持绝对对齐)
    # ==========================================
    mesh = trimesh.load(OBJ_PATH, force='mesh')
    pc, _ = trimesh.sample.sample_surface(mesh, 4096)
    
    # ❌ 移除了 pc -= pc_mean 的操作，保证特征不变！

    with open(JSON_PATH, 'r') as f:
        grasps = np.array(json.load(f)['grasps']['transforms'])

    results = []
    print("\n模型正在推理打分...")
    
    # ==========================================
    # 4. 推理循环 (严格 4103 拼接)
    # ==========================================
    for i, grasp in enumerate(grasps):
        grasp_pc_homo = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc_homo.T).T[:, :3]
        
        pc_combined = np.concatenate([pc, grasp_pc], axis=0)
        
        latent = np.concatenate([np.zeros(4096, dtype=np.float32), np.ones(7, dtype=np.float32)])
        latent = np.expand_dims(latent, axis=1)
        
        pc_4d = np.concatenate([pc_combined, latent], axis=1)
        pc_tensor = torch.tensor(pc_4d).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            logits = model.forward(pc_tensor, *nlp_tensors)
            prob = torch.sigmoid(logits).item()
            results.append(prob)

# ==========================================
    # 5. 官方可视化调用 (包含坐标系修复与 NameError 修复)
    # ==========================================
    probs = np.array(results)
    
    K = min(150, len(grasps))
    topk_inds = probs.argsort()[-K:][::-1]
    
    best_probs = probs[topk_inds]
    best_grasps = grasps[topk_inds]

    print("\nTop 5 推荐抓取概率:")
    for rank, p in enumerate(best_probs):
        print(f"Rank {rank+1}: {p:.4f}")

    # 生成颜色 (概率越高越绿)
    grasp_colors = np.stack([np.ones(K) - best_probs, best_probs, np.zeros(K)], axis=1)

    print("\n启动 Open3D 官方场景可视化...")

    # 🚀 视觉修复补丁：将你的 Z-轴接近 转换为 visualize.py 期望的 X-轴接近
    # 绕局部 Y 轴旋转 -90 度
    R_align = np.array([
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float32)
    
    # 仅在画图前旋转夹爪，不影响评分数据
    best_grasps_vis = np.matmul(best_grasps, R_align)

    # ✅ 修复点：移除了之前报错的 pc_mean 相关逻辑
    # 直接使用原始点云 pc 和变换后的位姿 best_grasps_vis
    draw_scene(
        pc,
        best_grasps_vis,
        subtract_pc_mean=False, # 确保可视化函数内部也不要去动坐标
        grasp_colors=list(grasp_colors),
        max_grasps=K,
        window_name="GraspGPT Result Visualization"
    )

if __name__ == "__main__":
    main()