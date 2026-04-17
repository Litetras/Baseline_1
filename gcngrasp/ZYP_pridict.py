import torch, sys, os
import numpy as np
import json
import trimesh
import open3d as o3d  # 🚀 新增：用于 3D 可视化
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults
from ZYP_custom_loader import ZYPGraspGPTDatasetWrapper
from gcngrasp.data_specification import TASKS

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Pointnet2_PyTorch'))

def create_gripper_visual(grasp_matrix):
    """
    生成一个简单的 3D 平行夹爪几何体用于可视化
    默认 Z 轴为抓取接近方向，X 轴为夹爪开合方向
    """
    w = 0.04  # 夹爪半宽
    d = 0.06  # 夹爪掌部深度
    h = 0.05  # 夹爪指尖长度
    
    # 定义夹爪的 6 个关键点
    points = np.array([
        [0, 0, 0],          # 0: 基座原点
        [0, 0, d],          # 1: 掌部中心分叉点
        [-w, 0, d],         # 2: 左指根
        [w, 0, d],          # 3: 右指根
        [-w, 0, d + h],     # 4: 左指尖
        [w, 0, d + h]       # 5: 右指尖
    ])
    
    # 定义连接点的线段
    lines = [[0, 1], [2, 3], [2, 4], [3, 5]]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色夹爪
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 🚀 核心：将夹爪变换到预测的 4x4 抓取位姿上
    line_set.transform(grasp_matrix)
    return line_set

def run_inference(checkpoint_path, obj_path, grasp_json_path, instruction):
    # ---------------- 1. 解析自然语言指令 ----------------
    instruction = instruction.lower()
    task_name, obj_class = "unknown", "unknown"
    
    if "cut" in instruction and "knife" in instruction:
        task_name, obj_class = "cut", "knife"
    elif "hammer" in instruction or "pound" in instruction:
        task_name, obj_class = "hammer", "hammer"
    else:
        print(f"警告: 无法从指令 '{instruction}' 中识别支持的任务。")
        return

    print(f"\n[自然语言解析] 识别到任务: [{task_name}], 目标物体: [{obj_class}]")

    # ---------------- 2. 加载配置和模型 ----------------
    cfg = get_cfg_defaults()
    yaml_config_path = "/home/zyp/pan1/GraspGPT_public/cfg/train/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml"
    cfg.merge_from_file(yaml_config_path)
    
    cfg.defrost()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.base_dir = os.path.join(current_dir, '../data') 
    cfg.freeze()

    model = GraspGPT_plain.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.cuda().eval()
    model.prepare_data()

    # ---------------- 3. 准备 NLP 嵌入特征 ----------------
    orig_dataset = model.train_dataloader().dataset
    wrapper = ZYPGraspGPTDatasetWrapper("/home/zyp/Desktop/zyp_dataset7/graspgpt_format/dataset_index.json", orig_dataset, TASKS)
    
    # 从官方库借用该任务的语言大模型特征
    _, _, _, _, _, _, _, *nlp_features = wrapper[0]
    nlp_tensors = [torch.as_tensor(t).float().unsqueeze(0).cuda() for t in nlp_features]

    # ---------------- 4. 处理点云与预测 ----------------
    mesh = trimesh.load(obj_path, force='mesh')
    pc, _ = trimesh.sample.sample_surface(mesh, 4096)
    
    with open(grasp_json_path, 'r') as f:
        grasps = np.array(json.load(f)['grasps']['transforms'])

    results = []
    print("\n模型正在思考并对所有抓取进行打分...")
    for i, grasp in enumerate(grasps):
        R, t = grasp[:3, :3], grasp[:3, 3]
        pc_transformed = (R.T @ (pc - t).T).T
        
        ones = np.ones((4096, 1), dtype=np.float32)
        pc_4d = np.concatenate([pc_transformed, ones], axis=-1)
        pc_tensor = torch.tensor(pc_4d).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            logits = model.forward(pc_tensor, *nlp_tensors)
            prob = torch.sigmoid(logits).item()
            results.append((i, prob, grasp))

    # 按概率排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Top 3 推荐抓取位姿:")
    for rank in range(3):
        idx, p, _ = results[rank]
        print(f"Rank {rank+1} - 抓取索引 {idx}: 成功概率 {p:.4f}")

    # ---------------- 5. Open3D 终极可视化 ----------------
    print("\n正在启动 3D 可视化 (按 Q 键或关闭窗口以退出)...")
    best_grasp_matrix = results[0][2]

    # 将 trimesh 转换为 open3d mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.6, 0.6, 0.6])  # 把物体涂成高级灰

    # 生成预测出的最佳夹爪
    gripper_visual = create_gripper_visual(best_grasp_matrix)
    
    # 画一个坐标系帮助你看方向 (可选)
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([mesh_o3d, gripper_visual], window_name="GraspGPT Result")

if __name__ == "__main__":
    CKPT = "/home/zyp/pan1/GraspGPT_public/checkpoints/gcngrasp_split_mode_o_split_idx_0__2026-04-17-17-31/weights/best.ckpt"
    
    # # 测试 1: 锤子
    # OBJ = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset/hammer_11_up_handle.obj"
    # JSON = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset/hammer_11_up_handle_grasps.json"
    # COMMAND = "Please use the robot to grasp the hammer to pound the nail."
    
    # # 测试 2: 刀具 (你可以取消注释这行来测你的刀)
    OBJ = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset/kitchen_knife_6_up_handle.obj"
    JSON = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset/kitchen_knife_6_up_handle_grasps.json"
    COMMAND = "grasp the knife to cut"

    run_inference(CKPT, OBJ, JSON, COMMAND)