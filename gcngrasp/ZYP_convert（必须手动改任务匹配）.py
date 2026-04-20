import os
import json
import glob
import numpy as np
import trimesh
from tqdm import tqdm

def build_graspgpt_dataset(src_grasp_dir, src_obj_dir, out_dir):
    """
    根据用户实际的文件命名规则转换数据
    """
    os.makedirs(out_dir, exist_ok=True)
    pc_dir = os.path.join(out_dir, "point_clouds")
    grasp_dir = os.path.join(out_dir, "grasps")
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(grasp_dir, exist_ok=True)

    dataset_index = []
    
    # 1. 遍历所有的 JSON 抓取文件
    json_files = glob.glob(os.path.join(src_grasp_dir, "*_grasps.json"))
    print(f"找到 {len(json_files)} 个抓取 JSON 文件，开始转换...")

    for json_path in tqdm(json_files):
        # 修正逻辑：从文件名推断 ID
        # 例如: hammer_11_down_handle_grasps.json -> ID 是 hammer_11_down_handle
        json_filename = os.path.basename(json_path)
        id_name = json_filename.replace('_grasps.json', '')
        
        # 2. 直接根据 ID 寻找对应的 OBJ 文件
        # 例如: hammer_11_down_handle.obj
        obj_filename = id_name + ".obj"
        obj_path = os.path.join(src_obj_dir, obj_filename)
        
        if not os.path.exists(obj_path):
            # print(f"跳过：未找到对应的模型文件 {obj_path}")
            continue

        # 3. 标签逻辑 (对比实验核心)
        # 规则：up_handle 是正样本 (1)，其他部位 (down_handle, head, blade等) 是负样本 (0)
        label = 0
        task = ""
        obj_class = ""

        if 'knife' in id_name.lower():
            obj_class = 'kitchen_knife'
            task = 'cut'
            if '_up_handle' in id_name:
                label = 1
        elif 'hammer' in id_name.lower():
            obj_class = 'hammer'
            task = 'hammer' # 对应 GraspGPT TASKS 列表中的 pound 或 hammer
            if '_up_handle' in id_name:
                label = 1
        else:
            continue

        # 4. 加载抓取数据
        try:
            with open(json_path, 'r') as f:
                grasp_data = json.load(f)
            # 假设你的 transforms 是一个三维数组 [N, 4, 4]
            grasps = np.array(grasp_data['grasps']['transforms'])
        except Exception as e:
            print(f"读取 JSON 失败 {json_filename}: {e}")
            continue

        # 5. 模型采样为点云 (GraspGPT 输入要求)
        try:
            mesh = trimesh.load(obj_path, force='mesh')
            # 统一采样 4096 个点
            pc, _ = trimesh.sample.sample_surface(mesh, 4096)
            pc_save_path = os.path.join(pc_dir, f"{id_name}_pc.npy")
            np.save(pc_save_path, pc.astype(np.float32))
        except Exception as e:
            print(f"处理模型失败 {obj_filename}: {e}")
            continue

        # 6. 保存每一个抓取位姿，并记录到索引
        for i, grasp_mat in enumerate(grasps):
            grasp_save_name = f"{id_name}_grasp_{i}.npy"
            grasp_save_path = os.path.join(grasp_dir, grasp_save_name)
            np.save(grasp_save_path, grasp_mat.astype(np.float32))
            
            dataset_index.append({
                'obj_id': id_name,
                'obj_class': obj_class,
                'task': task,
                'pc_path': pc_save_path,
                'grasp_path': grasp_save_path,
                'label': int(label)
            })

    # 7. 导出最终索引
    index_path = os.path.join(out_dir, "dataset_index.json")
    with open(index_path, 'w') as f:
        json.dump(dataset_index, f, indent=4)
        
    print(f"\n转换完成！")
    print(f"输出目录: {out_dir}")
    print(f"总计生成的评估对数量: {len(dataset_index)}")

if __name__ == "__main__":
    # 请根据你的实际路径修改
    # 建议使用绝对路径
    GRASP_DIR = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset"
    OBJ_DIR = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
    OUTPUT_DIR = "/home/zyp/Desktop/zyp_dataset7/graspgpt_format"
    
    build_graspgpt_dataset(GRASP_DIR, OBJ_DIR, OUTPUT_DIR)