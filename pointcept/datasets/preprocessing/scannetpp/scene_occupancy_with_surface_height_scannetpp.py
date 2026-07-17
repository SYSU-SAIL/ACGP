import os
import pickle
import random
from scipy.ndimage import distance_transform_edt
import numpy as np
from matplotlib.colors import ListedColormap
import open3d as o3d
from matplotlib import pyplot as plt
from pointcept.custom.scannetpp_constants import CLASS_LABELS_PP, INST_LABELS_PP

def visualize_pointcloud(coord, color=None):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coord)  # 设置点坐标
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # 坐标轴长度
        origin=[0, 0, 0]  # 原点位置
    )
    if color is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(color / 255.0)
    o3d.visualization.draw_geometries([pointcloud, coordinate_frame])


def visualize_occupancy_map(occupancy_map, x_bins=None, y_bins=None):
    colors = ['gray', 'white', 'black']  # 灰色（未知）、白色（空闲）、黑色（占用）
    cmap_custom = ListedColormap(colors)
    imshow_kwargs = dict(
        cmap=cmap_custom,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
    )
    if x_bins is not None and y_bins is not None and len(x_bins) > 0 and len(y_bins) > 0:
        grid_size_x = float(x_bins[1] - x_bins[0]) if len(x_bins) > 1 else 0.1
        grid_size_y = float(y_bins[1] - y_bins[0]) if len(y_bins) > 1 else 0.1
        imshow_kwargs["extent"] = [
            float(x_bins[0]),
            float(x_bins[-1]) + grid_size_x,
            float(y_bins[0]),
            float(y_bins[-1]) + grid_size_y,
        ]

    plt.imshow(occupancy_map, **imshow_kwargs)  # origin='lower' 让可视化以左下角为原点

    # 添加颜色条和标签
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Unknown', 'Free', 'Occupied'])  # 设置颜色条标签

    plt.title("Occupancy Map (-1=Unknown, 0=Free, 1=Occupied)")
    if x_bins is not None and y_bins is not None:
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
    else:
        plt.xticks([])
        plt.yticks([])  # 未提供真实坐标时隐藏坐标轴刻度
    plt.show()


def compute_per_cell_floor_height_map(
    points,
    semantic_label,
    x_bins,
    y_bins,
    floor_label_id=2,
    floor_z_range=(0.0, 0.2),
):
    """
    计算每个网格单元的局部地面高度。

    策略（按优先级）：
      1. 场景中存在语义地面点（floor_label_id）：
           - 有地面点的格子 → 取该格内地面点 z 均值
           - 无地面点的格子 → 最近邻插值（从已知地面格子传播）
      2. 场景中不存在语义地面点：
           - 使用 z ∈ floor_z_range 的点估计，取该格内这些点的 z 最小值
           - 仍无数据的格子 → 最近邻插值，最终回退到 floor_z_range 均值

    参数:
        points         : (N, 3) 点云坐标
        semantic_label : (N,) 语义标签，可为 None
        x_bins         : (W+1,) x 轴网格边界
        y_bins         : (H+1,) y 轴网格边界
        floor_label_id : 地面语义类别 ID（ScanNet++ 中为 2）
        floor_z_range  : 无地面标签时，认为地面所在的 z 绝对范围

    返回:
        floor_height_map : (H, W) float32，每格局部地面 z 坐标
        global_floor_z   : float，全局地面参考 z（调试用）
    """
    H = len(y_bins) - 1
    W = len(x_bins) - 1

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 向量化分配网格索引
    xi = np.digitize(x, x_bins) - 1
    yi = np.digitize(y, y_bins) - 1
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi_b = xi[in_bounds]
    yi_b = yi[in_bounds]
    z_b  = z[in_bounds]

    floor_height_map = np.full((H, W), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 判断场景是否存在语义地面点
    # ------------------------------------------------------------------ #
    has_floor_labels = (
        semantic_label is not None
        and np.any(semantic_label == floor_label_id)
    )

    if has_floor_labels:
        # --- 策略 1：用语义地面点估计每格地面高度 ---
        sl_b = semantic_label[in_bounds]
        floor_mask = sl_b == floor_label_id
        xif, yif, zf = xi_b[floor_mask], yi_b[floor_mask], z_b[floor_mask]

        z_sum = np.zeros((H, W), dtype=np.float64)
        z_cnt = np.zeros((H, W), dtype=np.int32)
        np.add.at(z_sum, (yif, xif), zf)
        np.add.at(z_cnt, (yif, xif), 1)
        valid = z_cnt > 0
        floor_height_map[valid] = z_sum[valid] / z_cnt[valid]

        global_floor_z = float(np.mean(zf))

    else:
        # --- 策略 2：无地面标签，使用 z 在 floor_z_range 内的点 ---
        z_lo, z_hi = floor_z_range
        range_mask = (z_b >= z_lo) & (z_b <= z_hi)

        if np.any(range_mask):
            xir, yir, zr = xi_b[range_mask], yi_b[range_mask], z_b[range_mask]
            z_min_map = np.full((H, W), np.inf, dtype=np.float64)
            np.minimum.at(z_min_map, (yir, xir), zr)
            has_range = z_min_map < np.inf
            floor_height_map[has_range] = z_min_map[has_range]
            global_floor_z = float(np.min(zr))
        else:
            global_floor_z = float(np.mean(floor_z_range))

    # ------------------------------------------------------------------ #
    # 对仍为 nan 的格子做最近邻插值（从已知格子传播）
    # ------------------------------------------------------------------ #
    still_nan = np.isnan(floor_height_map)
    if np.any(still_nan):
        known = ~still_nan
        if np.any(known):
            _, nearest_idx = distance_transform_edt(still_nan, return_indices=True)
            floor_height_map[still_nan] = floor_height_map[
                nearest_idx[0][still_nan], nearest_idx[1][still_nan]
            ]
        else:
            floor_height_map[:] = global_floor_z

    return floor_height_map.astype(np.float32), global_floor_z


def point_cloud_to_occupancy_and_surface_height(
    points,
    floor_height_map,
    x_bins,
    y_bins,
    floor_height_threshold=0.1,
    ceiling_height_threshold=2.5,
):
    """
    基于逐格局部地面高度，生成 occupancy map 和 surface height map。

    occupancy_map:
        1  : 该格有点云且在局部地面以上 floor_height_threshold 处存在物体
        0  : 该格有点云但无物体高于地面阈值（空闲）
       -1  : 该格无任何点云（未知）

    surface_height_map:
        occupancy==1 的格子：该格内物体点的最大绝对 z 高度（米）
        其余格子：0.0

    参数:
        points                  : (N, 3) 点云坐标
        floor_height_map        : (H, W) float32，每格局部地面 z
        x_bins                  : (W+1,) x 轴网格边界
        y_bins                  : (H+1,) y 轴网格边界
        floor_height_threshold  : 高于局部地面多少以上才算占用（米）
        ceiling_height_threshold: 相对局部地面的天花板截断高度（米），超过则忽略

    返回:
        occupancy_map      : (H, W) int8
        surface_height_map : (H, W) float32
    """
    H = len(y_bins) - 1
    W = len(x_bins) - 1

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    xi = np.digitize(x, x_bins) - 1
    yi = np.digitize(y, y_bins) - 1
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi_b = xi[in_bounds]
    yi_b = yi[in_bounds]
    z_b  = z[in_bounds]

    # 每个点的局部地面高度
    local_floor = floor_height_map[yi_b, xi_b]

    # 相对局部地面的高度
    above_ground = z_b - local_floor

    occupancy_map = np.full((H, W), -1, dtype=np.int8)
    surface_height_map = floor_height_map  # np.zeros((H, W), dtype=np.float32)
    # 标记有点的格子
    has_point = np.zeros((H, W), dtype=bool)
    has_point[yi_b, xi_b] = True
    occupancy_map[has_point] = 0

    # 过滤天花板以上的点（按局部地面计算）
    below_ceiling = above_ground <= ceiling_height_threshold
    xi_b = xi_b[below_ceiling]
    yi_b = yi_b[below_ceiling]
    z_b  = z_b[below_ceiling]
    above_ground = above_ground[below_ceiling]

    # 筛选高于地面阈值的点（物体点）
    above_floor_mask = above_ground >= floor_height_threshold
    if np.any(above_floor_mask):
        xia = xi_b[above_floor_mask]
        yia = yi_b[above_floor_mask]
        z_a = z_b[above_floor_mask]

        # 每格最大绝对 z 高度（scatter max）
        max_z = np.full((H, W), -np.inf, dtype=np.float32)
        np.maximum.at(max_z, (yia, xia), z_a)

        has_obj = max_z > -np.inf
        occupancy_map[has_obj] = 1
        surface_height_map[has_obj] = max_z[has_obj]

    return occupancy_map, surface_height_map


def process_scene(
    scene_name,
    scene_data_dir,
    occupancy_save_root,
    grid_size=0.1,
    floor_height_threshold=0.1,
    ceiling_height_threshold=2.5,
    floor_label_id=2,
    floor_z_range=(0.0, 0.2),
    semantic_npy_name="segment.npy",
):
    """
    处理单个 ScanNet++ 场景（chunk），生成并保存 occupancy map 和 surface height map。

    ScanNet++ 与 ScanNet 的主要差异：
      - 无轴对齐矩阵，坐标直接使用
      - segment.npy / instance.npy 形状为 (N, 1)，需取 [:, 0]
      - 场景名含 chunk 后缀，如 'c29b5e479c_16'

    保存格式（pkl）：
        {
            'occupancy_map':      int8  (H, W)    # -1=未知 / 0=空闲 / 1=占用
            'surface_height_map': float32 (H, W)  # 物体点的最大绝对 z 高度（米）
            'floor_height_map':   float32 (H, W)  # 局部地面绝对 z（米）
            'x_bins':             float64 (W,)
            'y_bins':             float64 (H,)
            'global_floor_z':     float
        }
    """
    save_path = os.path.join(occupancy_save_root, scene_name + ".pkl")
    if os.path.exists(save_path):
        print(f"  [skip] {scene_name} already processed.")
        # return

    # --- 加载数据 ---
    coord_path = os.path.join(scene_data_dir, scene_name, "coord.npy")
    scene_coord = np.load(coord_path)
    color_path = os.path.join(scene_data_dir, scene_name, "color.npy")
    scene_color = np.load(color_path)

    seg_path = os.path.join(scene_data_dir, scene_name, semantic_npy_name)
    if os.path.exists(seg_path):
        raw = np.load(seg_path)
        # segment.npy 在 ScanNet++ 中形状为 (N, 1)
        semantic_label = raw[:, 0] if raw.ndim == 2 else raw
    else:
        semantic_label = None

    # ScanNet++ 无轴对齐矩阵，坐标直接使用

    # --- 构建网格边界 ---
    x, y = scene_coord[:, 0], scene_coord[:, 1]
    x_bins = np.arange(x.min(), x.max() + grid_size, grid_size)
    y_bins = np.arange(y.min(), y.max() + grid_size, grid_size)

    # --- 逐格地面高度 ---
    floor_height_map, global_floor_z = compute_per_cell_floor_height_map(
        scene_coord,
        semantic_label=semantic_label,
        x_bins=x_bins,
        y_bins=y_bins,
        floor_label_id=floor_label_id,
        floor_z_range=floor_z_range,
    )

    # --- occupancy 及 surface height ---
    occupancy_map, surface_height_map = point_cloud_to_occupancy_and_surface_height(
        scene_coord,
        floor_height_map=floor_height_map,
        x_bins=x_bins,
        y_bins=y_bins,
        floor_height_threshold=floor_height_threshold,
        ceiling_height_threshold=ceiling_height_threshold,
    )

    # visualize_occupancy_map(occupancy_map)
    # visualize_pointcloud(scene_coord, scene_color)

    # --- 保存 ---
    result = dict(
        occupancy_map=occupancy_map,
        surface_height_map=surface_height_map,
        floor_height_map=floor_height_map,
        x_bins=x_bins[:-1],
        y_bins=y_bins[:-1],
        global_floor_z=global_floor_z,
    )
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    occ_ratio = (occupancy_map == 1).sum() / max((occupancy_map != -1).sum(), 1)
    print(
        f"  [done] {scene_name} | grid {occupancy_map.shape} | "
        f"occ_ratio={occ_ratio:.2%} | global_floor_z={global_floor_z:.3f}"
    )


def main():
    # ===== 路径配置 =====
    preprocess_data_root = "/data1/program/SGIFormer/data/scannetpp/"
    scene_data_dir = os.path.join(preprocess_data_root, "train")  # train_grid1mm_chunk6x6_stride3x3
    occupancy_save_root = os.path.join(preprocess_data_root, "instance_augment", "scannetpp_occupancy_surface_train")
    os.makedirs(occupancy_save_root, exist_ok=True)

    # ===== 超参数 =====
    GRID_SIZE = 0.1                  # m，网格分辨率
    FLOOR_HEIGHT_THRESHOLD = 0.1     # m，高于局部地面多少以上算"被占用"
    CEILING_HEIGHT_THRESHOLD = 2.0   # m，相对局部地面的天花板截断高度
    FLOOR_LABEL_ID = 2               # ScanNet++ segment.npy 中地面的类别 ID
    FLOOR_Z_RANGE = (0.0, 0.2)       # 无地面标签时的地面 z 范围（绝对坐标）
    SEMANTIC_NPY = "segment.npy"     # 语义标签文件名

    scene_names = sorted(os.listdir(scene_data_dir))
    random.shuffle(scene_names)

    print(f"Processing {len(scene_names)} scenes ...")
    for scene_name in scene_names:
        print(scene_name)
        try:
            process_scene(
                scene_name=scene_name,
                scene_data_dir=scene_data_dir,
                occupancy_save_root=occupancy_save_root,
                grid_size=GRID_SIZE,
                floor_height_threshold=FLOOR_HEIGHT_THRESHOLD,
                ceiling_height_threshold=CEILING_HEIGHT_THRESHOLD,
                floor_label_id=FLOOR_LABEL_ID,
                floor_z_range=FLOOR_Z_RANGE,
                semantic_npy_name=SEMANTIC_NPY,
            )
        except Exception as e:
            print(f"  [error] {scene_name}: {e}")

    print("All done.")


def merge_total_scene_occupancy():
    scene_occupancy_save_dir = '/data2/program/SGIFormer/data/scannetpp/instance_augment/scannetpp_occupancy_surface_train'
    scene_occupancy_files = os.listdir(scene_occupancy_save_dir)

    total_occupancy = dict()
    for scene_occupancy_file in scene_occupancy_files:
        scene_name = scene_occupancy_file[:-4]
        with open(os.path.join(scene_occupancy_save_dir, scene_occupancy_file), 'rb') as f:
            scene_occupancy = pickle.load(f)
        total_occupancy[scene_name] = scene_occupancy

    # add instance classname
    instance_class_ids = [CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP]
    preprocess_data_root = '/data1/program/SGIFormer/data/scannetpp/'
    scannetpp_train_whole_scene_dir = os.path.join(preprocess_data_root, 'train')  # train_grid1mm_chunk6x6_stride3x3
    whole_scene_names = os.listdir(scannetpp_train_whole_scene_dir)
    for scene_name in whole_scene_names:
        print(scene_name)
        total_occupancy[scene_name]['per_instance_classname'] = []
        semantic_label_path = os.path.join(scannetpp_train_whole_scene_dir, scene_name, 'segment.npy')
        semantic_label = np.load(semantic_label_path)[:, 0]
        instance_label_path = os.path.join(scannetpp_train_whole_scene_dir, scene_name, 'instance.npy')
        instance_label = np.load(instance_label_path)[:, 0]

        new_semantic_label = np.ones_like(semantic_label) * -1
        for i, ins_cls_id in enumerate(instance_class_ids):
            new_semantic_label[semantic_label == ins_cls_id] = i
        semantic_label = new_semantic_label

        mask = semantic_label == -1
        # mapping ignored instance to ignore index
        instance_label[mask] = -1
        instance_ids = np.unique(instance_label)
        for instance_id in instance_ids:
            if instance_id == -1:
                continue
            assert len(np.unique(semantic_label[instance_label == instance_id])) == 1
            semantic_classid = np.unique(semantic_label[instance_label == instance_id])[0]
            semantic_classname = INST_LABELS_PP[semantic_classid]
            total_occupancy[scene_name]['per_instance_classname'].append(semantic_classname)

    with open('/data1/program/SGIFormer/data/scannetpp/instance_augment/scene_occupancy_surface_info_full_train.pkl', 'wb') as f:  # 'wb' 表示二进制写入
        pickle.dump(total_occupancy, f)



if __name__ == "__main__":
    # main()
    merge_total_scene_occupancy()
