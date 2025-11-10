import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fairmotion.data import bvh
import imageio.v2 as imageio


def load_skl_ipm():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(root_dir, 'single-person', 'data', 'raw_data', 'indiv_data')
    print(f'dataPath: {dataPath}')

    sub_names = ['X05', 'X07', 'X08', 'X09']
    motion_set = {}
    ipm_set = {}
    for sub_name in sub_names:
        seqs_path = os.path.join(dataPath, sub_name)
        print(f'Loading... {seqs_path}')
        seqs_list = os.listdir(seqs_path)
        motion_list = []
        seqs_ipm_sts_list = []
        for seq in seqs_list:
            seq_bvh_path = os.path.join(seqs_path, seq)
            # 3Dスケルトンデータ
            motion = bvh.load(seq_bvh_path)
            positions = motion.positions(local=False)
            positions_scale = positions / 1000
            motion_list.append(positions_scale)

            # 倒立振り子パラメータの計算
            bvh_data_hip = positions_scale[:, 0, :]
            bvh_data_cart = (positions_scale[:, 16, :] + positions_scale[:, 20, :]) / 2
            rod_direction_norm = np.zeros((bvh_data_hip.shape[0], 1))
            rod_direction_norm[:, 0] = np.linalg.norm((bvh_data_hip - bvh_data_cart), axis=-1)
            rod_direction = (bvh_data_hip - bvh_data_cart) / rod_direction_norm
            phi = np.arcsin(-rod_direction[:, 1:2])
            theta = np.arcsin(rod_direction[:, 0:1] / np.cos(phi))

            seq_ipm_sts = np.concatenate((bvh_data_cart[:, :-1], theta, phi, rod_direction_norm,
                                            bvh_data_cart[:, -1:], bvh_data_hip[:, :-1]), axis=1)
            seqs_ipm_sts_list.append(seq_ipm_sts)

        motion_set[sub_name] = motion_list
        ipm_set[sub_name] = seqs_ipm_sts_list
    
    return motion_set, ipm_set


def visualize_motion_ipm(motion, seq_ipm_sts, sub_name, sample_idx, save_dir="visual_results"):
    """
    motion: (frames, joints, 3)
    seq_ipm_sts: (frames, 8)
    """
    os.makedirs(save_dir, exist_ok=True)
    frames = []
    parents = [
        -1, 0, 1, 2, 3, 4, 4, 6, 7, 8, 4, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20
    ]

    # tqdmで進捗バー表示
    for f_idx in tqdm(range(min(len(motion), len(seq_ipm_sts))), desc=f"Rendering {sub_name}"):
        frame = motion[f_idx]
        x_cart, y_cart, theta, phi, rod_len, z_cart, x_hip, y_hip = seq_ipm_sts[f_idx]

        cart = np.array([x_cart, y_cart, z_cart])
        hip = np.array([x_hip, y_hip, z_cart + rod_len])

        fig = plt.figure(figsize=(10, 5))
        elev = 16  # 0
        azim = -30  # -90

        # ---- 左：元のスケルトン ----
        ax1 = fig.add_subplot(121, projection='3d')
        # 質点のノード（関節点）を表示
        ax1.scatter(frame[0, 0], frame[0, 1], frame[0, 2], 
                    c='r', s=60, alpha=0.8)
        # 質点以外のノード（関節点）を表示
        ax1.scatter(frame[1:, 0], frame[1:, 1], frame[1:, 2], 
                    c='b', s=20)
        # 親子関係に基づいてエッジを描く
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx == -1:
                continue
            p1 = frame[child_idx]
            p2 = frame[parent_idx]
            ax1.plot([p1[0], p2[0]], 
                     [p1[1], p2[1]], 
                     [p1[2], p2[2]], 
                     'k-', linewidth=2)
        # カートを薄く表示
        cart_location = (frame[16, :] + frame[20, :]) / 2
        ax1.scatter(cart_location[0], cart_location[1], cart_location[2], 
                    c='b', s=100, alpha=0.5, marker='s')
        ax1.plot([frame[16, 0], frame[20, 0]], 
                 [frame[16, 1], frame[20, 1]],
                 [frame[16, 2], frame[20, 2]],
                 'b--', linewidth=1.5, alpha=0.5)
        # ロットを薄く表示
        ax1.plot([frame[0, 0], cart_location[0]], 
                 [frame[0, 1], cart_location[1]],
                 [frame[0, 2], cart_location[2]],
                 'k-', linewidth=1.5, alpha=0.5)
        # 軸設定
        ax1.set_xlim(-0.3, 1.8)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(0, 1.8)
        ax1.view_init(elev=elev, azim=azim)
        ax1.axis('off')

        # ---- 右：倒立振り子モデル ----
        ax2 = fig.add_subplot(122, projection='3d')
        # 質点を表示
        ax2.scatter(hip[0], hip[1], hip[2], 
                    c='r', s=60, label='Point mass')
        # ロットを表示
        ax2.plot([cart[0], hip[0]], 
                 [cart[1], hip[1]], 
                 [cart[2], hip[2]], 
                 'k-', linewidth=1.5, label='Rod')
        # カートを表示
        ax2.scatter(cart[0], cart[1], cart[2], 
                    c='b', s=100, marker='s', label='Cart')
        # 軸設定
        ax2.set_xlim(-0.3, 1.8)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(0, 1.8)
        ax2.legend(fontsize=20)
        ax2.view_init(elev=elev, azim=azim)
        ax2.axis('off')

        tmp_path = f'frame_{f_idx:04d}.png'
        plt.tight_layout()
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames.append(imageio.imread(tmp_path))
        os.remove(tmp_path)

    save_path = os.path.join(save_dir, f'{sub_name}-sample{sample_idx}.gif')
    imageio.mimsave(save_path, frames, fps=50)  # 実際は60Hz
    print(f"✅ Saved {save_path}")


if __name__ == '__main__':
    # 3Dスケルトンデータと倒立振り子モデル（IPM）を準備
    motion_set, ipm_set = load_skl_ipm()

    #=== 1サンプルずつ可視化したい場合 ===
    sub_name = 'X08'
    sample_idx = 11
    motion = motion_set[sub_name][sample_idx]
    seq_ipm_sts = ipm_set[sub_name][sample_idx]
    visualize_motion_ipm(motion, seq_ipm_sts, sub_name, sample_idx)

    #=== 全サンプルをまとめて可視化したい場合 ===
    # sub_names = ['X05', 'X07', 'X08', 'X09']
    # for sub_name in sub_names:
    #     for sample_idx, (motion, ipm) in enumerate(zip(motion_set[sub_name], ipm_set[sub_name])):
    #         visualize_motion_ipm(motion, ipm, sub_name, sample_idx)
    