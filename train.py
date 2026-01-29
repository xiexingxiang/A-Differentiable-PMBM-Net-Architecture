# train.py (最终版 - 包含训练/验证循环和多指标可视化)
import sys
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import imageio.v2 as imageio
import matplotlib.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

# 引入MOTA计算所需库
import motmetrics as mm
import pandas as pd

# 将当前文件的父目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.nuscenes_dataset import NuScenesDataset
from models.d_pmbm_net import D_PMBM_Net
from losses.gospa_loss import GOSPALoss

# ======================================================================
# 修改后的函数：绘制包含 GOSPA 和 MOTA 的多面板训练曲线
# ======================================================================
def plot_training_results(epochs_list, history, save_path="training_summary.png"):
    #绘制并保存训练过程中的所有关键指标曲线
    fig, axs = plt.subplots(2, 1, figsize=(12, 10)) # 创建2行1列的子图

    # --- 子图1: 损失和学习率 ---
    color = 'tab:red'
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Average Loss', color=color)
    axs[0].plot(epochs_list, history['train_loss'], color=color, marker='o', label='Train Loss')
    axs[0].plot(epochs_list, history['val_loss'], color=color, marker='s', linestyle='--', label='Validation Loss')
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].legend(loc='upper left')
    
    ax2 = axs[0].twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(epochs_list, history['lr'], color=color, marker='x', linestyle=':', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    axs[0].set_title('Training & Validation Loss and Learning Rate')

    # --- 子图2: GOSPA 和 MOTA ---
    color = 'tab:green'
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('GOSPA Score (Lower is Better)', color=color)
    axs[1].plot(epochs_list, history['gospa'], color=color, marker='o', label='GOSPA')
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].legend(loc='upper left')
    
    ax3 = axs[1].twinx()
    color = 'tab:purple'
    ax3.set_ylabel('MOTA (Higher is Better)', color=color)
    ax3.plot(epochs_list, history['mota'], color=color, marker='s', linestyle='--', label='MOTA')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='upper right')
    axs[1].set_title('Validation Metrics (GOSPA & MOTA)')
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training summary plot saved to {save_path}")
    
    
    # ======================================================================
# 2. 新增与修改的评估函数
# ======================================================================
def calculate_gospa_for_eval(pred_set, gt_set, c=10.0, p=1):
    #计算单帧的GOSPA度量，用于评估
    if pred_set.numel() == 0 and gt_set.numel() == 0:
        return 0.0
    if pred_set.numel() == 0 or gt_set.numel() == 0:
        # 如果一个为空，另一个不为空，惩罚是所有物体的代价之和
        num_missed_or_false = len(gt_set) if pred_set.numel() == 0 else len(pred_set)
        return ( (c**p) * num_missed_or_false )**(1/p)

    pred_pos = pred_set[:, :2].cpu().numpy()
    gt_pos = gt_set[:, :2].cpu().numpy()
    
    cost_matrix = cdist(pred_pos, gt_pos)
    cost_matrix = np.minimum(cost_matrix, c)**p
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    loc_cost = cost_matrix[row_ind, col_ind].sum()
    card_error = (c**p) * (len(pred_set) + len(gt_set) - 2 * len(row_ind))
    
    gospa_val = (loc_cost + card_error)**(1/p)
    return gospa_val

def format_for_motmetrics(frame_idx, states, is_gt):
    #将单帧的状态(预测或真值)转换为 motmetrics 需要的 DataFrame 格式。"""
    data_list = []
    # 假设 states 张量的格式为 [N, D]，其中 D >= 4，前四维是 [x, y, vx, vy]
    # 我们还需要一个跟踪ID。这里我们假设第5维是 track_id。
    # **重要**: 您需要根据您数据集的实际情况来获取 track_id。
    # NuScenesDataset的collate_fn需要返回track_id。
    for i in range(states.shape[0]):
        obj = states[i].cpu().numpy()
        row = {
            'FrameId': frame_idx,
            'Id': int(obj[4]) if len(obj) > 4 else i, # 假设第5个元素是track_id，否则用索引
            'X': obj[0] - 0.5, # 假设是中心点，转换为左上角
            'Y': obj[1] - 0.5,
            'Width': 1.0,      # 假设固定大小，您需要按需修改
            'Height': 1.0,
        }
        if not is_gt:
            row['Confidence'] = 1.0 # 模型没有置信度输出，默认为1.0
        data_list.append(row)
    return pd.DataFrame(data_list)


# ======================================================================
# 3. 新的核心验证函数 (计算所有指标) - 正确版本
# ======================================================================
def validate_and_calculate_metrics(model, val_loader, loss_fn, device):
    #在一个epoch上运行验证，并计算Loss, GOSPA和MOTA。
    model.eval()
    total_val_loss = 0
    epoch_gospa_scores = []
    
    # 初始化MOTA计算所需
    acc = mm.MOTAccumulator(auto_id=True)
    
    # 首先，在一个列表中收集所有帧的数据
    all_gt_dfs = []
    all_preds_dfs = []
    frame_counter = 0

    print("Running validation and collecting data for MOTA...")
    with torch.no_grad():
        # 这个循环只负责收集数据
        for batch in val_loader: # 这里不需要tqdm，因为它会和最终的tqdm混淆
            all_pcs_seq, all_gts_seq = batch
            for scene_idx in range(len(all_pcs_seq)):
                point_clouds_seq = all_pcs_seq[scene_idx]
                gt_seq = all_gts_seq[scene_idx]
                prev_tracks = None
                val_sequence_loss = torch.tensor(0.0, device=device)
                
                for t in range(len(point_clouds_seq)):
                    point_clouds = point_clouds_seq[t].to(device)
                    gt_states = gt_seq[t].to(device)
                    
                    propagated_tracks, new_detection_features = model(point_clouds, prev_tracks)
                    predicted_states = model.extract_states(new_detection_features)
                    
                    # 1. 计算损失
                    loss = loss_fn(predicted_states, gt_states)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_sequence_loss += loss

                    # 2. 计算GOSPA
                    gospa_score = calculate_gospa_for_eval(predicted_states, gt_states)
                    epoch_gospa_scores.append(gospa_score)

                    # 3. 为MOTA收集数据
                    gt_df = format_for_motmetrics(frame_counter, gt_states, is_gt=True)
                    preds_df = format_for_motmetrics(frame_counter, predicted_states, is_gt=False)
                    all_gt_dfs.append(gt_df)
                    all_preds_dfs.append(preds_df)
                    
                    frame_counter += 1
                    prev_tracks = propagated_tracks
                
                total_val_loss += val_sequence_loss.item()

    # 数据收集完毕后，进行MOTA的计算
    print("All validation data collected. Calculating MOTA...")
    
    # 1. 将所有数据合并成一个大 DataFrame
    if not all_gt_dfs: # 如果验证集为空，处理边界情况
        return {'val_loss': 0, 'gospa': -1, 'mota': 0}
        
    final_gt_df = pd.concat(all_gt_dfs)
    final_preds_df = pd.concat(all_preds_dfs)

    # 2. 逐帧更新累加器 (使用唯一的FrameId)
    for frame_id in final_gt_df['FrameId'].unique():
        gt_frame = final_gt_df[final_gt_df['FrameId'] == frame_id]
        preds_frame = final_preds_df[final_preds_df['FrameId'] == frame_id]
        
        distance_matrix = mm.distances.iou_matrix(gt_frame[['X', 'Y', 'Width', 'Height']], preds_frame[['X', 'Y', 'Width', 'Height']], max_iou=0.5)
        acc.update(
            gt_frame['Id'].values,
            preds_frame['Id'].values,
            distance_matrix
        )

    # 3. 计算平均指标
    avg_val_loss = total_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
    avg_gospa = np.mean(epoch_gospa_scores) if epoch_gospa_scores else -1.0
    
    # 4. 计算最终MOTA
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota'], name='acc_overview')
    mota_score = summary['mota']['acc_overview'] if not summary.empty else 0.0

    print("MOTA calculation finished.")
    return {
        'val_loss': avg_val_loss,
        'gospa': avg_gospa,
        'mota': mota_score
    }
 


# ======================================================================
# 4. 可视化函数 (保持不变)
# ======================================================================
def visualize_tracking_results(model, dataset, device, save_dir="tracking_visualization"):
    #此函数与您之前的版本相同，无需修改
    pass # 占位，保持原有代码

# ======================================================================
# 5. 主函数 (已更新)
# ======================================================================
def main():
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading ---

    train_dataset = NuScenesDataset(
        version='v1.0-mini',
        dataroot="/home/huangjinrong/d_pmbm_net/data/sets",
        split='train',
        seq_len=10
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=NuScenesDataset.collate_fn)
    val_dataset = NuScenesDataset(
        version='v1.0-mini',
        dataroot="/home/huangjinrong/d_pmbm_net/data/sets",
        split='val',
        seq_len=10
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=NuScenesDataset.collate_fn)
    
    # --- 3. Model, Loss, Optimizer ---
    model = D_PMBM_Net(existence_threshold=0.1).to(device)
    loss_fn = GOSPALoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    # 初始化历史记录字典，新增'mota'
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'gospa': [], 'mota': []}
    best_val_loss = float('inf')

    # --- 4. Training Loop ---
    epochs = 100
    for epoch in range(epochs):
        # ======================= TRAINING STEP =======================
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            total_batch_loss = torch.tensor(0.0, device=device)
            all_pcs_seq, all_gts_seq = batch

            for scene_idx in range(len(all_pcs_seq)):
                point_clouds_seq = all_pcs_seq[scene_idx]
                gt_seq = all_gts_seq[scene_idx]
                prev_tracks = None
                sequence_loss = torch.tensor(0.0, device=device)
                
                for t in range(len(point_clouds_seq)):
                    point_clouds, gt_states = point_clouds_seq[t].to(device), gt_seq[t].to(device)
                    propagated_tracks, new_detection_features = model(point_clouds, prev_tracks)
                    predicted_states = model.extract_states(new_detection_features)
                    
                    loss = loss_fn(predicted_states, gt_states)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        sequence_loss += loss
                    prev_tracks = propagated_tracks

                total_batch_loss += sequence_loss
            
            if total_batch_loss > 0 and total_batch_loss.requires_grad:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_train_loss += total_batch_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ======================= VALIDATION STEP =======================
        val_metrics = validate_and_calculate_metrics(model, val_loader, loss_fn, device)
        avg_val_loss = val_metrics['val_loss']
        avg_gospa = val_metrics['gospa']
        avg_mota = val_metrics['mota']
        
        # ======================= LOGGING & SAVING =======================
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['gospa'].append(avg_gospa)
        history['mota'].append(avg_mota) # 存储MOTA

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | GOSPA: {avg_gospa:.4f} | MOTA: {avg_mota:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "d_pmbm_net_best.pth")

        scheduler.step()

    # --- 5. Final Actions ---
    final_model_path = "d_pmbm_net_final_epoch.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining finished. Final model saved to {final_model_path}")

    plot_training_results(range(1, epochs + 1), history)
    
    print("Loading best model for visualization...")
    model.load_state_dict(torch.load("d_pmbm_net_best.pth"))
    visualize_tracking_results(model, val_dataset, device)
    
if __name__ == '__main__':
    main()

