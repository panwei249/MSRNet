# -*- coding: utf-8 -*-
"""EEG preprocessing + dataset loading utilities.

This file is split out from the original `pasted.txt`.
Dependency:
    - `ms_1d_conv.py` (EnhancedTemporalLearner)
"""

import os
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

from ms_1d_conv import EnhancedTemporalLearner

def segment_eeg_sliding(eeg_array, window_size=8, step=4):
    """
    对输入 EEG 数据（形状: (62, T, 5)）使用滑动窗口切分：
      每个窗口的形状为 (62, window_size, 5)
      返回窗口列表（通过减小步长增大窗口重叠率，保证不丢失数据）
    """
    channels, T, f = eeg_array.shape
    segments = []
    for start in range(0, T - window_size + 1, step):
        seg = eeg_array[:, start:start+window_size, :]
        segments.append(seg)
    return segments
# -------------------------------------------
# 数据加载函数：load_data_inde
# -------------------------------------------

def load_data_inde1(path, subject):
    """
    Independent实验数据加载：
      - 指定 subject 对应的 .mat 文件作为测试集，
      - 目录下其他所有 .mat 文件作为训练集。
    假设每个 .mat 文件包含 24 个 trial，每个 trial 的数据形状为 (62, T, 5)。
    对每个 trial，先采用滑动窗口切分（窗口大小为16，步长为4，即窗口重叠较大），
    然后对每个窗口使用 TemporalLearner 提取时序特征，输出为 (62, 5) 的表示（每个频段提取1个特征）。
    标签按 label_seed4[1] 提取（4分类：0,1,2,3）。
    """
    # 定义 TemporalLearner 和融合层，用于将多尺度卷积输出降维为 1 维
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50 * 3, 1)  # 输出维度为 1
    # 固定为 eval 模式，不需要梯度
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed4 = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        ]
        data = loadmat(mat_path)
        samples_list = []
        labels_list = []
        # 对每个 trial (1~24)
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        for i in range(1, 25):
            prefix = keys[0].split("_")[0]  # 如 "ha
            key = f"{prefix}_eeg{i}"

            if key in data:
                trial_data = data[key].astype(np.float32)
                print(key)# 原始形状: (62, T, 5)
                print(data[key].shape)
                # 替换原来的 segment_eeg_sliding 调用逻辑
                actual_window_size = min(trial_data.shape[1], 16)
                windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
                # 对每个窗口提取时序特征
                for window in windows:
                    # window: (62, window_size, 5)
                    with torch.no_grad():
                        window_tensor = torch.from_numpy(window)  # (62, window_size, 5)
                        channel_features = []  # 每个频段提取特征，目标输出 (62, 1)
                        for f in range(5):
                            # 取出第 f 个频段：形状 (62, window_size)
                            freq_data = window_tensor[:, :, f]
                            # 调整为 (62, window_size, 1)
                            freq_data = freq_data.unsqueeze(-1)
                            # TemporalLearner 提取特征，输出 (62, 50*3)
                            temp_feat = temporal_learner(freq_data)
                            # 通过融合层将特征降为 (62, 1)
                            feat_reduced = fusion_fc(temp_feat)
                            channel_features.append(feat_reduced)
                        # 拼接得到窗口样本，形状 (62, 5)
                        window_feature = torch.cat(channel_features, dim=-1)
                        window_feature = window_feature.cpu().numpy()
                    samples_list.append(window_feature)
                    # 对于每个窗口，标签取 trial 对应的标签： label_seed4[1][i-1]
                    labels_list.append(label_seed4[1][i - 1])
        if not samples_list:
            raise ValueError(f"No valid trials found in {mat_path}")
        # 合并所有窗口样本，得到 (n_windows, 62, 5)
        samples_arr = np.stack(samples_list)
        labels_arr = np.array(labels_list, dtype=np.int32)
        return samples_arr, labels_arr

    train_samples, train_labels = [], []
    test_samples, test_labels = [], []
    for filename in os.listdir(path):
        if not filename.endswith('.mat'):
            continue
        mat_path = os.path.join(path, filename)
        if filename == subject:
            x_ts, y_ts = parse_mat_file(mat_path)
            # 使用 standardize 标准化，转换为 Tensor，再转为 numpy
            x_ts = standardize(x_ts).numpy()
            test_samples.append(x_ts)
            test_labels.append(y_ts)
        else:
            print(mat_path)
            x_tr, y_tr = parse_mat_file(mat_path)
            x_tr = standardize(x_tr).numpy()
            train_samples.append(x_tr)
            train_labels.append(y_tr)
    if not test_samples:
        raise ValueError(f"指定的测试被试文件 {subject} 不存在或不是 .mat 文件")
    if train_samples:
        x_tr_merged = np.concatenate(train_samples, axis=0)
        y_tr_merged = np.concatenate(train_labels, axis=0)
    else:
        raise ValueError("没有找到除测试被试外的任何 .mat 文件用于训练，请检查目录。")
    x_ts_merged = np.concatenate(test_samples, axis=0)
    y_ts_merged = np.concatenate(test_labels, axis=0)
    print(f"独立实验: 训练样本总数 {x_tr_merged.shape[0]}, 测试样本总数 {x_ts_merged.shape[0]}")
    print(f"训练集形状: {x_tr_merged.shape}, 测试集形状: {x_ts_merged.shape}")
    return {
        "x_tr": x_tr_merged,
        "y_tr": y_tr_merged,
        "x_ts": x_ts_merged,
        "y_ts": y_ts_merged
    }

def load_data_inde2(path, subject):
    """
    Independent实验数据加载：
      - 指定 subject 对应的 .mat 文件作为测试集，
      - 目录下其他所有 .mat 文件作为训练集。
    假设每个 .mat 文件包含 24 个 trial，每个 trial 的数据形状为 (62, T, 5)。
    对每个 trial，先采用滑动窗口切分（窗口大小为16，步长为4，即窗口重叠较大），
    然后对每个窗口使用 TemporalLearner 提取时序特征，输出为 (62, 5) 的表示（每个频段提取1个特征）。
    标签按 label_seed4[1] 提取（4分类：0,1,2,3）。
    """
    # 定义 TemporalLearner 和融合层，用于将多尺度卷积输出降维为 1 维
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50 * 3, 1)  # 输出维度为 1
    # 固定为 eval 模式，不需要梯度
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed = [
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ]
        data = loadmat(mat_path)
        samples_list = []
        labels_list = []
        # 对每个 trial (1~24)
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        for i in range(1, 16):
            prefix = keys[0].split("_")[0]  # 如 "ha
            key = f"{prefix}_eeg{i}"

            if key in data:
                trial_data = data[key].astype(np.float32)
                print(key)# 原始形状: (62, T, 5)
                print(data[key].shape)
                # 使用滑动窗口切分，步长设置为 4 增大重叠
                actual_window_size = min(trial_data.shape[1], 16)
                windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
                # 对每个窗口提取时序特征
                for window in windows:
                    # window: (62, window_size, 5)
                    with torch.no_grad():
                        window_tensor = torch.from_numpy(window)  # (62, window_size, 5)
                        channel_features = []  # 每个频段提取特征，目标输出 (62, 1)
                        for f in range(5):
                            # 取出第 f 个频段：形状 (62, window_size)
                            freq_data = window_tensor[:, :, f]
                            # 调整为 (62, window_size, 1)
                            freq_data = freq_data.unsqueeze(-1)
                            # TemporalLearner 提取特征，输出 (62, 50*3)
                            temp_feat = temporal_learner(freq_data)
                            # 通过融合层将特征降为 (62, 1)
                            feat_reduced = fusion_fc(temp_feat)
                            channel_features.append(feat_reduced)
                        # 拼接得到窗口样本，形状 (62, 5)
                        window_feature = torch.cat(channel_features, dim=-1)
                        window_feature = window_feature.cpu().numpy()
                    samples_list.append(window_feature)
                    # 对于每个窗口，标签取 trial 对应的标签： label_seed4[1][i-1]
                    labels_list.append(label_seed[1][i - 1])
        if not samples_list:
            raise ValueError(f"No valid trials found in {mat_path}")
        # 合并所有窗口样本，得到 (n_windows, 62, 5)
        samples_arr = np.stack(samples_list)
        labels_arr = np.array(labels_list, dtype=np.int32)
        return samples_arr, labels_arr

    train_samples, train_labels = [], []
    test_samples, test_labels = [], []
    for filename in os.listdir(path):
        if not filename.endswith('.mat'):
            continue
        mat_path = os.path.join(path, filename)
        if filename == subject:
            x_ts, y_ts = parse_mat_file(mat_path)
            # 使用 standardize 标准化，转换为 Tensor，再转为 numpy
            x_ts = standardize(x_ts).numpy()
            test_samples.append(x_ts)
            test_labels.append(y_ts)
        else:
            print(mat_path)
            x_tr, y_tr = parse_mat_file(mat_path)
            x_tr = standardize(x_tr).numpy()
            train_samples.append(x_tr)
            train_labels.append(y_tr)
    if not test_samples:
        raise ValueError(f"指定的测试被试文件 {subject} 不存在或不是 .mat 文件")
    if train_samples:
        x_tr_merged = np.concatenate(train_samples, axis=0)
        y_tr_merged = np.concatenate(train_labels, axis=0)
    else:
        raise ValueError("没有找到除测试被试外的任何 .mat 文件用于训练，请检查目录。")
    x_ts_merged = np.concatenate(test_samples, axis=0)
    y_ts_merged = np.concatenate(test_labels, axis=0)
    print(f"独立实验: 训练样本总数 {x_tr_merged.shape[0]}, 测试样本总数 {x_ts_merged.shape[0]}")
    print(f"训练集形状: {x_tr_merged.shape}, 测试集形状: {x_ts_merged.shape}")
    return {
        "x_tr": x_tr_merged,
        "y_tr": y_tr_merged,
        "x_ts": x_ts_merged,
        "y_ts": y_ts_merged
    }

def load_data_inde3(path, subject):
    print(subject)
    """
    Independent实验数据加载：
      - 指定 subject 对应的 .mat 文件作为测试集，
      - 目录下其他所有 .mat 文件作为训练集。
    假设每个 .mat 文件包含 24 个 trial，每个 trial 的数据形状为 (62, T, 5)。
    对每个 trial，先采用滑动窗口切分（窗口大小为16，步长为4，即窗口重叠较大），
    然后对每个窗口使用 TemporalLearner 提取时序特征，输出为 (62, 5) 的表示（每个频段提取1个特征）。
    标签按 label_seed4[1] 提取（4分类：0,1,2,3）。
    """
    # 定义 TemporalLearner 和融合层，用于将多尺度卷积输出降维为 1 维
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50 * 3, 1)  # 输出维度为 1
    # 固定为 eval 模式，不需要梯度
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed = [
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
        ]
        data = loadmat(mat_path)
        samples_list = []
        labels_list = []
        # 对每个 trial (1~24)
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        for i in range(1, 29):
            key = f"session_{i:03d}"
            if key in data:
                trial_data = data[key].astype(np.float32)
                print(key)# 原始形状: (62, T, 5)
                print(data[key].shape)
                # 使用滑动窗口切分，步长设置为 4 增大重叠
                actual_window_size = min(trial_data.shape[1], 16)
                windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
                # 对每个窗口提取时序特征
                for window in windows:
                    # window: (62, window_size, 5)
                    with torch.no_grad():
                        window_tensor = torch.from_numpy(window)  # (62, window_size, 5)
                        channel_features = []  # 每个频段提取特征，目标输出 (62, 1)
                        for f in range(5):
                            # 取出第 f 个频段：形状 (62, window_size)
                            freq_data = window_tensor[:, :, f]
                            # 调整为 (62, window_size, 1)
                            freq_data = freq_data.unsqueeze(-1)
                            # TemporalLearner 提取特征，输出 (62, 50*3)
                            temp_feat = temporal_learner(freq_data)
                            # 通过融合层将特征降为 (62, 1)
                            feat_reduced = fusion_fc(temp_feat)
                            channel_features.append(feat_reduced)
                        # 拼接得到窗口样本，形状 (62, 5)
                        window_feature = torch.cat(channel_features, dim=-1)
                        window_feature = window_feature.cpu().numpy()
                    samples_list.append(window_feature)
                    # 对于每个窗口，标签取 trial 对应的标签： label_seed4[1][i-1]
                    labels_list.append(label_seed[1][i - 1])
        if not samples_list:
            raise ValueError(f"No valid trials found in {mat_path}")
        # 合并所有窗口样本，得到 (n_windows, 62, 5)
        samples_arr = np.stack(samples_list)
        labels_arr = np.array(labels_list, dtype=np.int32)
        return samples_arr, labels_arr

    train_samples, train_labels = [], []
    test_samples, test_labels = [], []
    for filename in os.listdir(path):
        if not filename.endswith('.mat'):
            continue
        mat_path = os.path.join(path, filename)
        if filename == subject:
            x_ts, y_ts = parse_mat_file(mat_path)
            # 使用 standardize 标准化，转换为 Tensor，再转为 numpy
            x_ts = standardize(x_ts).numpy()
            test_samples.append(x_ts)
            test_labels.append(y_ts)
        else:
            print(mat_path)
            x_tr, y_tr = parse_mat_file(mat_path)
            x_tr = standardize(x_tr).numpy()
            train_samples.append(x_tr)
            train_labels.append(y_tr)
    if not test_samples:
        raise ValueError(f"指定的测试被试文件 {subject} 不存在或不是 .mat 文件")
    if train_samples:
        x_tr_merged = np.concatenate(train_samples, axis=0)
        y_tr_merged = np.concatenate(train_labels, axis=0)
    else:
        raise ValueError("没有找到除测试被试外的任何 .mat 文件用于训练，请检查目录。")
    x_ts_merged = np.concatenate(test_samples, axis=0)
    y_ts_merged = np.concatenate(test_labels, axis=0)
    print(f"独立实验: 训练样本总数 {x_tr_merged.shape[0]}, 测试样本总数 {x_ts_merged.shape[0]}")
    print(f"训练集形状: {x_tr_merged.shape}, 测试集形状: {x_ts_merged.shape}")
    return {
        "x_tr": x_tr_merged,
        "y_tr": y_tr_merged,
        "x_ts": x_ts_merged,
        "y_ts": y_ts_merged
    }


def load_data_denpendent3(path, subject, fold_idx=1):
    """
    Dependent 跨会话6折交叉实验：
    fold_idx: 0~5
    每个标签在24个trial中各6次出现，取每组第fold_idx个为测试，其余为训练
    """
    # 特征提取
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50*3,1)
    temporal_learner.eval(); fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed4 = [
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
            [1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6],
           ]
        labels = label_seed4[2]  # 使用第一行标签
        # 按标签分组索引
        label_indices = {lbl: [] for lbl in set(labels)}
        for idx, lbl in enumerate(labels, start=1):
            label_indices[lbl].append(idx)
        # 每个标签第fold_idx次出现为测试
        test_trials = [label_indices[lbl][fold_idx] for lbl in sorted(label_indices.keys())]

        data = loadmat(mat_path)
        train_samples, train_labels = [], []
        test_samples, test_labels = [], []
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        prefix = keys[0].split("_")[0] if keys else ""

        for i in range(1,29):
            key = f"{prefix}_eeg{i}"
            key = f"session_{i:03d}"
            if key not in data: continue
            trial_data = data[key].astype(np.float32)
            actual_window_size = min(trial_data.shape[1], 16)
            windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
            for win in windows:
                with torch.no_grad():
                    wt = torch.from_numpy(win)
                    feats = []
                    for f in range(5):
                        tmp = temporal_learner(wt[:,:,f].unsqueeze(-1))
                        red = fusion_fc(tmp)
                        feats.append(red)
                    feat62x5 = torch.cat(feats, dim=-1).cpu().numpy()
                if i in test_trials:
                    test_samples.append(feat62x5)
                    test_labels.append(labels[i-1])
                else:
                    train_samples.append(feat62x5)
                    train_labels.append(labels[i-1])

        if not train_samples or not test_samples:
            raise ValueError("Empty split for subject {} fold {}".format(subject, fold_idx))
        return (
            np.stack(train_samples), np.array(train_labels, np.int32),
            np.stack(test_samples),  np.array(test_labels, np.int32)
        )

    mat_path = os.path.join(path, subject)
    x_tr, y_tr, x_ts, y_ts = parse_mat_file(mat_path)
    x_tr = standardize(x_tr); x_ts = standardize(x_ts)
    return {"x_tr": x_tr.detach().numpy(), "y_tr": y_tr,
            "x_ts": x_ts.detach().numpy(), "y_ts": y_ts}
def load_data_denpendent2(path, subject, fold_idx=1):
    """
    Dependent 跨会话6折交叉实验：
    fold_idx: 0~5
    每个标签在24个trial中各6次出现，取每组第fold_idx个为测试，其余为训练
    """
    # 特征提取
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50*3,1)
    temporal_learner.eval(); fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed4 = [
           [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
           [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
           [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
           ]
        labels = label_seed4[2]  # 使用第一行标签
        # 按标签分组索引
        label_indices = {lbl: [] for lbl in set(labels)}
        for idx, lbl in enumerate(labels, start=1):
            label_indices[lbl].append(idx)
        # 每个标签第fold_idx次出现为测试
        test_trials = [label_indices[lbl][fold_idx] for lbl in sorted(label_indices.keys())]

        data = loadmat(mat_path)
        train_samples, train_labels = [], []
        test_samples, test_labels = [], []
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        prefix = keys[0].split("_")[0] if keys else ""

        for i in range(1,25):
            key = f"{prefix}_eeg{i}"
            print(key)
            trial_data = data[key].astype(np.float32)
            actual_window_size = min(trial_data.shape[1], 16)
            windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
            for win in windows:
                with torch.no_grad():
                    wt = torch.from_numpy(win)
                    feats = []
                    for f in range(5):
                        tmp = temporal_learner(wt[:,:,f].unsqueeze(-1))
                        red = fusion_fc(tmp)
                        feats.append(red)
                    feat62x5 = torch.cat(feats, dim=-1).cpu().numpy()
                if i in test_trials:
                    test_samples.append(feat62x5)
                    test_labels.append(labels[i-1])
                else:
                    train_samples.append(feat62x5)
                    train_labels.append(labels[i-1])

        if not train_samples or not test_samples:
            raise ValueError("Empty split for subject {} fold {}".format(subject, fold_idx))
        return (
            np.stack(train_samples), np.array(train_labels, np.int32),
            np.stack(test_samples),  np.array(test_labels, np.int32)
        )

    mat_path = os.path.join(path, subject)
    x_tr, y_tr, x_ts, y_ts = parse_mat_file(mat_path)
    x_tr = standardize(x_tr); x_ts = standardize(x_ts)
    return {"x_tr": x_tr.detach().numpy(), "y_tr": y_tr,
            "x_ts": x_ts.detach().numpy(), "y_ts": y_ts}

def load_data_denpendent1(path, subject, fold_idx=2):
    """
    Dependent 跨会话5折交叉实验 for SEED：
      - 每个 trial 对应一个会话，总共 15 个会话，3 类标签
      - fold_idx: 0~4；对每个标签，取它在 15 个会话中第 fold_idx 次出现的会话做测试
      - 其余会话全部做训练
    返回:
      {
        "x_tr": (N_train, 62, 5),
        "y_tr": (N_train,),
        "x_ts": (N_test,  62, 5),
        "y_ts": (N_test,)
      }
    """
    # 1) 特征提取模块（与原版保持一致）
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50*3, 1)
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        # 2) SEED 15 会话的标签列表，0/1/2 各出现 5 次
        label_sessions = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]
        # 构建索引：label -> 会话编号列表
        label_indices = {lbl: [] for lbl in set(label_sessions)}
        for sess_idx, lbl in enumerate(label_sessions, start=1):
            label_indices[lbl].append(sess_idx)
        # 每个标签取第 fold_idx 次出现的 sess_idx 作为测试集
        test_trials = [ label_indices[lbl][fold_idx] for lbl in sorted(label_indices.keys()) ]

        data = loadmat(mat_path)
        train_samples, train_labels = [], []
        test_samples,  test_labels  = [], []
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
        prefix = keys[0].split("_")[0] if keys else ""

        # 3) 把每个 trial 拆成滑动窗口，再提时序特征
        for i in range(1, 16):
            key = f"{prefix}_eeg{i}"
            print(key)
            if key not in data: continue
            trial_data = data[key].astype(np.float32)  # (62, T, 5)
            actual_window_size = min(trial_data.shape[1], 16)
            windows = segment_eeg_sliding(trial_data, window_size=actual_window_size, step=4)
            for win in windows:
                with torch.no_grad():
                    wt = torch.from_numpy(win)  # (62, win_len, 5)
                    feats = []
                    for f in range(5):
                        tmp = temporal_learner(wt[:, :, f].unsqueeze(-1))  # (62,150)
                        red = fusion_fc(tmp)                               # (62,1)
                        feats.append(red)
                    feat62x5 = torch.cat(feats, dim=-1).cpu().numpy()     # (62,5)

                if i in test_trials:
                    test_samples.append(feat62x5)
                    test_labels.append(label_sessions[i-1])
                else:
                    train_samples.append(feat62x5)
                    train_labels.append(label_sessions[i-1])

        if not train_samples or not test_samples:
            raise ValueError(f"Empty split for {subject}, fold {fold_idx}")

        # 4) 合并、标准化、返回
        x_tr = np.stack(train_samples)  # (N_train, 62, 5)
        y_tr = np.array(train_labels, dtype=np.int64)
        x_ts = np.stack(test_samples)   # (N_test, 62, 5)
        y_ts = np.array(test_labels,  dtype=np.int64)

        # 按通道标准化
        x_tr, x_ts = standardize_data_per_channel(x_tr, x_ts)

        return x_tr, y_tr, x_ts, y_ts

    mat_path = os.path.join(path, subject)
    x_tr, y_tr, x_ts, y_ts = parse_mat_file(mat_path)

    return {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "x_ts": x_ts,
        "y_ts": y_ts
    }


def standardize_data_per_channel(train_data, test_data):
    """
    对 train_data 和 test_data 按照通道维度（axis=1）进行标准化。
    """
    # 避免出现除 0
    eps = 1e-8

    # 计算每个通道的均值和标准差
    mean_ = train_data.mean(axis=(0, 2), keepdims=True)
    std_ = train_data.std(axis=(0, 2), keepdims=True)

    # 防止 std_ 为 0，避免通道标准差为0的情况
    std_ = std_ if np.all(std_ > eps) else eps

    # 对训练数据和测试数据进行标准化
    train_data = (train_data - mean_) / std_
    test_data = (test_data - mean_) / std_

    return train_data, test_data


def normalize(features, select_dim=0):
    # 如果 features 是 numpy 数组，先转换成 Tensor
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    # 保证维度对齐（例如，unsqueeze 在 select_dim 位置）
    features_norm = (features - features_min.unsqueeze(select_dim)) / (features_max - features_min).unsqueeze(
        select_dim)
    return features_norm


def standardize(features, select_dim=0):
    """
    对特征进行标准化（零均值，单位方差）。
    如果 features 是 numpy 数组，先转换成 Tensor
    select_dim: 要标准化的维度 (0 表示按通道，1 表示按时间步，2 表示按特征等)
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    # 计算均值和标准差
    features_mean = features.mean(dim=select_dim, keepdim=True)
    features_std = features.std(dim=select_dim, keepdim=True)

    # 防止标准差为0的情况
    eps = 1e-8
    features_std = features_std if torch.all(features_std > eps) else eps

    # 标准化
    features_standardized = (features - features_mean) / features_std
    return features_standardized
