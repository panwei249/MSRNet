import os
import numpy as np
import scipy.io as sio
import numpy as np
import os
from scipy.io import loadmat
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def load_data_de(path, subject):
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    """加载SEED-IV数据集的依赖实验数据

    Args:
        path: 数据文件路径
        subject: 被试文件名（例如：2_20150920.mat）

    Returns:
        包含训练和测试数据的字典
    """
    mat_path = os.path.join(path, subject)
    data = loadmat(mat_path)

    samples = []
    labels = []

    # 调试信息
    print(f"Loading data from: {mat_path}")

    # SEED-IV数据集中每个trial的差分熵特征
    for i in range(1, 25):  # 24个trial
        key = f'de_LDS{i}'
        if key in data:
            print("111111111111111")
            print(key)
            # 获取当前trial的数据
            trial_data = data[key]  # shape: (62, time_steps, 5)

            # # 对时间维度进行平均
            # if len(trial_data.shape) == 3:
            #     trial_data = np.mean(trial_data, axis=1)  # shape: (62, 5)
            #
            # # 确保数据类型和维度正确
            # trial_data = trial_data.astype(np.float32)
            #
            # if trial_data.shape != (62, 5):
            #     print(f"Warning: Invalid shape for {key} after processing: {trial_data.shape}")
            #     continue
            #
            # samples.append(trial_data)
            #
            # # SEED-IV的情感标签: 0-平静, 1-悲伤, 2-恐惧, 3-高兴
            # labels=label_seed4[0]
            trial_data = trial_data.astype(np.float32)

            # 遍历时间维度的每个切片
            if len(trial_data.shape) == 3:
                time_steps = trial_data.shape[1]
                for t in range(time_steps):
                    time_slice = trial_data[:, t, :]  # 单个时间片段, shape: (62, 5)
                    samples.append(time_slice)
                    labels.append(label_seed4[0][i - 1])  # 对应trial的标签

    if not samples:
        raise ValueError(f"No valid samples found in {subject}")

    # 转换为numpy数组前检查维度
    print(f"Number of samples collected: {len(samples)}")
    print(f"Shape of first sample: {samples[0].shape}")

    # 转换为numpy数组
    try:
        samples = np.stack(samples)  # shape: (n_trials, 62, 5)
        labels = np.array(labels)

        # 添加时间维度
        # samples = np.expand_dims(samples, axis=2)  # shape: (n_trials, 62, 1, 5)

        print(f"Final samples shape: {samples.shape}")
        print(f"Final labels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels)}")

        # 可选：对每个trial的特征进行标准化

    except ValueError as e:
        print("Error stacking samples. Sample shapes:")
        for i, s in enumerate(samples):
            print(f"Sample {i} shape: {s.shape}")
        raise

    # 随机打乱数据
    indices = np.random.permutation(len(samples))
    samples = samples[indices]
    labels = labels[indices]

    # 划分训练集和测试集（80%训练，20%测试）
    n_train = int(0.6 * len(samples))

    return {
        "x_tr": samples[:n_train],  # shape: (19, 62, 5)
        "y_tr": labels[:n_train],  # shape: (19,)
        "x_ts": samples[n_train:],  # shape: (5, 62, 5)
        "y_ts": labels[n_train:]  # shape: (5,)
    }


def load_data_de1(path, subject):
    """
    将每个 trial 的时间维度拆分成多个样本。
    同时将原始标签 -1, 0, 1 映射到 0, 1, 2。
    """

    # 1. 读取主体数据
    mat_path = os.path.join(path, subject)
    data = loadmat(mat_path)  # 例如 shape: (62, time_steps, 5)

    # 2. 读取标签文件
    mat_path1 = os.path.join(path, "label.mat")
    label_dict = loadmat(mat_path1)
    if 'label' not in label_dict.keys():
        raise ValueError("label.mat 文件中未找到 'label' 键。")

    # 形如 (15,), 里头可能包含 -1, 0, 1
    label_data = label_dict['label'].squeeze()
    print("label_data shape:", label_data.shape)
    print("label_data:", label_data)

    samples = []
    labels = []

    print(f"Loading data from: {mat_path}")

    # 假设有 15 个 trial（若有 24 个则把 range 改为 range(1,25)）
    for i in range(1, 16):
        key = f'de_LDS{i}'
        if key not in data:
            print(f"Warning: {key} not found in {subject}, skip.")
            continue

        # trial_data: (62, time_steps, 5)
        trial_data = data[key].astype(np.float32)
        print(trial_data.shape)

        time_steps = trial_data.shape[1]

        for t in range(time_steps):
            time_slice = trial_data[:, t, :]
            samples.append(time_slice)

            # --- 这里对标签进行偏移处理 ---
            # 如果原始标签仅有 -1,0,1，则让 -1->0, 0->1, 1->2
            raw_label = label_data[i - 1]  # 原始标签
            mapped_label = raw_label + 1  # 偏移 1
            # 如果原本是 -1，则 mapped_label=0；原本是0->1；原本是1->2
            labels.append(mapped_label)

    if not samples:
        raise ValueError(f"No valid samples found in {subject}")

    print(f"Number of samples collected: {len(samples)}")
    print(f"Shape of first sample: {samples[0].shape}")

    # stack 成 (n_total, 62, 5)
    samples = np.stack(samples)
    labels = np.array(labels, dtype=np.int32)

    # 扩展维度 -> (n_total, 62, 1, 5)
    samples = np.expand_dims(samples, axis=2)

    print(f"Final samples shape: {samples.shape}")
    print(f"Final labels shape: {labels.shape}")

    # 可选的归一化

    # 打乱
    indices = np.random.permutation(len(samples))
    samples = samples[indices]
    labels = labels[indices]

    # 划分训练集 (80%) 和 测试集 (20%)
    n_train = int(0.8 * len(samples))
    return {
        "x_tr": samples[:n_train],
        "y_tr": labels[:n_train],
        "x_ts": samples[n_train:],
        "y_ts": labels[n_train:]
    }


def load_data_de2(path, subject, session):
    pass


def load_data_de3(path, subject):
    pass


def load_data_vmd(path, subject):
    mat_path = os.path.join(path, subject)
    data = np.load(mat_path, allow_pickle=True)
    data = data.item()
    # 例如 shape: (62, time_steps, 5)

    # 2. 读取标签文件
    mat_path1 = os.path.join(path, "label.mat")
    label_dict = loadmat(mat_path1)
    if 'label' not in label_dict.keys():
        raise ValueError("label.mat 文件中未找到 'label' 键。")

    # 形如 (15,), 里头可能包含 -1, 0, 1
    label_data = label_dict['label'].squeeze()
    print("label_data shape:", label_data.shape)
    print("label_data:", label_data)

    label_data = label_dict['label'].squeeze()  # shape: (15,)
    print(label_data)
    for i in range(0, 15):
        label_data[i] = label_data[i] + 1
    print(label_data)
    if label_data.shape[0] != 15:
        raise ValueError(f"期望 label 大小为15，实际为 {label_data.shape[0]}。")

    print(f"加载文件: {mat_path}")
    print(f"加载标签: {label_data}, label shape = {label_data.shape}")

    # 3. 收集样本与标签
    samples_list = []
    labels_list = []

    # 筛选出非 '__xxx__' 开头的键，通常是有效会话
    keys = [k for k in data.keys() if not k.startswith('__')]
    if len(keys) != 15:
        print(f"警告: {subject} 中实际会话数量为 {len(keys)}，而非 15，请核对数据文件。")

    # 假设 keys 的顺序与 label_data 一一对应
    # 若顺序不一致，需要根据实际情况排序或匹配
    for i, key in enumerate(keys):
        session_data = data[key]  # shape: (time, 62, 5)
        if not isinstance(session_data, np.ndarray):
            print(f"跳过 {key}, 数据类型不是 ndarray")
            continue

        # 确保是 float32
        session_data = session_data.astype(np.float32)

        # time, 62, 5
        time_steps = session_data.shape[0]
        if session_data.shape[1] != 62 or session_data.shape[2] != 5:
            print(f"警告: {key} 的维度不是 (time, 62, 5)，实际为 {session_data.shape}")

        # 将每个时间步视为一个样本 (time_steps, 62, 5)
        # 如果您想保留时序信息，可不要 reshape
        # 也可根据需要添加额外维度
        # 这里直接使用 (time_steps, 62, 5)
        # 对每个时间步赋同样的标签
        label_val = label_data[i]

        samples_list.append(session_data)  # (time_steps, 62, 5)
        labels_list.append(np.full((time_steps,), label_val, dtype=np.int32))

    if not samples_list:
        raise ValueError(f"{subject} 中没有有效的会话数据或键。")

    # 4. 合并所有会话
    #   - X shape: (sum_time, 62, 5)
    #   - Y shape: (sum_time,)
    X = np.concatenate(samples_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)

    print(f"合并后 X shape = {X.shape}, Y shape = {Y.shape}")

    # 5. 训练/测试集划分
    test_size = 0.4
    random_state = 42
    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"训练集大小: {x_tr.shape[0]}, 测试集大小: {x_ts.shape[0]}")
    # x_tr = np.expand_dims(x_tr, axis=2)
    # x_ts = np.expand_dims(x_ts, axis=2)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_ts.shape)
    print(y_ts.shape)

    # 6. 返回结果
    return {
        "x_tr": x_tr,  # shape: (N_train, 62, 5)
        "y_tr": y_tr,
        "x_ts": x_ts,  # shape: (N_test, 62, 5)
        "y_ts": y_ts
    }


def load_data_vmd1(path, subject):
    # 1. 加载数据文件
    mat_path = os.path.join(path, subject)
    data = np.load(mat_path, allow_pickle=True)
    data = data.item()
    # 假设每个键对应一个会话，数据形状为 (time_steps, 62, 5)

    # 2. 定义新的标签 label_seed4，共 24 个标签
    label_seed4 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    label_seed4 = np.array(label_seed4, dtype=np.int32)

    # 3. 收集样本与标签
    samples_list = []
    labels_list = []

    # 筛选出非 '__xxx__' 开头的键（共 24 个键）
    keys = [k for k in data.keys() if not k.startswith('__')]
    if len(keys) != 24:
        print(f"警告: 数据文件中有效会话数量为 {len(keys)}，而非 24，请核对数据文件。")

    # 为确保顺序一致，可以对键进行排序（前提是键名称可排序）
    def extract_numeric_session(key):
        match = re.search(r'\d+', key)  # 提取数字部分
        return int(match.group()) if match else float('inf')

    keys = sorted(data.keys(), key=extract_numeric_session)

    # 遍历每个会话数据，并为每个会话的所有时间步赋予对应的标签
    for i, key in enumerate(keys):
        print(i)
        print(key)
        session_data = data[key]  # 形状: (time_steps, 62, 5)
        if not isinstance(session_data, np.ndarray):
            print(f"跳过 {key}，数据类型不是 ndarray")
            continue

        # 确保数据为 float32 类型
        session_data = session_data.astype(np.float32)
        time_steps = session_data.shape[0]
        if session_data.shape[1] != 62 or session_data.shape[2] != 5:
            print(f"警告: {key} 的数据形状为 {session_data.shape}，而非 (time_steps, 62, 5)")

        # 当前会话对应的标签
        label_val = label_seed4[i]
        # 为当前会话的所有时间步生成相同的标签数组
        session_labels = np.full((time_steps,), label_val, dtype=np.int32)

        samples_list.append(session_data)
        labels_list.append(session_labels)

    if not samples_list:
        raise ValueError("没有找到有效的会话数据。")

    # 4. 合并所有会话数据
    #    X shape: (总时间步, 62, 5)
    #    Y shape: (总时间步,)
    X = np.concatenate(samples_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)

    print(f"合并后 X shape = {X.shape}, Y shape = {Y.shape}")

    # 5. 划分训练集和测试集
    test_size = 0.4
    random_state = 42
    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"训练集大小: {x_tr.shape[0]}, 测试集大小: {x_ts.shape[0]}")
    print("x_tr shape:", x_tr.shape)
    print("y_tr shape:", y_tr.shape)
    print("x_ts shape:", x_ts.shape)
    print("y_ts shape:", y_ts.shape)

    # 6. 返回结果
    return {
        "x_tr": x_tr,  # shape: (N_train, 62, 5)
        "y_tr": y_tr,
        "x_ts": x_ts,  # shape: (N_test, 62, 5)
        "y_ts": y_ts
    }


import os
import numpy as np
from scipy.io import loadmat


def load_data_vmd_leave_one_subject_out(path, test_subject, random_state=42):
    """
    留一个个体作为测试集 (Leave-One-Subject-Out Cross Validation, LOSO)

    参数：
    - path: 数据文件所在目录
    - test_subject: 作为测试集的受试者文件名 (应为 .npy 文件)
    - random_state: 随机种子，保证可复现

    返回：
    - dict, 包含训练/测试数据：
        {
            "x_tr": np.ndarray, # 训练数据 (N_train, 62, 5)
            "y_tr": np.ndarray, # 训练标签 (N_train,)
            "x_ts": np.ndarray, # 测试数据 (N_test, 62, 5)
            "y_ts": np.ndarray  # 测试标签 (N_test,)
        }
    """
    # 获取所有受试者文件（排除 label.mat）
    subjects = [f for f in os.listdir(path) if f.endswith('.npy') and f != "label.mat"]
    if len(subjects) < 2:
        raise ValueError("数据集受试者数量过少，至少需要 2 个受试者才能进行 LOSO 训练！")

    # 确保测试受试者存在
    if test_subject not in subjects:
        raise ValueError(f"指定的测试受试者 {test_subject} 不存在，请检查文件名。")

    # 读取标签信息
    label_path = os.path.join(path, "label.mat")
    label_dict = loadmat(label_path)
    if 'label' not in label_dict:
        raise ValueError("label.mat 文件中未找到 'label' 键")

    label_data = label_dict['label'].squeeze()  # shape: (N_subjects, 15) 每个受试者有 15 个会话的标签
    print(label_data)
    label_data += 1
    print(label_data)
    # if label_data.shape[0] != len(subjects) or label_data.shape[1] != 15:
    #     raise ValueError(f"标签矩阵 {label_data.shape} 与受试者数量 {len(subjects)} 不匹配！")

    # 训练受试者列表（排除测试受试者）
    train_subjects = [subj for subj in subjects if subj != test_subject]

    print(f"训练受试者数量: {len(train_subjects)}, 测试受试者: {test_subject}")

    # **🔹 加载单个受试者的数据**
    def load_subject_data(subject_list, is_test=False):
        samples_list, labels_list = [], []
        for subj in subject_list:
            subj_path = os.path.join(path, subj)
            data = np.load(subj_path, allow_pickle=True).item()

            # 获取当前受试者的索引，以便正确匹配 `label_data`
            # subj_index = subjects.index(subj)
            # subj_labels = label_data[subj_index]  # shape: (15,)

            # 确保会话顺序匹配 `label_data`
            def extract_numeric_session(key):
                """ 从会话 key 中提取数字部分，例如 '13_20140527_xyl_eeg1_VMD_DE8' -> 13 """
                match = re.search(r'\d+', key)  # 提取数字部分
                return int(match.group()) if match else float('inf')

            session_keys = sorted(data.keys(), key=extract_numeric_session)
            if len(session_keys) != 15:
                raise ValueError(f"{subj} 受试者的会话数 {len(session_keys)} 不等于 15，请检查数据完整性！")

            for i, key in enumerate(session_keys):
                # print(i)
                # print(key)
                session_data = data[key]  # shape: (time_steps, 62, 5)
                time_steps = session_data.shape[0]

                # 赋予正确的会话标签
                label = label_data[i]  # 当前会话对应的情绪标签
                samples_list.append(session_data)  # (time_steps, 62, 5)
                labels_list.append(np.full((time_steps,), label, dtype=np.int32))

        # **🔹 合并数据**
        x_data = np.concatenate(samples_list, axis=0)  # (N_samples, 62, 5)
        y_data = np.concatenate(labels_list, axis=0)  # (N_samples,)

        return x_data, y_data

    # 加载训练数据
    x_tr, y_tr = load_subject_data(train_subjects)

    # 加载测试数据
    x_ts, y_ts = load_subject_data([test_subject], is_test=True)

    print(f"训练集大小: {x_tr.shape[0]}, 测试集大小: {x_ts.shape[0]}")
    print(f"x_tr shape: {x_tr.shape}, y_tr shape: {y_tr.shape}")
    print(f"x_ts shape: {x_ts.shape}, y_ts shape: {y_ts.shape}")

    return {
        "x_tr": x_tr,  # shape: (N_train, 62, 5)
        "y_tr": y_tr,
        "x_ts": x_ts,  # shape: (N_test, 62, 5)
        "y_ts": y_ts
    }


# def load_data_inde(path, subject):
#     """
#     Independent实验数据加载：
#       - 以 `subject` 对应文件作为测试集
#       - 目录下其他所有 .mat 文件作为训练集
#     假设每个 .mat 文件都包含与 load_data_de 类似的差分熵特征 de_LDS1~de_LDS24，
#     以及用 label_seed4[0] 来给 trial 做标签（4分类：0,1,2,3）。
#
#     如果你的数据标签/处理逻辑不同，请在 parse_mat_file 函数中自行修改。
#     """
#
#     # 先定义一个辅助函数，用来解析单个被试的 .mat 文件
#     def parse_mat_file(mat_path):
#         """
#         这里的逻辑与你 load_data_de 类似，
#         读取 24 个 trial，每个 trial 的 shape: (62, time_steps, 5)
#         并将时间维度拆成多个样本；以 label_seed4[0] 作为标签。
#         """
#         label_seed4 = [
#             [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
#             [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
#             [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
#         ]
#
#         data = loadmat(mat_path)
#         samples_list = []
#         labels_list = []
#
#         # 这里写死了只用 label_seed4[0]，如果你想根据不同被试/会话用其他行，请改一下。
#         for i in range(1, 25):  # 假设每个被试都有24个trial
#             key = f'de_LDS{i}'
#             if key in data:
#                 trial_data = data[key].astype(np.float32)  # (62, time_steps, 5)
#                 time_steps = trial_data.shape[1]
#
#                 # 拆分时间维度
#                 for t in range(time_steps):
#                     time_slice = trial_data[:, t, :]  # shape (62, 5)
#                     samples_list.append(time_slice)
#                     # 这里的标签取 label_seed4[0][i-1]，你可以改成别的行或别的逻辑
#                     labels_list.append(label_seed4[0][i - 1])
#
#         if not samples_list:
#             raise ValueError(f"No valid trials found in {mat_path}")
#
#         # 转成 numpy
#         samples_arr = np.stack(samples_list)  # (n_samples, 62, 5)
#         labels_arr = np.array(labels_list, dtype=np.int32)
#
#
#         # 如果你需要增加维度形状 (n_samples, 62, 1, 5)
#         # samples_arr = np.expand_dims(samples_arr, axis=2)
#
#         return samples_arr, labels_arr
#
#     # 分别收集“训练用”的数组 和 “测试用”的数组
#     train_samples, train_labels = [], []
#     test_samples, test_labels = [], []
#
#     # 遍历目录下所有文件
#     for filename in os.listdir(path):
#         # 只处理 .mat 文件，可根据实际情况再做判断
#         if not filename.endswith('.mat'):
#             continue
#
#         # 构造完整路径
#         mat_path = os.path.join(path, filename)
#
#         # 如果该文件就是我们指定的 subject => 做测试集
#         if filename == subject:
#             x_ts, y_ts = parse_mat_file(mat_path)
#             x_ts=standardize(x_ts)
#             test_samples.append(x_ts)
#             test_labels.append(y_ts)
#         else:
#             # 否则并入训练集
#             x_tr, y_tr = parse_mat_file(mat_path)
#             x_tr = standardize(x_tr)
#             train_samples.append(x_tr)
#             train_labels.append(y_tr)
#
#     # 如果测试被试没找到，可能 subject 名字不对？
#     if not test_samples:
#         raise ValueError(f"指定的测试被试文件 {subject} 不存在或不是 .mat 文件")
#
#     # 合并所有训练被试
#     if train_samples:
#         x_tr_merged = np.concatenate(train_samples, axis=0)
#         y_tr_merged = np.concatenate(train_labels, axis=0)
#     else:
#         raise ValueError("没有找到除测试被试外的任何 .mat 文件用于训练，请检查目录。")
#
#     # 合并测试被试（如果 subject 只有一个文件，一般就只有一次 append）
#     x_ts_merged = np.concatenate(test_samples, axis=0)
#     y_ts_merged = np.concatenate(test_labels, axis=0)
#
#     print(f"独立实验: 训练样本总数 {x_tr_merged.shape[0]}, 测试样本总数 {x_ts_merged.shape[0]}")
#     print(f"训练集形状: {x_tr_merged.shape}, 测试集形状: {x_ts_merged.shape}")
#     # ============ 在这里随机打乱训练集 =============
#     # train_perm = np.random.permutation(len(x_tr_merged))
#     # x_tr_merged = x_tr_merged[train_perm]
#     # y_tr_merged = y_tr_merged[train_perm]
#     #
#     # # 如果也想打乱测试集，则可再加一段
#     # test_perm = np.random.permutation(len(x_ts_merged))
#     # x_ts_merged = x_ts_merged[test_perm]
#     # y_ts_merged = y_ts_merged[test_perm]
#     # x_tr_merged, x_ts_merged = standardize_data(x_tr_merged, x_ts_merged)
#
#
#     # 不做二次切分，因为我们就是 (train vs. test) 在被试层面划分
#     return {
#         "x_tr": x_tr_merged,
#         "y_tr": y_tr_merged,
#         "x_ts": x_ts_merged,
#         "y_ts": y_ts_merged
#     }

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleCoreResidualFusion(nn.Module):
    """
    以 f1 为核心特征，f2 作为辅助特征：
      1) 通过 Cross-Attention 让 f1 从 f2 中获取信息
      2) 得到的注意力结果与原始 f1 残差相加
      3) 最后输出融合后的特征

    注意：
      - embed_dim 必须能被 num_heads 整除
      - 若 f1/f2 的 shape = (batch, seq_len, feature_dim)
        则 embed_dim = feature_dim，num_heads * head_dim = embed_dim
    """

    def __init__(self, in_dim, out_dim, num_heads=1):
        super(SingleCoreResidualFusion, self).__init__()
        # f1 作为“核心”，只做一次 Cross-Attention: Query=f1, Key=Value=f2
        # batch_first=True => 输入/输出 (batch, seq_len, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 将残差后的 f1 做线性变换
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, f1, f2):
        """
        f1, f2: shape=(batch_size, seq_len, in_dim)
        返回: shape=(batch_size, seq_len, out_dim)
        """
        # 1) Cross-Attention: f1 (query) 从 f2 (key=value) 获取信息
        #    attn_f1 的 shape 与 f1 一致 (batch, seq_len, in_dim)
        attn_f1, _ = self.cross_attn(query=f1, key=f2, value=f2)

        # 2) 残差连接：f1 + attn_f1
        fused_f1 = attn_f1 + f1

        # 3) 全连接层或其他后续处理
        out = self.fc(fused_f1)  # (batch, seq_len, out_dim)
        return out


class ResidualFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualFusion, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=5, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=5, batch_first=True)
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(self, f1, f2):
        # 通过自注意力机制处理两个特征
        attn_f1, _ = self.attention1(f1, f2, f2)
        attn_f2, _ = self.attention2(f2, f1, f1)

        # 残差连接：将注意力后的特征与原始特征相加
        fused_f1 = attn_f1 + f1
        fused_f2 = attn_f2 + f2

        # 合并残差后的特征
        combined = torch.cat((fused_f1, fused_f2), dim=-1)

        # 通过全连接层生成最终输出
        out = self.fc(combined)
        return out


import torch
import torch.nn as nn

class CAB(nn.Module):
    """通道注意力模块（适用于1D卷积）"""
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out

class SAB(nn.Module):
    """空间（时间）注意力模块（适用于1D卷积）"""
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale

class MSCB(nn.Module):
    """多尺度卷积模块（适用于1D卷积）"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MSCB, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        self.pointwise = nn.Sequential(
            nn.Conv1d(len(kernel_sizes)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        out = self.pointwise(out)
        return out


class EnhancedTemporalLearner(nn.Module):
    """
    融合多尺度与注意力机制的1D卷积特征提取模块（输出维度与原版保持一致）
    输入: (B, T, 1)
    输出: (B, out_channels * num_kernels)
    """

    def __init__(self, kernel_sizes=[3, 5, 7], out_channels=50):
        super(EnhancedTemporalLearner, self).__init__()
        self.initial_conv = nn.Conv1d(1, out_channels, 1)
        self.cab = CAB(out_channels)
        self.sab = SAB()

        # 注意这里修改 MSCB 输出通道数
        self.mscb = MSCB(out_channels, out_channels, kernel_sizes)

        # BN和ReLU层修改为对应的输出维度（out_channels * num_kernels）
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, T, 1)
        x = x.transpose(1, 2)  # (B, 1, T)
        x = self.initial_conv(x)  # (B, out_channels, T)
        x = self.cab(x)  # 通道注意力 (B, out_channels, T)
        x = self.sab(x)  # 空间注意力 (B, out_channels, T)

        # 改为直接拼接多尺度特征
        multi_scale_feats = [conv(x) for conv in self.mscb.convs]  # 每个尺度 (B,out_channels,T)
        x_cat = torch.cat(multi_scale_feats, dim=1)  # 拼接 (B,150,T)

        x = self.bn(x_cat)
        x = self.relu(x)
        out = x.mean(dim=2)  # 平均池化 (B, 150)
        return out


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
def load_data_inde_yuan(path, subject):
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
    temporal_learner = TemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
    fusion_fc = torch.nn.Linear(50 * 3, 1)  # 输出维度为 1
    # 固定为 eval 模式，不需要梯度
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed4 = [
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ]
        data = loadmat(mat_path)
        samples_list = []
        labels_list = []
        # 对每个 trial (1~15)
        for i in range(1, 16):
            key = f'de_LDS{i}'
            if key in data:
                trial_data = data[key].astype(np.float32)  # 原始形状: (62, T, 5)
                # 使用滑动窗口切分，步长设置为 4 增大重叠率
                windows = segment_eeg_sliding(trial_data, window_size=16, step=4)
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
                    labels_list.append(label_seed4[0][i - 1])
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
def load_data_denpendent(path, subject):
    """
    Independent实验数据加载：
      - 所有的 .mat 文件作为同一数据集，不区分训练集和测试集。
    假设每个 .mat 文件包含 24 个 trial，每个 trial 的数据形状为 (62, T, 5)。
    对每个 trial，先采用滑动窗口切分（窗口大小为16，步长为4，即窗口重叠较大），
    然后对每个窗口使用 TemporalLearner 提取时序特征，输出为 (62, 5) 的表示（每个频段提取1个特征）。
    标签按 label_seed4[1] 提取（4分类：0,1,2,3）。
    """
    # 定义 TemporalLearner 和融合层，用于将多尺度卷积输出降维为 1 维
    temporal_learner = EnhancedTemporalLearner(kernel_sizes=[3, 5,7], out_channels=50)
    fusion_fc = torch.nn.Linear(50 * 3, 1)  # 输出维度为 1
    # 固定为 eval 模式，不需要梯度
    temporal_learner.eval()
    fusion_fc.eval()

    def parse_mat_file(mat_path):
        label_seed4 = [
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ]
        data = loadmat(mat_path)
        train_samples, train_labels = [], []
        test_samples, test_labels = [], []
        # 对每个 trial (1~24)
        keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]

        for i in range(1, 16):
            # key = f'de_LDS{i}'
            prefix = keys[0].split("_")[0]  # 如 "ha
            key = f"{prefix}_eeg{i}"
            if key in data:
                print(key)
                print(data[key].shape)
                trial_data = data[key].astype(np.float32)  # 原始形状: (62, T, 5)
                # 使用滑动窗口切分，步长设置为 4 增大重叠率
                windows = segment_eeg_sliding(trial_data, window_size=16, step=4)
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
                    if i<=12:
                       train_samples.append(window_feature)
                    # 对于每个窗口，标签取 trial 对应的标签： label_seed4[1][i-1]
                       train_labels.append(label_seed4[0][i - 1])
                    else:
                       test_samples.append(window_feature)
                        # 对于每个窗口，标签取 trial 对应的标签： label_seed4[1][i-1]
                       test_labels.append(label_seed4[0][i - 1])
        if not train_samples:
            raise ValueError(f"No valid trials found in {mat_path}")
        # 合并所有窗口样本，得到 (n_windows, 62, 5)
        samples_tr = np.stack(train_samples)
        labels_tr = np.array(train_labels, dtype=np.int32)
        samples_ts = np.stack(test_samples)
        labels_ts = np.array(test_labels, dtype=np.int32)
        return samples_tr, labels_tr,samples_ts,labels_ts
        # 打乱

    mat_path = os.path.join(path, subject)
    x_tr, y_tr,x_ts, y_ts = parse_mat_file(mat_path)
    x_tr = standardize(x_tr)
    x_ts=standardize(x_ts)

    print(f"训练集大小: {x_tr.shape[0]}, 测试集大小: {x_ts.shape[0]}")
    # x_tr = np.expand_dims(x_tr, axis=2)
    # x_ts = np.expand_dims(x_ts, axis=2)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_ts.shape)
    print(y_ts.shape)

    # 6. 返回结果
    return {
        "x_tr": x_tr.detach().numpy(),  # shape: (N_train, 62, 5)
        "y_tr": y_tr,
        "x_ts": x_ts.detach().numpy(),  # shape: (N_test, 62, 5)
        "y_ts": y_ts
    }

import os
import torch
import numpy as np
from scipy.io import loadmat

# def load_data_denpendent2(path, subject):
#     """
#     Independent 实验数据加载（跨会话实验）：
#       - 对于每个标签（0,1,2,3），选取第一个出现该标签的 trial 整个会话作为测试，
#         其余 trials 作为训练。
#       - 假设每个 .mat 文件包含 24 个 trial，每个 trial 的数据形状为 (62, T, 5)。
#       - 对每个 trial，滑动窗口切分（window_size=16, step=4），
#         然后用 TemporalLearner 提取时序特征，输出 (62,1)×5→(62,5)。
#       - 标签从 label_seed4[0] 中取。
#     """
#     # 定义特征提取模型（Conv1d 时序特征）
#     temporal_learner = TemporalLearner(kernel_sizes=[3, 5, 7], out_channels=50)
#     fusion_fc = torch.nn.Linear(50 * 3, 1)
#     temporal_learner.eval()
#     fusion_fc.eval()
#
#     def parse_mat_file(mat_path):
#         # 预定义的 24 个 trial 标签
#         label_seed4 = [
#             [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
#             [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
#             [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
#         ]
#         labels = label_seed4[2]  # 取第一组标签
#         # 选取每个标签第一次出现的 trial 作为测试集
#         test_trials = []
#         for target_label in range(4):
#             for idx, lbl in enumerate(labels, start=1):
#                 if lbl == target_label:
#                     test_trials.append(idx)
#                     break
#
#         data = loadmat(mat_path)
#         train_samples, train_labels = [], []
#         test_samples,  test_labels  = [], []
#
#         keys = [k for k in data if k.endswith("eeg1") or k.endswith("eeg01")]
#         prefix = keys[0].split("_")[0] if keys else ""
#
#         for i in range(1, 25):
#             key = f"{prefix}_eeg{i}"
#             if key not in data:
#                 continue
#
#             trial_data = data[key].astype(np.float32)  # (62, T, 5)
#             windows = segment_eeg_sliding(trial_data, window_size=16, step=4)
#
#             for window in windows:
#                 with torch.no_grad():
#                     # window: (62,16,5)
#                     channel_feats = []
#                     wt = torch.from_numpy(window)
#                     for f in range(5):
#                         freq_data = wt[:, :, f].unsqueeze(-1)          # (62,16,1)
#                         tmp = temporal_learner(freq_data)               # (62,150)
#                         red = fusion_fc(tmp)                            # (62,1)
#                         channel_feats.append(red)
#                     feat62x5 = torch.cat(channel_feats, dim=-1).cpu().numpy()  # (62,5)
#
#                 if i in test_trials:
#                     test_samples.append(feat62x5)
#                     test_labels.append(labels[i-1])
#                 else:
#                     train_samples.append(feat62x5)
#                     train_labels.append(labels[i-1])
#
#         if not train_samples:
#             raise ValueError(f"No training windows in {mat_path}")
#         if not test_samples:
#             raise ValueError(f"No testing windows in {mat_path}")
#
#         return (
#             np.stack(train_samples),
#             np.array(train_labels, dtype=np.int32),
#             np.stack(test_samples),
#             np.array(test_labels,  dtype=np.int32),
#         )
#
#     mat_path = os.path.join(path, subject)
#     x_tr, y_tr, x_ts, y_ts = parse_mat_file(mat_path)
#
#     # 标准化
#     x_tr = standardize(x_tr)
#     x_ts = standardize(x_ts)
#
#     print(f"训练集: samples={x_tr.shape}, labels={y_tr.shape}")
#     print(f"测试集: samples={x_ts.shape}, labels={y_ts.shape}")
#
#     return {
#         "x_tr": x_tr.detach().numpy(),  # (N_train, 62, 5)
#         "y_tr": y_tr,
#         "x_ts": x_ts.detach().numpy(),  # (N_test,  62, 5)
#         "y_ts": y_ts
#     }


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
            key = f"session_{i:03d}"
            print(key)
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
            if key not in data: continue
            trial_data = data[key].astype(np.float32)
            windows = segment_eeg_sliding(trial_data, window_size=16, step=4)
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


import numpy as np


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

