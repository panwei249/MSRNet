import os
import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from pykalman import KalmanFilter  # LDS 滤波器

# 路径设置
data_dir = "/home/hc/work/pw/SEED4/old/1/"
save_dir = "/home/hc/work/pw/SEED4/new/7/"
os.makedirs(save_dir, exist_ok=True)

# 参数配置
sfreq = 400
window_sec = 2
window_size = window_sec * sfreq
stride = window_size  # ✅ 无重叠：每步滑动整个窗口长度

FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

# 带通滤波器
def bandpass_filter(data, sfreq, low=0.1, high=75):
    nyquist = 0.5 * sfreq
    b, a = butter(4, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data)

# 微分熵计算
def compute_de(signal_data):
    sigma = np.std(signal_data)
    return -0.5 * np.log(2 * np.pi * np.e * sigma ** 2)

# LDS 多频段特征平滑
def lds_multiband_filter(sequence_matrix):
    n_band = sequence_matrix.shape[1]
    seq_var = np.var(sequence_matrix, axis=0)
    mean_var = np.mean(seq_var)
    Q_scale = np.clip(mean_var, 1e-6, 1.0)
    R_scale = Q_scale * 5
    Q = np.eye(n_band) * 0.01 * Q_scale
    R = np.eye(n_band) * 1.00 * R_scale
    kf = KalmanFilter(
        transition_matrices=np.eye(n_band),
        observation_matrices=np.eye(n_band),
        initial_state_mean=sequence_matrix[0],
        initial_state_covariance=np.eye(n_band),
        transition_covariance=Q,
        observation_covariance=R
    )
    smoothed_state_means, _ = kf.smooth(sequence_matrix)
    return smoothed_state_means

# 主处理流程
file_names = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

for file_name in tqdm(file_names, desc="Processing Files", unit="file"):
    mat = sio.loadmat(os.path.join(data_dir, file_name))
    eeg_keys = [k for k in mat.keys() if "eeg" in k.lower() and isinstance(mat[k], np.ndarray)]
    processed_dict = {}

    for key in eeg_keys:
        data = mat[key]  # shape: (62, T)
        ch_num, n_time = data.shape

        # 1. 带通滤波
        data_filt = np.array([bandpass_filter(ch, sfreq, 0.1, 75) for ch in data])

        # 2. 重参考
        data_filt = data_filt - np.mean(data_filt, axis=0, keepdims=True)

        # 3. 滑窗（无重叠）
        segments = [
            data_filt[:, i:i + window_size]
            for i in range(0, n_time - window_size + 1, stride)
        ]

        # 4. 特征提取（62, 段数, 5）
        all_features = []
        for seg in segments:
            seg = seg - np.mean(seg, axis=1, keepdims=True)
            feat = np.zeros((ch_num, len(FREQ_BANDS)))
            for i, (band, (low, high)) in enumerate(FREQ_BANDS.items()):
                for ch in range(ch_num):
                    filtered = bandpass_filter(seg[ch], sfreq, low, high)
                    feat[ch, i] = compute_de(filtered)
            all_features.append(feat)

        features_concat = np.stack(all_features, axis=1)  # (62, 段数, 5)

        # 5. LDS 平滑每个通道多频段特征
        for ch in range(ch_num):
            features_concat[ch, :, :] = lds_multiband_filter(features_concat[ch, :, :])

        processed_dict[key] = features_concat  # shape: (62, 段数, 5)

    save_path = os.path.join(save_dir, f"processed_{file_name}")
    sio.savemat(save_path, processed_dict)

print("✅ 全部处理完成（无重叠滑窗）")
