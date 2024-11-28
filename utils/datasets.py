from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from scipy.signal import butter, sosfilt, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from utils.helper_code import *
import torch.fft as fft
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class dataset:
    classes = ['164889003', '164890007', '6374002', '426627000', '733534002',
               '713427006', '270492004', '713426002', '39732003', '445118002',
               '164947007', '251146004', '111975006', '698252002', '426783006',
               '284470004', '10370003', '365413008', '427172004', '164917005',
               '47665007', '427393009', '426177001', '427084000', '164934002',
               '59931005']
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'],
                          ['284470004', '63593006'],
                          ['427172004', '17338001'],
                          ['733534002', '164909002']]

    def __init__(self, header_files):
        self.files = []
        self.num_leads = None
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h
            tmp['record'] = h.replace('.hea', '.mat')
            hdr = load_header(h)
            tmp['nsamp'] = get_nsamp(hdr)
            tmp['leads'] = get_leads(hdr)
            tmp['age'] = get_age(hdr)
            tmp['sex'] = get_sex(hdr)
            tmp['dx'] = get_labels(hdr)
            tmp['fs'] = get_frequency(hdr)
            tmp['target'] = np.zeros((26,))
            tmp['dx'] = replace_equivalent_classes(tmp['dx'], dataset.equivalent_classes)
            for dx in tmp['dx']:
                # in SNOMED code is in scored classes
                if dx in dataset.classes:
                    idx = dataset.classes.index(dx)
                    tmp['target'][idx] = 1
            self.files.append(tmp)

        self.files = pd.DataFrame(self.files)

    def summary(self, output):
        if output == 'pandas':
            return pd.Series(np.stack(self.files['target'].to_list(), axis=0).sum(axis=0), index=dataset.classes)
        if output == 'numpy':
            return np.stack(self.files['target'].to_list(), axis=0).sum(axis=0)

    def __len__(self):
        return len(self.files)


def read_data(data_name, data_directory):
    header_files, recording_files = find_challenge_files(data_directory)
    print('File Reading!')
    full_dataset = dataset(header_files)
    print(full_dataset.summary('numpy'))
    data_list = []
    label_list = []
    print('Data Reading!')
    for i in tqdm(range(len(full_dataset))):
        data = load_recording(full_dataset.files.iloc[i]['record'])
        target = full_dataset.files.iloc[i]['target']
        data_list.append(data)
        label_list.append(target)
    data_array = np.array(data_list)
    label_array = np.array(label_list)
    print(data_array.shape)
    print('Data Processing!')
    data_t = data_process(data_array, 0.05, 75, 50, 500)
    print(data_t.shape)
    file_name_t = '../dataset/ecg/' + data_name
    file_name_l = '../dataset/ecg/' + data_name + '_label'
    np.save(file_name_t, data_t)
    np.save(file_name_l, label_array)


def to_one_hot(labels):
    # 获取标签的唯一值
    unique_labels = np.unique(labels)

    # 创建一个全零矩阵作为One-Hot编码结果的容器
    one_hot_labels = np.zeros((labels.shape[0], unique_labels.shape[0]))

    # 遍历标签数组，将对应位置的元素设置为1
    for i, label in enumerate(labels):
        one_hot_labels[i, int(label)] = 1
    return one_hot_labels


def down_samples(fs_o, fs_t, signal):
    original_sample_rate = fs_o  # 填入原始信号的采样率
    # 目标采样率
    target_sample_rate = fs_t  # 填入目标采样率
    # 计算降采样的比例
    downsampling_factor = original_sample_rate / target_sample_rate
    # 计算目标样本数量
    target_num_samples = int(signal.shape[-1] / downsampling_factor)
    # 进行降采样
    downsampled_signal = resample(signal, target_num_samples, axis=1)

    return downsampled_signal


def data_process(data, lowcut, highcut, notch_freq, fs):
    # 原始信号
    signal = data  # 填入你的原始信号数据
    print(signal.shape)
    # 生成巴特沃斯带通滤波器系数
    butter_order = 2  # 滤波器阶数
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos1 = butter(butter_order, [low, high], btype='band', output='sos')

    # 陷波滤波器参数
    notch_freq = notch_freq  # 陷波滤波器的频率 (Hz)
    notch_width = 2.0  # 陷波滤波器的带宽 (Hz)
    notch_low = (notch_freq - notch_width / 2) / nyquist
    notch_high = (notch_freq + notch_width / 2) / nyquist
    sos2 = butter(butter_order, [notch_low, notch_high],
                              btype='bandstop', output='sos')

    final_signal_list = []
    for ss in tqdm(signal):
        # 应用巴特沃斯带通滤波器
        filtered_signal = sosfilt(sos1, sosfiltfilt(sos2, ss))
        # 下采样10倍
        down_signal = down_samples(fs, 100, filtered_signal)
        # 标准化
        final_signal = StandardScaler().fit_transform(down_signal.T).T
        final_signal_list.append(final_signal)

    return np.array(final_signal_list)


def normalize_and_get_stats(tensor):
    # 计算均值和方差，dim=2 表示沿最后一个维度计算
    mean = tensor.mean(axis=-1, keepdims=True)
    std = tensor.std(axis=-1, keepdims=True)

    # 标准化张量
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # 1e-8 用于防止除以零

    # 返回标准化后的张量、均值和方差
    return normalized_tensor, mean, std


class DatasetECG_TF(Dataset):

    def __init__(self, signals, labels):
        super(DatasetECG_TF, self).__init__()
        self.signals = signals
        self.labels = labels

    def __getitem__(self, index):
        data_time = self.signals[index]
        data_time = torch.tensor(data_time.copy(), dtype=torch.float)
        data_freq = fft.fft(data_time).abs()
        data_freq, _, _ = normalize_and_get_stats(data_freq)
        data_labels = torch.tensor(self.labels[index].copy(), dtype=torch.float)
        len_f = int(0.5 * data_freq.shape[1])
        data_dict = {'ecg_t': data_time, 'ecg_f': data_freq[:, :len_f]}
        return data_dict, data_labels

    def __len__(self):
        return len(self.labels)


def build_dataset_ecg(args):
    data_t = np.load(args.data_path)
    print(data_t.shape[0])
    if args.labels_path == 'dataset/ecg/ningbo_label.npy':
        label = np.load(args.labels_path)[:, 1:]
    elif args.labels_path == 'dataset/ecg/ptb-xl_label.npy':
        label0 = np.load(args.labels_path)
        label = np.concatenate((label0[:, :2], label0[:, 4:17], label0[:, 19:]), axis=1)
    elif args.labels_path == 'dataset/ecg/shaoxing_label.npy':
        label0 = np.load(args.labels_path)
        label = np.concatenate(
            (label0[:, :2], label0[:, 4:7], label0[:, 8:9], label0[:, 10:16], label0[:, 18:21], label0[:, 22:]), axis=1)
    else:
        label = np.load(args.labels_path)

    # 使用分层采样划分数据集
    stratified_split = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    index_list = []
    for train_index, val_index in stratified_split.split(data_t, label):
        index_list.append(val_index)

    t_index = index_list[0]
    v_index = index_list[4]
    if args.folds != 1:
        for i in range(1, args.folds):
            t_index = np.concatenate([t_index, index_list[i]])

    X1_train, X1_val = data_t[t_index], data_t[v_index]
    y_train, y_val = label[t_index], label[v_index]
    print(t_index)

    weights = y_train.shape[0] / (y_train.shape[1] * y_train.sum(axis=0))
    weights = torch.from_numpy(weights.astype(np.float32))

    print(weights)

    data_train = DatasetECG_TF(X1_train, y_train)
    data_val = DatasetECG_TF(X1_val, y_val)

    return data_train, data_val, weights


if __name__ == '__main__':
    read_data('The name of the dataset', 'The absolute path to the dataset')
