# @Time : 2024/7/30 0030 18:32
# @Author :李旺森
# @File : mian.py
# @Software : PyCharm
# coding=utf-8
# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
from scipy import signal
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter

import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
import torch
import torch.nn.functional as F


# 设置随机种子
torch.manual_seed(42)


def count_non_negative_ones(lst):
    count = 0
    return [(count := count + (num != -1)) - (num != -1) for num in lst]


def filter_data(data):
    # 平滑
    # N：滤波器阶数（极点和零点的数量）。
    # Wn：截止频率。奈奎斯特频率的 0.01 至 0.5 范围
    # padlen：填充长度以避免边缘效应。
    # B和A：巴特沃斯滤波器传递函数的分子和分母系数。
    # 奈奎斯特频率 ( nyquist):
    # 奈奎斯特频率是采样率的一半。换句话说，它是采样数据中能够准确表示而不会出现混叠的最大频率。在数学上，它由下式给出
    Wn = 0.2
    N = 2
    B, A = butter(N, Wn, output='ba')
    smooth_values_column = filtfilt(B, A, data)
    return smooth_values_column


def reduce_by_max_abs(arr):
    # 初始化一个列表，用于存储处理后的结果
    reduced = []

    # 设置第一个元素为当前最大值
    current_max = arr[0]

    # 遍历数组，从第二个元素开始
    for i in range(1, len(arr)):
        # 如果当前元素和前一个元素符号相同
        if np.sign(arr[i]) == np.sign(current_max):
            # 更新当前最大值
            current_max = arr[i] if abs(arr[i]) > abs(current_max) else current_max
        else:
            # 如果符号不同，将当前最大值加入结果并重置当前最大值
            reduced.append(current_max)
            current_max = arr[i]

    # 把最后一个最大值加入结果
    reduced.append(current_max)

    return np.array(reduced)


if __name__ == '__main__':
    #改成你们自己的文件路径
    root_data = '喉部数据'
    output_eeg = 'output_' + root_data + '/eeg'
    output_speech = 'output_' + root_data + '/speech'
    output_speech_cut = 'output_' + root_data + '/speech/cut'
    output_speech_downSampled = 'output_' + root_data + '/speech/downSampled'

    output_eeg_err = 'output_' + root_data + '/eeg_err/'
    output_speech_err = 'output_' + root_data + '/speech_err/'

    # 确保文件夹存在
    os.makedirs(output_eeg, exist_ok=True)
    os.makedirs(output_speech, exist_ok=True)
    os.makedirs(output_speech_cut, exist_ok=True)
    os.makedirs(output_speech_downSampled, exist_ok=True)
    os.makedirs(output_eeg_err, exist_ok=True)
    os.makedirs(output_speech_err, exist_ok=True)

    # 指定EDF文件夹的路径
    # edf_folder = '../' + root_data + '_data/喉部-520hz/'
    edf_folder = '../' + root_data + '/喉部-520hz/'

    # 获取所有以 .edf 结尾的文件路径，并按文件名排序
    edf_files = glob.glob(os.path.join(edf_folder, '*.xlsx'))

    all_data = []
    for file_path in edf_files:
        # 读取 .xlsx 文件
        sheet_name = 'Sheet1'
        # 读取数据到 DataFrame，跳过前两行
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
        # 选择偶数列（假设偶数列是指列索引从 1 开始的偶数列）
        even_columns = [col for i, col in enumerate(df.columns) if i % 2 == 1]
        data = df[even_columns]
        # 将数据转换为 NumPy 数组
        data_np_1 = data.to_numpy()
        data_np_1 = np.transpose(data_np_1, (1, 0))
        all_data.append(data_np_1)

    data_np = np.vstack(all_data)
    # 假设这是你3秒的音频数据，采样率250Hz，总样本数=3秒*250Hz
    sfreq_speech = 500
    duration = 3  # seconds
    samples = duration * sfreq_speech

    # 删除前n秒数据
    begin_cut = 0 * sfreq_speech
    # 一个单词n次
    end_cut = begin_cut + 1 * samples
    # 截取数据
    data_np = data_np[:, begin_cut:end_cut]

    sfreq_speech = 100
    all_extracted_speech_data = []
    all_speech_label = []
    all_range_start = []

    # 每个单词有n组
    per_num = 100
    # 数据每100个一组（每个3s[一个单词1次]）
    for i in range(data_np.shape[0]):
        ave = 100
        while abs(ave) > 1e-10:
            ave = np.average(data_np[i])
            data_np[i] = data_np[i] - ave
        word_data = reduce_by_max_abs(data_np[i])
        print(word_data.shape)
        if word_data.shape[0] < 300 and word_data.shape[0] > 310:
            continue
        word_data = word_data[:300]
        # 每隔3秒提取一次
        segments = np.array([word_data[i:i + samples] for i in range(0, len(word_data), samples)])
        for j in range(segments.shape[0]):
            data = segments[j]

            # 使用短时能量来检测说话部分
            window_size = int(0.1 * sfreq_speech)  # 100ms的滑动窗口
            ste = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='valid')

            # 找到能量最大的一段，假设这段是说话部分
            max_index = np.argmax(ste)
            print('max_index:')
            print(max_index)
            ste_threshold = np.mean(ste) + np.std(ste)  # 设置一个阈值
            speaking_indices = np.where(ste > ste_threshold)[0]

            # 数据找不到说话部位
            if len(speaking_indices) == 0:
                print(f'数据找不到说话部位_{i}_{j}')
                all_range_start.append(-1)

                continue
            # 确定说话部分的开始和结束(在信号连续的情况下，使用这种。不连续就还得改进)
            start_speaking = speaking_indices[0]
            end_speaking = speaking_indices[-1]

            # 计算说话部分长度
            speaking_duration = end_speaking - start_speaking

            # 需要补充的数据长度
            # needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
            needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
            remaining_duration = needed_duration - speaking_duration

            if remaining_duration < 0:
                print(f'说话长度太长_阈值小了_{i}_{j}')
                all_range_start.append(-1)

                print(f'默认处理_{i}_{j}，取0.5-2s')

                left = int(0.5 * sfreq_speech)
                right = int(2 * sfreq_speech)
                all_extracted_speech_data.append(data[left:right])
                all_speech_label.append(i // per_num)
                all_range_start.append(left)

                continue
            # 左右补充的数据量
            left_padding = right_padding = int(remaining_duration / 2)

            # 确保索引不超出边界，处理边界情况
            if start_speaking - left_padding < 0:
                left = 0  # 向左延伸到边界
                right = 1.5 * sfreq_speech
            elif end_speaking + right_padding > samples:
                left = samples - 1.5 * sfreq_speech
                right = samples
            else:
                if remaining_duration % 2 == 1:  # 说明是奇数，之前被除以2的时候少了1，截断操作！！！
                    left = start_speaking - left_padding
                    right = end_speaking + right_padding + 1  # 默认从右边扣除
                else:
                    left = start_speaking - left_padding
                    right = end_speaking + right_padding

            left = int(left)
            right = int(right)
            if right - left != 750:  # debug用
                print()

            # 提取最终的1.5秒数据
            extracted_data = data[left:right]

            print(f'完成_{i}_{j}')

            all_extracted_speech_data.append(extracted_data)
            all_speech_label.append(i // per_num)
            all_range_start.append(left)

    # 计算每个位置之前（包括该位置）非-1的个数
    non_negative_ones_counts = count_non_negative_ones(all_range_start)

    all_extracted_speech_data = np.array(all_extracted_speech_data)
    all_extracted_eeg_data = all_extracted_speech_data

    labels = np.array(all_speech_label)
    print(f'label种类：{np.unique(labels)}')
    only_use_20_label = False
    # only_use_20_label = False

    if only_use_20_label:
        # 计算每个标签的个数
        label_counts = Counter(labels)

        # 找出出现次数为20的标签
        desired_labels = {label for label, count in label_counts.items() if count == per_num}

        # 使用布尔索引进行筛选
        mask = np.isin(labels, list(desired_labels))
        all_extracted_speech_data = all_extracted_speech_data[mask]
        all_extracted_eeg_data = all_extracted_eeg_data[mask]
        labels = labels[mask]

    label_stander = True

    if label_stander:
        # 重映射标签为从0开始连续的
        unique_labels = np.unique(labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print('label_mapping')
        print(label_mapping)
        labels = np.vectorize(label_mapping.get)(labels)

    all_extracted_speech_data_my_pre = []
    for i in range(all_extracted_speech_data.shape[0]):
        aaa = all_extracted_speech_data[i]

        from scipy import signal
        from scipy.signal import hilbert, savgol_filter

        # 使用 Hilbert 变换计算包络
        analytic_signal = hilbert(aaa)
        amplitude_envelope = np.abs(analytic_signal)

        # 2. 使用 Savitzky-Golay 滤波器进行平滑
        # 这里的窗口大小和多项式阶数可以调整以获得更平滑的结果
        # smoothed_envelope = savgol_filter(amplitude_envelope, window_length=51, polyorder=3)
        smoothed_envelope = savgol_filter(amplitude_envelope, window_length=11, polyorder=5)

        # 使用 SciPy 的 resample 函数进行下采样，长度减半 -> 调整至1/3
        downsampled_data = signal.resample(smoothed_envelope, len(smoothed_envelope) // 3)

        all_extracted_speech_data_my_pre.append(downsampled_data)

    all_extracted_speech_data = np.array(all_extracted_speech_data_my_pre)

    res_dict = {}

    # m = KNeighborsClassifier()  # 0.9198484848484847

    data = all_extracted_speech_data

    print(f'{data.shape}_____{labels.shape}')

    nclass = len(np.unique(labels))
    skfolds = StratifiedKFold(n_splits=10)

    result = []
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.tensor(data).float()
    lab = torch.tensor(labels).view(-1).to(torch.long)

    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    scalered_data = scaler.fit_transform(data)  # 标准化
    pca_data = PCA(n_components=23).fit_transform(scalered_data)  # PCA降纬

    # 网格搜索优化参数
    # defining parameter range
    # (1) for SVC model
    # 最佳参数： {'C': 6.15, 'gamma': 0.32, 'kernel': 'rbf'}
    # 最佳模型准确率： 0.9418989898989899; 1/3采样; PCA -> 15

    # (2) for KNN model
    # 最佳参数： {'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}
    # 最佳模型准确率： 0.9509090909090908; 1/3采样; PCA -> 23
    # param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
    #               'weights': ['uniform'],
    #               'p': [1, 2, 3, 4, 5, 6]}
    # (3) for XGBoost model
    # param_dist = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    #     'min_child_weight': [1, 2, 3, 4, 5],
    # }
    # # 初始化XGBoost分类器
    # xgb = XGBClassifier()
    # grid_search = GridSearchCV(xgb, param_dist, cv=10, scoring='accuracy')
    # # 执行网格搜索
    # grid_search.fit(pca_data, lab)
    # # 输出最佳参数和最佳模型的性能指标
    # print("最佳参数：", grid_search.best_params_)
    # print("最佳模型准确率：", grid_search.best_score_)

    for fold, (train_index, test_index) in enumerate(skfolds.split(data, lab)):
        x_train, x_test = pca_data[train_index], pca_data[test_index]
        y_train, y_test = lab[train_index], lab[test_index]

        # KNN model
        m = KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')
        m.fit(x_train, y_train)
        label1 = m.predict(x_test)
        accuracy = accuracy_score(y_test, label1)
        result.append(accuracy)
    print(np.average(result))
