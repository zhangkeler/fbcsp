import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from array import array
import os
import matplotlib



from addHandFeatures.min2net.preprocessing.BCIC2a import raw
from addHandFeatures.min2net.preprocessing.FBCSP import FBCSP
from addHandFeatures.min2net.preprocessing.config import CONSTANT
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas as pd


CONSTANT = CONSTANT['BCIC2a']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
n_trials_per_class = CONSTANT['n_trials_per_class']
n_chs = CONSTANT['n_chs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']


def subject_dependent_setting(k_folds, pick_smp_freq, n_components, n_features, bands, order, save_path, num_class=2,
                              sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path = save_path + '/BCIC2a/fbcsp/{}_class/subject_dependent'.format(num_class)
    n_chs = len(sel_chs)
    n_trials = n_trials_per_class * num_class

    X_train_all, y_train_all = np.zeros((n_subjs, n_trials, n_chs, int(MI_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))
    X_test_all, y_test_all = np.zeros((n_subjs, n_trials, n_chs, int(MI_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))

    id_chosen_chs = raw.chanel_selection(sel_chs)
    for s in range(n_subjs):
        X_train, y_train, X_test, y_test = __load_BCIC2a(raw_path, s + 1, pick_smp_freq, num_class, id_chosen_chs)
        X_train_all[s], y_train_all[s] = X_train, y_train
        X_test_all[s], y_test_all[s] = X_test, y_test

    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Carry out subject-dependent setting with 5-fold cross validation
    for person, (X_tr, y_tr, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        if len(X_tr.shape) != 3:
            raise Exception('Dimension Error, must have 3 dimension')

        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr, y_tr)):
            print('FOLD:', fold + 1, 'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))
            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            X_te_fbcsp1List = []
            X_tr_fbcsp1List = []
            X_val_fbcsp1List = []
            # freq_ = int(X_tr_cv.shape[2] / pick_smp_freq)
            # for start in range(0, freq_):
            #     step_length = 100
            #     X_te_fbcsp1, X_tr_fbcsp1, X_val_fbcsp1 = getCSP(X_te[:, :, start * step_length:(start + 1) * step_length],
            #                                                     X_tr_cv[:, :, start * step_length:(start + 1) * step_length],
            #                                                     X_val_cv[:, :, start * step_length:(start + 1) * step_length], bands,
            #                                                     n_components, n_features,
            #                                                     num_class, order, pick_smp_freq, y_tr_cv)
            #
            #     X_te_fbcsp1List.append(X_te_fbcsp1)
            #     X_tr_fbcsp1List.append(X_tr_fbcsp1)
            #     X_val_fbcsp1List.append(X_val_fbcsp1)

            # X_te_fbcsp, X_tr_fbcsp, X_val_fbcsp = getCSP(X_te, X_tr_cv, X_val_cv, bands, n_components, n_features,
            #                                              num_class, order, pick_smp_freq, y_tr_cv)
            #
            # X_te_fbcsp1List.append(X_te_fbcsp)
            # X_tr_fbcsp1List.append(X_tr_fbcsp)
            # X_val_fbcsp1List.append(X_val_fbcsp)
            #
            # X_tr_fbcsp = np.hstack(X_tr_fbcsp1List)
            # X_val_fbcsp = np.hstack(X_val_fbcsp1List)
            # X_te_fbcsp = np.hstack(X_te_fbcsp1List)

            X_te_fbcsp, X_tr_fbcsp, X_val_fbcsp = getFBCSPFeaturs(X_te, X_tr_cv, X_val_cv, bands, n_components,
                                                                  n_features, num_class, order, pick_smp_freq, y_tr_cv)

            print('Check dimension of training data {}, val data {} and testing data {}'.format(X_tr_fbcsp.shape,
                                                                                                X_val_fbcsp.shape,
                                                                                                X_te_fbcsp.shape))

            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person + 1, fold + 1)
            __save_data_with_valset(save_path, SAVE_NAME, X_tr_fbcsp, y_tr_cv, X_val_fbcsp, y_val_cv, X_te_fbcsp, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person + 1, fold + 1))

# FBCSPF特征提取
def getFBCSPFeaturs(X_te, X_tr_cv, X_val_cv, bands, n_components, n_features, num_class, order, pick_smp_freq, y_tr_cv):

    # 在此处挂图
    two_feature_abnormal_detection(X_te[0][0])




    X_te_fbcsp1, X_tr_fbcsp1, X_val_fbcsp1 = getCSP(X_te[:, :, 0:200], X_tr_cv[:, :, 0:200],
                                                    X_val_cv[:, :, 0:200], bands, n_components, n_features,
                                                    num_class, order, pick_smp_freq, y_tr_cv)
    X_te_fbcsp2, X_tr_fbcsp2, X_val_fbcsp2 = getCSP(X_te[:, :, 200:410], X_tr_cv[:, :, 200:410],
                                                    X_val_cv[:, :, 200:410], bands, n_components, n_features,
                                                    num_class, order, pick_smp_freq, y_tr_cv)
    X_te_fbcsp3, X_tr_fbcsp3, X_val_fbcsp3 = getCSP(X_te[:, :, 100:301], X_tr_cv[:, :, 100:301],
                                                    X_val_cv[:, :, 100:301], bands, n_components, n_features,
                                                    num_class, order, pick_smp_freq, y_tr_cv)
    X_te_fbcsp, X_tr_fbcsp, X_val_fbcsp = getCSP(X_te, X_tr_cv, X_val_cv, bands, n_components, n_features,
                                                 num_class, order, pick_smp_freq, y_tr_cv)

    X_tr_fbcsp = np.hstack((X_tr_fbcsp, X_tr_fbcsp1, X_tr_fbcsp2, X_tr_fbcsp3))
    X_val_fbcsp = np.hstack((X_val_fbcsp, X_val_fbcsp1, X_val_fbcsp2, X_val_fbcsp3))
    X_te_fbcsp = np.hstack((X_te_fbcsp, X_te_fbcsp1, X_te_fbcsp2, X_te_fbcsp3))


    #在此测试
    X_tr_fbcsp_max_list1 = extrema_max_list(X_te)
    X_tr_fbcsp_max_list = [[item for sublist in subsublist for item in sublist] for subsublist in  X_tr_fbcsp_max_list1]

    # 沿第二个维度进行拼接
    X_te_fbcsp = np.concatenate((X_te_fbcsp,X_tr_fbcsp_max_list), axis=1)


    return X_te_fbcsp, X_tr_fbcsp, X_val_fbcsp


def getCSP(X_te, X_tr_cv, X_val_cv, bands, n_components, n_features, num_class, order, pick_smp_freq, y_tr_cv):
    # Peforming FBCSP feature extraction
    fbcsp_scaler = FBCSP(bands=bands, smp_freq=pick_smp_freq, num_class=num_class, order=order,
                         n_components=n_components, n_features=n_features)
    # 特征拟合并提取
    X_tr_fbcsp = fbcsp_scaler.fit_transform(X_tr_cv, y_tr_cv)
    X_val_fbcsp = fbcsp_scaler.transform(X_val_cv)
    X_te_fbcsp = fbcsp_scaler.transform(X_te)

    return X_te_fbcsp, X_tr_fbcsp, X_val_fbcsp


def subject_independent_setting(k_folds, pick_smp_freq, n_components, n_features, bands, order, save_path, num_class=2,
                                sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path = save_path + '/BCIC2a/fbcsp/{}_class/subject_independent'.format(num_class)
    n_chs = len(sel_chs)
    n_trials = n_trials_per_class * num_class

    X_train_all, y_train_all = np.zeros((n_subjs, n_trials, n_chs, int(MI_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))
    X_test_all, y_test_all = np.zeros((n_subjs, n_trials, n_chs, int(MI_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))

    id_chosen_chs = raw.chanel_selection(sel_chs)
    for s in range(n_subjs):
        X_train, y_train, X_test, y_test = __load_BCIC2a(raw_path, s + 1, pick_smp_freq, num_class, id_chosen_chs)
        X_train_all[s], y_train_all[s] = X_train, y_train
        X_test_all[s], y_test_all[s] = X_test, y_test

    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Carry out subject-independent setting with 5-fold cross validation
    for person, (X_val, y_val, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        train_subj = [i for i in range(n_subjs)]
        train_subj = np.delete(train_subj, person)  # remove test subject

        # Generating fake data to used for k-fold cross-validation only
        fake_tr = np.zeros((len(train_subj), 2))
        fake_tr_la = np.zeros((len(train_subj)))

        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        for fold, (train_ind, val_ind) in enumerate(skf.split(fake_tr, fake_tr_la)):
            print('FOLD:', fold + 1, 'TRAIN:', len(train_ind), 'VALIDATION:', len(val_ind))
            train_index, val_index = train_subj[train_ind], train_subj[val_ind]
            X_train_cat = np.concatenate((X_train_all[train_index], X_test_all[train_index]), axis=0)
            X_val_cat = np.concatenate((X_train_all[val_index], X_test_all[val_index]), axis=0)
            y_train_cat = np.concatenate((y_train_all[train_index], y_test_all[train_index]), axis=0)
            y_val_cat = np.concatenate((y_train_all[val_index], y_test_all[val_index]), axis=0)

            X_train = X_train_cat.reshape(-1, X_train_cat.shape[2], X_train_cat.shape[3])
            y_train = y_train_cat.reshape(-1)
            X_val = X_val_cat.reshape(-1, X_val_cat.shape[2], X_val_cat.shape[3])
            y_val = y_val_cat.reshape(-1)
            X_test = X_te
            y_test = y_te

            # Peforming FBCSP feature extraction
            fbcsp_scaler = FBCSP(bands=bands, smp_freq=pick_smp_freq, num_class=num_class, order=order,
                                 n_components=n_components, n_features=n_features)
            X_train_fbcsp = fbcsp_scaler.fit_transform(X_train, y_train)
            X_val_fbcsp = fbcsp_scaler.transform(X_val)
            X_test_fbcsp = fbcsp_scaler.transform(X_test)
            print("Check dimension of training data {}, val data {} and testing data {}".format(X_train_fbcsp.shape,
                                                                                                X_val_fbcsp.shape,
                                                                                                X_test_fbcsp.shape))
            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person + 1, fold + 1)
            __save_data_with_valset(save_path, SAVE_NAME, X_train_fbcsp, y_train, X_val_fbcsp, y_val, X_test_fbcsp,
                                    y_test)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person + 1, fold + 1))


def __load_BCIC2a(PATH, subject, new_smp_freq, num_class, id_chosen_chs):
    start = CONSTANT['MI']['start']  # 2
    stop = CONSTANT['MI']['stop']  # 6
    X_train, y_tr, X_test, y_te = raw.load_crop_data(
        PATH=PATH, subject=subject, start=start, stop=stop, new_smp_freq=new_smp_freq, num_class=num_class,
        id_chosen_chs=id_chosen_chs)
    return X_train, y_tr, X_test, y_te


def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')


# 提取极值点特征并计算特征个数
def extrema_max_list(three_dimension_time_series_data):
    # num_samples = data.shape[0]
    # num_channels = data.shape[1]
    # max_count = 0
    # min_count = 0
    data_list = list(three_dimension_time_series_data)
    X_tr_fbcsp_max = []


    for i in range(len(data_list)):
        # X_tr_fbcsp_max.append([])  # 确保索引为i的元素存在且为空列表
        X_tr_fbcsp_max_single = []
        for j in range(len(data_list[i])):
            single_channel_data = data_list[i][j]
            float_data_array = np.array(single_channel_data)

            # 寻找极值点（极大值和极小值）
            extrema_max_points = []
            extrema_min_points = []
            for k in range(1, len(float_data_array) - 1):
                if float_data_array[k] > float_data_array[k - 1] and float_data_array[k] > float_data_array[k + 1]:
                    extrema_max_points.append((k, float_data_array[k], 'max'))
                elif float_data_array[k] < float_data_array[k - 1] and float_data_array[k] < float_data_array[k + 1]:
                    extrema_min_points.append((k, float_data_array[k], 'min'))

            # 将极值点按照时间顺序排序
            extrema_max_points.sort(key=lambda x: x[0])
            extrema_min_points.sort(key=lambda x: x[0])
            # 将极大值点按照极大值大小排序
            extrema_max_points.sort(key=lambda x: x[1], reverse=True)
            # 获取前十个值最大的数据点并存储到新列表中
            top_10_max_points = extrema_max_points[:10]
            top_10_max_values1 = [point[1] for point in top_10_max_points]
            top_10_max_values = list(top_10_max_values1)
            X_tr_fbcsp_max_single.append(top_10_max_values)

        X_tr_fbcsp_max.insert(i,X_tr_fbcsp_max_single)

    return X_tr_fbcsp_max


def two_feature_abnormal_detection(one_dimension_time_series):

    # 传入一维时间序列数据（自闭症脑电数据）
    data_list=one_dimension_time_series
    # data_list = one_dimension_time_series.split()  # 使用split()方法按空格分割字符串
    float_data_list = [float(num) for num in data_list]  # 将每个子字符串转换为浮点数
    # float_data_array是一维时间序列数据
    float_data_array = np.array(float_data_list)

    # 提取极值点（极大值和极小值）
    extrema_points = []
    for i in range(1, len(float_data_array) - 1):
        if float_data_array[i] > float_data_array[i - 1] and float_data_array[i] > float_data_array[i + 1]:
            extrema_points.append((i, float_data_array[i], 'max'))
        elif float_data_array[i] < float_data_array[i - 1] and float_data_array[i] < float_data_array[i + 1]:
            extrema_points.append((i, float_data_array[i], 'min'))

    # 将极值点按照时间顺序排序
    extrema_points.sort(key=lambda x: x[0])

    # 附加点
    # 创建一个新列表用于存储修改后的数据点
    modified_extrema_points = []
    for i in range(len(extrema_points)):
        # 将当前数据点添加到新列表中
        modified_extrema_points.append(extrema_points[i])

        if i < len(extrema_points) - 1:  # 确保不是最后一个数据点
            x1, y1, _ = extrema_points[i]
            x2, y2, _ = extrema_points[i + 1]


            if x2 - x1 >= 3:  # 检查前后位置信息之差是超过2
                # 计算新点的位置和数值
                x_new = (x1 + x2) / 2
                y_new = y1 + ((y2 - y1) / (x2 - x1)) * (x_new - x1)
                new_point = (x_new, y_new, 'add')  # 新点类型设为 "add"
                # 将新点添加到列表中
                modified_extrema_points.append(new_point)

    # # 打印修改后的列表
    # print(modified_extrema_points)




    # 构建分段线性表示
    # x = np.arange(len(float_data_array))
    # y = np.interp(x, [p[0] for p in extrema_points], [p[1] for p in extrema_points])

    # 构建分段线性表示
    x = np.arange(len(float_data_array))
    y = np.interp(x, [p[0] for p in modified_extrema_points], [p[1] for p in modified_extrema_points])

    # 使用LOF检测异常
    # lof = LocalOutlierFactor()
    # outliers = lof.fit_predict(y.reshape(-1, 1))

    # 解决中文乱码
    matplotlib.rc("font", family='Microsoft YaHei')

    # 折线图可视化展示
    plt.figure(figsize=(14, 7))
    # plt.plot(float_data_array, label='原始数据', color='black')
    plt.plot(y, label='基于重要点（极值点与附加点）的分段线性表示', color='c')

    # 创建散点图
    max_points = plt.scatter([], [], color='b', label='极大值点')
    min_points = plt.scatter([], [], color='yellow', label='极小值点')
    add_points = plt.scatter([], [], color='red', label='附加点')

    # max_points = plt.scatter([], [], color='b', label='极大值点', marker='o', edgecolors='b', facecolors='none')  # 空心圆
    # min_points = plt.scatter([], [], color='yellow', label='极小值点', marker='o', edgecolors='yellow',
    #                          facecolors='none')  # 空心圆
    # add_points = plt.scatter([], [], color='black', label='附加点', marker='o', edgecolors='black',
    #                          facecolors='none')  # 空心圆
    # 添加附加点的图例
    max1=[]
    min1=[]
    for point in modified_extrema_points:
        if point[2] == 'max':
            max_points = plt.scatter(point[0], point[1], color='b')
            # max_points = plt.scatter(point[0], point[1], color='b', marker='o',facecolors='none')  匹配论文空心圆
            max1.append(point)

        elif point[2] == 'min':
            min_points = plt.scatter(point[0], point[1], color='yellow')
            min1.append(point)

        elif point[2] == 'add':  # 绘制附加点为红色
            plt.scatter(point[0], point[1], color='red')
    # plt.scatter(np.where(outliers == -1), y[outliers == -1], color='g', label='LOF检测出的异常点')

    # 添加X轴和Y轴标签
    plt.xlabel('数据点 (个)',fontsize=20)
    plt.ylabel('值 (μV)',fontsize=20)

    # plt.legend()
    plt.legend(prop={'size': 14})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    print("成功")