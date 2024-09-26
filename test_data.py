import os
import numpy as np
import glob
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt, gridspec
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from scipy import signal
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import copy
import pandas as pd
import math
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
import torch
import torch.nn.functional as F
from show_data import reduce_by_max_abs,count_non_negative_ones
from WLOF import compute_feature,extrema_max_list,important_point,WLOF
from load_data import load_laryngea_520_10, load_laryngea_250_100, load_EGG
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def test(data,labels):
    nclass = len(np.unique(labels))
    skfolds = StratifiedKFold(n_splits=10)

    result = []
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.tensor(data).float()
    lab = torch.tensor(labels).view(-1).to(torch.long)

    # scaler = StandardScaler()
    # scalered_data = scaler.fit_transform(data)  # 标准化
    # pca_data = PCA(n_components=200).fit_transform(scalered_data)  # PCA降维
    pca_data = data
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
        # m = KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')
        # m.fit(x_train, y_train)
        # label1 = m.predict(x_test)

        #dtw+knn
        knn_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
        knn_dtw.fit(x_train, y_train)
        label1 = knn_dtw.predict(x_test)

        accuracy = accuracy_score(y_test, label1)
        result.append(accuracy)
    res = np.average(result)
    print(res)
    return res

def plot(D,data2,i,k,s,extrema_points, important_points, outfactor):

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 2)
    # ax1, ax2, ax3, ax4 = axes.flatten()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])
    # Plot A on the first subplot
    ax1.plot(D, color='blue')
    ax2.plot(data2[:, 1])
    extrema_points_y = [item[1] for item in extrema_points]
    extrema_points_x = [item[0] for item in extrema_points]
    important_points_y = [item[1] for item in important_points]
    important_points_x = [item[0] for item in important_points]
    print("extrema_points")
    print(extrema_points_y)
    ax3.plot(extrema_points_x, extrema_points_y, '-o', color='red', zorder=1)
    ax3.scatter(important_points_x, important_points_y, marker='^', s=100, zorder=2)
    ax4.plot(outfactor, color='yellow')
    # plt.plot(outfactor)
    plt.grid(True)
    #plt.show()
    dir = ""
    plt.savefig(f"png/ind_{i}_k_{k}_s_{s}.png")
    plt.close()

if __name__ == '__main__':
    #改成你们自己的文件路径
    root_data = 'mytest'
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

    all_extracted_speech_data=np.loadtxt('data_EEG/data_EEG.txt')
    # all_extracted_speech_data = all_extracted_speech_data.reshape(2000,58,1024)
    labels=np.loadtxt('data_EEG/label_EEG.txt')
    # print(all_extracted_speech_data.shape)
    # print(labels.shape)

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
        # if i<100:
        #     plt.plot(downsampled_data)
        #     plt.savefig(f"data_EEG/4/downsampled_data/i_{i}.png")
        #     plt.close()
    all_extracted_speech_data = np.array(all_extracted_speech_data_my_pre)
    scaler = StandardScaler()
    scalered_data = scaler.fit_transform(all_extracted_speech_data)  # 标准化
    all_extracted_speech_data = PCA(n_components=200).fit_transform(scalered_data)  # PCA降维

    beta = 1 / 2
    # #param1 = [5, 10, 15, 20]
    param1 =[1,3,5]
    # param1 = [1,2,3,4,5]
    # # k = 10
    param2 = [0.1, 0.3, 0.5, 0.7, 0.9]
    # #param2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # # s = 0.3
    param3 = [all_extracted_speech_data.shape[1]] #时间序列长
    # print("all_extracted!!!!!shape"+str(all_extracted_speech_data.shape))
    # #param3 = [40]
    # # g = 60
    w = math.floor(0.1 * all_extracted_speech_data.shape[1])
    #
    with open("res.txt", "w") as file:
        file.write("result of pre-processed data in different params with important points\n")
    # for k in param1:
    #     for s in param2:
    #         for g in param3:
    #             data = []
    #             # data=np.zeros([all_extracted_speech_data.shape[0],4])
    #             important_points_save=[]
    #             features_save=[]
    #             for i in range(all_extracted_speech_data.shape[0]):
    #                 D = all_extracted_speech_data[i, :].T
    #                 size = D.shape[0]
    #                 index_list = list(range(size))
    #                 D2 = np.column_stack((index_list, D))
    #                 print("i"+str(i))
    #
    #                 extrema_points, important_points, features, outfactor = WLOF(D2, w, g, s, beta, k)
    #                 #plot(D,D2,i,k,s,extrema_points,important_points,outfactor)
    #     # features = np.array(features)
    #     # features = features.flatten()
    #                 features = np.array([item for sublist in features for item in sublist])
    #                 important_points =np.array(important_points).T
    #                 # important_points=important_points[7:g-8,1].T
    #                 important_now = np.zeros((1,g))
    #                 important_now[0,:important_points.shape[1]]=important_points[1,:]
    #                 important_now[0,important_points.shape[1]:] = important_points[1,-1]
    #                 important_points = important_now.astype(float)
    #                 # print(features.shape)
    #                 print(important_points.shape)
    #                 important_points = important_points.reshape((important_points.shape[1]))
    #                 print(important_points.shape)
    #                 important_points_save.append(important_points)
    #                 print(features.shape)
    #                 features=np.array(features)
    #                 # features = features.reshape(1,-1)
    #                 print(features.shape)
    #                 features_save.append(features)
    #                 # features =features + important_points
    #                 # features = np.concatenate((features,important_points))
    #                 #print(features)
    #                 #print(features.shape)
    #                 data.append(important_points)
        # data[i,:]=features
        # print("i"+str(i))
        # print(len(data))
    data = all_extracted_speech_data
    #             data=np.array(data)
                # print(data.shape)
                # data.reshape((1997,50))
                # print(data.shape)
                # print(all_extracted_speech_data.shape)
                # np.savetxt(f'important_points_k_{k}_s_{s}.txt', important_points_save)
                # np.savetxt(f'feature_k_{k}_s_{s}.txt', features_save)
                # data = np.column_stack((all_extracted_speech_data,data))
                # print(f'{data.shape}_____{labels.shape}')
    res = test(data,labels)
    print(res)
                # with open("res.txt", "a") as file:
                #     file.write(f"k={k},s={s},g={g},res={res}\n")

