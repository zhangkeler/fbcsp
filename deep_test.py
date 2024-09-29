import math
import sys
import io
import torch
from keras.src.callbacks import ModelCheckpoint

from arl_eegmodels import EEGModels
from dtaidistance.connectors import sktime
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils import np_utils
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.deep_learning import CNNClassifier, MVTSTransformerClassifier, LSTMFCNClassifier
from WLOF import WLOF
from arl_eegmodels.EEGModels import EEGNet, DeepConvNet, ShallowConvNet
from load_data import load_EGG_10
import sys
import io
# from ConvTran.utils import Setup, Initialization, dataset_class, Data_Verifier
# from ConvTran.Models.model import model_factory, count_parameters
# from ConvTran.Models.optimizers import get_optimizer
# from ConvTran.Models.loss import get_loss_module
# from ConvTran.Models.utils import load_model
# from ConvTran.Training import SupervisedTrainer, train_runner
import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *

# logger = logging.getLogger('__main__')
# parser = argparse.ArgumentParser()
# # -------------------------------------------- Input and Output --------------------------------------------------------
# parser.add_argument('--output_dir', default='Results',
#                     help='Root output directory. Must exist. Time-stamped directories will be created inside.')
# parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
# parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
# parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# # ----------------------------------------------------------------------------------------------------------------------
# # ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
# parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
#                                                                               "Transformers (T)")
# # Transformers Parameters ------------------------------
# parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
# parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
# parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
#                     help='Fix Position Embedding')
# parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
#                     help='Relative Position Embedding')
# # Training Parameters/ Hyper-Parameters ----------------
# parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
# parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
# parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
# parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
#                     help='Metric used for defining best epoch')
# # ----------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------ System --------------------------------------------------------
# parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
# parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
# parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# args = parser.parse_args()

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

    ####ConvTran
    # config = Setup(args)  # configuration dictionary
    # device = Initialization(config)
    # Data_Verifier(config)
    # config['data_dir'] = os.path.join(config['data_path'], problem)
    # print(text2art(problem, font='small'))
    # # ------------------------------------ Load Data ---------------------------------------------------------------
    # logger.info("Loading Data ...")
    # Data = Data_Loader(config)
    # train_dataset = dataset_class(Data['train_data'], Data['train_label'])
    # val_dataset = dataset_class(Data['val_data'], Data['val_label'])
    # test_dataset = dataset_class(Data['test_data'], Data['test_label'])
    #
    # train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # # --------------------------------------------------------------------------------------------------------------
    # # -------------------------------------------- Build Model -----------------------------------------------------
    # dic_position_results = [config['data_dir'].split('/')[-1]]
    #
    # logger.info("Creating model ...")
    # config['Data_shape'] = Data['train_data'].shape
    # config['num_labels'] = int(max(Data['train_label'])) + 1
    # model = model_factory(config)
    # logger.info("Model:\n{}".format(model))
    # logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # # -------------------------------------------- Model Initialization ------------------------------------
    # optim_class = get_optimizer("RAdam")
    # config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    # config['loss_module'] = get_loss_module()
    # save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
    # tensorboard_writer = SummaryWriter('summary')
    # model.to(device)
    # # ---------------------------------------------- Training The Model ------------------------------------
    # logger.info('Starting training...')
    # trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
    #                             print_interval=config['print_interval'], console=config['console'],
    #                             print_conf_mat=False)
    # val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
    #                                   print_interval=config['print_interval'], console=config['console'],
    #                                   print_conf_mat=False)
    #
    # train_runner(config, model, trainer, val_evaluator, save_path)
    # best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    # best_model.to(device)
    #
    # best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
    #                                         print_interval=config['print_interval'], console=config['console'],
    #                                         print_conf_mat=True)
    # best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    # print_str = 'Best Model Test Summary: '
    # for k, v in best_aggr_metrics_test.items():
    #     print_str += '{}: {} | '.format(k, v)
    # print(print_str)
    # dic_position_results.append(all_metrics['total_accuracy'])

    for fold, (train_index, test_index) in enumerate(skfolds.split(pca_data, lab)):
        x_train, x_test = pca_data[train_index], pca_data[test_index]
        y_train, y_test = lab[train_index], lab[test_index]
        y_train = np.array(y_train)
        # print(y_train.shape)
        x_train = np.array(x_train)
        samples=x_train.shape[1]
        x_test = np.array(x_test)
        print(x_train.shape)
        print(x_test.shape)
        # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        # x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        # x_train=np.array(x_train.T)
        # x_test=np.array(x_test.T)
        x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1],1)
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1],1)
        y_train = np_utils.to_categorical(y_train)
        # y_test=np_utils.to_categorical(y_test)
        print(x_train.shape,y_train.shape)
        #EEGNet
        # model = EEGNet(nb_classes=10, Chans=1, Samples=samples,
        #                dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
        #                dropoutType='Dropout')
        model = DeepConvNet(nb_classes=10, Chans=1, Samples=samples,
                    dropoutRate=0.5)
        # model = ShallowConvNet(nb_classes=10, Chans = 1, Samples = samples, dropoutRate = 0.5)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        # count number of parameters in the model
        numParams = model.count_params()

        # set a valid path for your system to record model checkpoints
        # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
        #                                save_best_only=True)
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1}
        fittedModel = model.fit(x_train, y_train, batch_size=16, epochs=300,
                                verbose=2, class_weight=class_weights)
        # model.load_weights('/tmp/checkpoint.h5')
        probs = model.predict(x_test)
        print("probs")
        print(probs.shape)
        print(x_test.shape)
        label1 = probs.argmax(axis=-1)
        # KNN model
        # m = KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')
        # m.fit(x_train, y_train)
        # label1 = m.predict(x_test)
        #dtw+knn
        # knn_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
        # knn_dtw.fit(x_train, y_train)
        # label1 = knn_dtw.predict(x_test)
        #cnn
        # clf = CNNClassifier()
        # # sktime.datatypes.check_raise(data, 'numpy3D')
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #cntc
        # clf =CNTCClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #fcn
        # clf = FCNClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #inceptiontime
        # clf = InceptionTimeClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #lstmfcn
        # clf = LSTMFCNClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #macnn
        # clf = MACNNClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #mcdcnn
        # clf = MCDCNNClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #mlp
        # clf = MLPClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #mvts_transformer
        # clf = MVTSTransformerClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #resnet
        # clf = ResNetClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #rnn
        # clf = SimpleRNNClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #tapnet
        # clf = TapNetClassifier()
        # clf.fit(x_train, y_train)
        # label1 = clf.predict(x_test)
        #convtran
        # config = Setup(args)  # configuration dictionary
        # device = Initialization(config)


        accuracy = accuracy_score(y_test, label1)
        print(y_test)
        print(label1)
        result.append(accuracy)

    res = np.average(result)
    # print(res)
    # np.savetxt('deep_res.txt',res)
    return res

if __name__ == '__main__':
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # all_extracted_speech_data=np.loadtxt('data_EEG/data_EEG.txt')
    # all_extracted_speech_data = all_extracted_speech_data.reshape(2000,58,1024)
    # labels=np.loadtxt('data_EEG/label_EEG.txt')
    all_extracted_speech_data,labels = load_EGG_10()
    print(all_extracted_speech_data.shape)
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
    # scaler = StandardScaler()
    # scalered_data = scaler.fit_transform(all_extracted_speech_data)  # 标准化
    # all_extracted_speech_data = PCA(n_components=200).fit_transform(scalered_data)  # PCA降维

    beta = 1 / 2
    # #param1 = [5, 10, 15, 20]
    # param1 =[1,3,5]
    k=1
    # param1 = [1,2,3,4,5]
    # # k = 10
    # param2 = [0.1, 0.3, 0.5, 0.7, 0.9]
    s=0.1
    # #param2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # # s = 0.3
    # param3 = [all_extracted_speech_data.shape[1]] #时间序列长
    g=[all_extracted_speech_data.shape[1]]
    # print("all_extracted!!!!!shape"+str(all_extracted_speech_data.shape))
    # #param3 = [40]
    # # g = 60
    w = math.floor(0.1 * all_extracted_speech_data.shape[1])
    #
    with open("res.txt", "w") as file:
        file.write("result of pre-processed data in different params with important points\n")

    data = []
    # data=np.zeros([all_extracted_speech_data.shape[0],4])
    important_points_save=[]
    features_save=[]
    # for i in range(all_extracted_speech_data.shape[0]):
    #     D = all_extracted_speech_data[i, :].T
    #     size = D.shape[0]
    #     index_list = list(range(size))
    #     D2 = np.column_stack((index_list, D))
    #     print("i"+str(i))
    #
    #     extrema_points, important_points, features, outfactor = WLOF(D2, w, g, s, beta, k)
                    #plot(D,D2,i,k,s,extrema_points,important_points,outfactor)
        # features = np.array(features)
        # features = features.flatten()
        # features = np.array([item for sublist in features for item in sublist])
        # important_points =np.array(important_points).T
        #             # important_points=important_points[7:g-8,1].T
        # important_now = np.zeros((1,g))
        # important_now[0,:important_points.shape[1]]=important_points[1,:]
        # important_now[0,important_points.shape[1]:] = important_points[1,-1]
        # important_points = important_now.astype(float)
        #             # print(features.shape)
        # print(important_points.shape)
        # important_points = important_points.reshape((important_points.shape[1]))
        # print(important_points.shape)
                    # important_points_save.append(important_points)
        # print(features.shape)
        # features=np.array(features)
                    # features = features.reshape(1,-1)
        # print(features.shape)
                    # features_save.append(features)
                    # features =features + important_points
        # features = np.concatenate((features,important_points))
                    #print(features)
                    #print(features.shape)
                    # data.append(important_points)
        # data.append(important_points)
    #     data[i,:]=features
    #     print("i"+str(i))
    #     print(len(data))
    data = all_extracted_speech_data
    data=np.array(data)
    res = test(data,labels)
    # print(data.shape)
    # print(labels.shape)
    # res = test(data, labels)
    print(res)
    # with open("res.txt", "a") as file:
    #     file.write(f"k={k},s={s},g={g},res={res}\n")

