import os
import numpy as np
import glob
from collections import Counter
import copy
import pandas as pd
from scipy.interpolate import interp1d
import json
from show_data import reduce_by_max_abs,count_non_negative_ones
from mne.io import concatenate_raws,read_raw_edf
import matplotlib.pyplot as plt
import mne

def load_laryngea_520_10():
    # 指定EDF文件夹的路径
    # edf_folder = '../' + root_data + '_data/喉部-520hz/'
    edf_folder = '喉部-520hz/'

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
        print("data_np_1"+str(data_np_1.shape))

    data_np = np.vstack(all_data)
    print("data_np" + str(data_np.shape))
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
        #print(word_data.shape)
        if word_data.shape[0] < 300 and word_data.shape[0] > 310:
            continue


        word_data = word_data[:300]
        #print(word_data.shape)
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

    print("all_extracted_speech_data")
    print(all_extracted_speech_data.shape[1])
    print("labels")
    print(labels.shape)

    return all_extracted_speech_data,labels

def load_laryngea_250_100():
    # 指定EDF文件夹的路径
    # edf_folder = '../' + root_data + '_data/喉部-520hz/'
    edf_folder = '喉部-250hz/'

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
    print("data_np"+str(data_np.shape))
    # 假设这是你3秒的音频数据，采样率250Hz，总样本数=3秒*250Hz
    sfreq_speech = 250
    duration = 3  # seconds
    samples = duration * sfreq_speech

    # 删除前n秒数据
    begin_cut = 0 * sfreq_speech
    # 一个单词n次
    end_cut = begin_cut + 1 * samples
    # 截取数据(全部)
    data_np = data_np[:, begin_cut:end_cut]

    sfreq_speech = 100
    all_extracted_speech_data = []
    all_speech_label = []
    all_range_start = []

    # 每个单词有n组
    per_num = 20
    # 数据每100个一组（每个3s[一个单词1次]）
    error_num =0
    error_num2 = 0
    for i in range(data_np.shape[0]): #每个样本 (2000,751)
        ave = 100
        while abs(ave) > 1e-10:
            ave = np.average(data_np[i])
            data_np[i] = data_np[i] - ave
        word_data = reduce_by_max_abs(data_np[i])
        print("word_data"+str(word_data.shape))
        if word_data.shape[0] < 300 and word_data.shape[0] > 310:
            continue
        if word_data.shape[0] < 300:
            ind1 = int(np.round((300-word_data.shape[0])/2))
            ind2 = 300 - word_data.shape[0] - ind1
            array_left = np.full(ind1, word_data[0])
            print(array_left.shape)
            array_right = np.full(ind2, word_data[-1])
            print(array_right.shape)
            word_data = np.concatenate([array_left,word_data,array_right])
        print("word_data" + str(word_data.shape))
        word_data = word_data[:300]
        # 每隔3秒提取一次
        segments = np.array([word_data[i:i + samples] for i in range(0, len(word_data), samples)])
        print("segments"+str(segments.shape))
        for j in range(segments.shape[0]):
            data = segments[j]

            # 使用短时能量来检测说话部分
            window_size = int(0.1 * sfreq_speech)  # 100ms的滑动窗口
            ste = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='valid')

            # 找到能量最大的一段，假设这段是说话部分
            max_index = np.argmax(ste)
            # print('max_index:')
            # print(max_index)
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
            #print("speaking_duration"+str(speaking_duration))

            # 需要补充的数据长度
            # needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
            needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
            #print("needed_duration"+str(needed_duration))
            remaining_duration = needed_duration - speaking_duration
            #print("remaining_duration"+str(remaining_duration))

            if remaining_duration < 0:
                print(f'说话长度太长_阈值小了_{i}_{j}')
                all_range_start.append(-1)

                print(f'默认处理_{i}_{j}，取0.5-2s')

                left = int(0.5 * sfreq_speech)
                right = int(2 * sfreq_speech)
                extracted_data = data[left:right]
                extracted_data = [float(i) for i in extracted_data]
                if len(extracted_data) !=150:
                    error_num2 +=1
                all_extracted_speech_data.append(extracted_data)
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
            if right - left != 150:  # debug用
                print("error")
            if right > data.shape[0]: #data.shape
                if data.shape[0] >= 300:
                    left = int(300 - 1.5 * 100)
                    right = 300
                else :
                    left =  data.shape[0] -150
                    right = data.shape[0]


            # 提取最终的1.5秒数据
            extracted_data = data[left:right]

            print(f'完成_{i}_{j}')
            # print("extracted_data")
            # print(extracted_data.shape)
            #print(extracted_data)
            extracted_data = [float(i) for i in extracted_data]
            if len(extracted_data) != 150:
                error_num += 1
                print(data.shape)
                print("samples"+str(samples))
                print("left"+str(left)+"right"+str(right))
                print("extracted_data"+str(extracted_data))
                print("len_extracted"+str(len(extracted_data)))
            all_extracted_speech_data.append(extracted_data)
            all_speech_label.append(i // per_num)
            all_range_start.append(left)
    print("error"+str(error_num)+str(error_num2))
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

        labels = np.vectorize(label_mapping.get)(labels)

    print("all_extracted_speech_data")
    print(all_extracted_speech_data.shape)
    print("labels")
    print(labels.shape)

    return all_extracted_speech_data,labels

def load_EGG():
    # 指定EDF文件夹的路径
    # edf_folder = '../' + root_data + '_data/喉部-520hz/'
    edf_folder = 'EEG-256hz/'

    # 获取所有以 .edf 结尾的文件路径，并按文件名排序
    edf_files = glob.glob(os.path.join(edf_folder, '*.edf'))
    i = 0
    all_data = []
    num = 0
    for file_path in edf_files:
        if not file_path.endswith(".md.edf"):
            df = read_raw_edf(file_path, preload=True)
            data = df.get_data()
            data = data.T
        # sheet_name = 'Sheet1'
        # # 读取数据到 DataFrame，跳过前两行
        # df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
        # # 选择偶数列（假设偶数列是指列索引从 1 开始的偶数列）
            even_columns = [x for x in range(1, data.shape[1]) if x % 2 == 1]
            data = data[:,even_columns]
            # data = data.T
            # data= data.reshape(-1, 1)
            # # 将数据转换为 NumPy 数组
            data_np_1 = data
            data_np_1 = np.transpose(data_np_1, (1, 0))
            print("data_np_1"+str(data_np_1.shape))
            if data_np_1.shape[1] < 1024:
                x_new = np.linspace(0, data_np_1.shape[1]-1, 150)
                # 使用线性插值生成新的 300 个点
                f = interp1d(np.arange(data_np_1.shape[1]), data_np_1, kind='linear')
                data_np_1 = f(x_new)
                data_np_1=np.array(data_np_1)
            # elif len(extracted_data) >150:
            #     extracted_data = extracted_data[:150]
            elif data_np_1.shape[1] > 1024:
                data_np_1=data_np_1[:,:1024]
            print("data_np_1"+str(data_np_1.shape))
            if data_np_1.shape[1]!=1024:
                num+=1
            data_np_1 = data_np_1.reshape((data_np_1.shape[0]*data_np_1.shape[1],))
            all_data.append(data_np_1)

    print("num"+str(num))
    # data_np = np.vstack(all_data)
    data_np=np.array(all_data)
    # print("data_np" + str(data_np.shape))
    # np.savetxt("data_np.txt", data_np)
    # 假设这是你3秒的音频数据，采样率250Hz，总样本数=3秒*250Hz
    # sfreq_speech = 256
    # duration = 4  # seconds
    # samples = duration * sfreq_speech
    #
    # # 删除前n秒数据
    # begin_cut = 0 * sfreq_speech
    # # 一个单词n次
    # end_cut = begin_cut + 1 * samples
    # # 截取数据
    # data_np = data_np[:, begin_cut:end_cut]
    #
    # sfreq_speech = 100
    all_extracted_speech_data = []
    all_speech_label = []
    all_range_start = []

    # 每个单词有n组
    per_num = 20 #或乘20
    error_num = 0
    error_num2 = 0
    length=[]
    # 数据每100个一组（每个3s[一个单词1次]）
    data1=[]
    data2=[]
    data3=[]
    for i in range(data_np.shape[0]):
        # if i<100:
        #     plt.plot(data_np[i])
        #     plt.savefig(f"data_EEG/3/data_np/ind_{i}.png")
        #     plt.close()
        # print(data_np[i].shape)
        ave = 10
        while abs(ave) > 1e-10:
            ave = np.average(data_np[i])
            data_np[i] = data_np[i] - ave
        word_data = reduce_by_max_abs(data_np[i])
        # print(word_data.shape)
        if word_data.shape[0] < 300 and word_data.shape[0] > 310:
            continue

        # word_data = word_data[:300]
        # if i<100:
        #     plt.plot(word_data)
        #     plt.savefig(f"data_EEG/3/word_data/ind_{i}.png")
        #     plt.close()
        # # print(word_data.shape)
        # # 每隔3秒提取一次
        # segments = np.array([word_data[i:i + samples] for i in range(0, len(word_data), samples)])
        # for j in range(segments.shape[0]):
        #     data = segments[j]
        #
        #     # 使用短时能量来检测说话部分
        #     window_size = int(0.1 * sfreq_speech)  # 100ms的滑动窗口
        #     ste = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='valid')
        #
        #     # 找到能量最大的一段，假设这段是说话部分
        #     max_index = np.argmax(ste)
        #     # print('max_index:')
        #     # print(max_index)
        #     ste_threshold = np.mean(ste) + np.std(ste)  # 设置一个阈值
        #     speaking_indices = np.where(ste > ste_threshold)[0]
        #
        #     # 数据找不到说话部位
        #     if len(speaking_indices) == 0:
        #         #print(f'数据找不到说话部位_{i}_{j}')
        #         all_range_start.append(-1)
        #
        #         continue
        #     # 确定说话部分的开始和结束(在信号连续的情况下，使用这种。不连续就还得改进)
        #     start_speaking = speaking_indices[0]
        #     end_speaking = speaking_indices[-1]
        #
        #     # 计算说话部分长度
        #     speaking_duration = end_speaking - start_speaking
        #     # print("speaking_duration"+str(speaking_duration))
        #
        #     # 需要补充的数据长度
        #     # needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
        #     needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
        #     # print("needed_duration"+str(needed_duration))
        #     remaining_duration = needed_duration - speaking_duration
        #     # print("remaining_duration"+str(remaining_duration))
        #
        #     if remaining_duration < 0:
        #         print(f'说话长度太长_阈值小了_{i}_{j}')
        #         all_range_start.append(-1)
        #
        #         print(f'默认处理_{i}_{j}，取0.5-2s')
        #
        #         left = int(0.5 * sfreq_speech)
        #         right = int(2 * sfreq_speech)
        #         extracted_data = data[left:right]
        #         extracted_data = [float(i) for i in extracted_data]
        #         extracted_data = np.array(extracted_data)
        #         data2.append(extracted_data)
        #         if i < 100:
        #             plt.plot(extracted_data)
        #             plt.savefig(f"data_EEG/3/data2/i_{i}_j_{j}.png")
        #             plt.close()
        #         # if extracted_data.shape[0] < 150:
        #         #     ind1 = int(np.round((150 - extracted_data.shape[0]) / 2))
        #         #     ind2 = 150 - extracted_data.shape[0] - ind1
        #         #     array_left = np.full(ind1, extracted_data[0])
        #         #     # print(array_left.shape)
        #         #     array_right = np.full(ind2, extracted_data[-1])
        #         #     # print(array_right.shape)
        #         #     extracted_data = np.concatenate([array_left, extracted_data, array_right])
        #         if extracted_data.shape[0] < 150:
        #             x_new = np.linspace(0, extracted_data.shape[0]-1, 150)
        #             # 使用线性插值生成新的 300 个点
        #             f = interp1d(np.arange(extracted_data.shape[0]), extracted_data, kind='linear')
        #             extracted_data = np.array(f(x_new))
        #         elif extracted_data.shape[0] >150:
        #             extracted_data = extracted_data[:150]
        #         if len(extracted_data) != 150:
        #             error_num2 += 1
        #         length.append(len(extracted_data))
        #         all_extracted_speech_data.append(extracted_data)
        #         all_speech_label.append(i // per_num)
        #         all_range_start.append(left)

                # continue
            # 左右补充的数据量
            # left_padding = right_padding = int(remaining_duration / 2)
            #
            # # 确保索引不超出边界，处理边界情况
            # if start_speaking - left_padding < 0:
            #     left = 0  # 向左延伸到边界
            #     right = 1.5 * sfreq_speech
            # elif end_speaking + right_padding > samples:
            #     left = samples - 1.5 * sfreq_speech
            #     right = samples
            # else:
            #     if remaining_duration % 2 == 1:  # 说明是奇数，之前被除以2的时候少了1，截断操作！！！
            #         left = start_speaking - left_padding
            #         right = end_speaking + right_padding + 1  # 默认从右边扣除
            #     else:
            #         left = start_speaking - left_padding
            #         right = end_speaking + right_padding
            #
            # left = int(left)
            # right = int(right)
            # if right - left != 150:  # debug用
            #     print("error")
            # if right > data.shape[0]:  # data.shape
            #     if data.shape[0] >= 300:
            #         left = int(300 - 1.5 * 100)
            #         right = 300
            #     else:
            #         left = data.shape[0] - 150
            #         right = data.shape[0]
            #
            # # 提取最终的1.5秒数据
            # extracted_data = data[left:right]
            # data3.append(extracted_data)
            # plt.plot(extracted_data)
            # plt.savefig(f"data_EEG/3/data3/i_{i}_j_{j}.png")
            # plt.close()

            # if extracted_data.shape[0] < 150:
            #     ind1 = int(np.round((150 - extracted_data.shape[0]) / 2))
            #     ind2 = 150 - extracted_data.shape[0] - ind1
            #     array_left = np.full(ind1, extracted_data[0])
            #     # print(array_left.shape)
            #     array_right = np.full(ind2, extracted_data[-1])
            #     # print(array_right.shape)
            #     extracted_data = np.concatenate([array_left, extracted_data, array_right])

            # if len(extracted_data) < 150:
            #     x_new = np.linspace(0, len(extracted_data)-1, 150)
            #     # 使用线性插值生成新的 300 个点
            #     f = interp1d(np.arange(len(extracted_data)), extracted_data, kind='linear')
            #     extracted_data = f(x_new)
            # elif len(extracted_data) >150:
            #     extracted_data = extracted_data[:150]
            # if len(extracted_data) != 150:
            #     error_num+=1
            # print(f'完成_{i}_{j}')
            # print("extracted_data" + str(len(extracted_data)))
            # length.append(len(extracted_data))
            # all_extracted_speech_data.append(extracted_data)
            # all_speech_label.append(i // per_num)
            # all_range_start.append(left)
        all_extracted_speech_data.append(data_np[i])
        all_speech_label.append(i // per_num)
    print("error" + str(error_num) + " "+str(error_num2))
    # 计算每个位置之前（包括该位置）非-1的个数
    # non_negative_ones_counts = count_non_negative_ones(all_range_start)
    #
    all_extracted_speech_data = np.array(all_extracted_speech_data)
    # all_extracted_eeg_data = all_extracted_speech_data
    # np.savetxt("data1.txt", all_extracted_speech_data)
    # np.savetxt("data2.txt", all_extracted_speech_data)
    # np.savetxt("data3.txt", all_extracted_speech_data)
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
        # all_extracted_eeg_data = all_extracted_eeg_data[mask]
        labels = labels[mask]

    label_stander = True

    if label_stander:
        # 重映射标签为从0开始连续的
        unique_labels = np.unique(labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print('label_mapping')
        print(label_mapping)
        labels = np.vectorize(label_mapping.get)(labels)

    # print("all_extracted_speech_data")
    # print(all_extracted_speech_data.shape)
    # np.savetxt("data_EEG/data_EEG.txt", all_extracted_speech_data)
    # print("labels")
    # print(labels.shape)
    # np.savetxt("data_EEG/label_EEG.txt", labels)
    # np.savetxt("length_EGG.txt", length)

    return all_extracted_speech_data, labels

def load_EGG_10():

    dir = 'EEG-512hz-10/brain_wave_format.json'
    with open(dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 读取 JSON 文件并解析为 Python 字典
    record_list = data.get('RECORDS', [])

    # 遍历 record 列表，查找并打印每个项中的 raw_brainwaves
    all_data = []
    i = 0
    for record in record_list:
        raw_brainwaves = record.get('raw_brainwaves')  # 获取 raw_brainwaves 项
        raw_brainwaves = raw_brainwaves[1:-1]
        # print(i)
        if i == 405:
            last = all_data[-1]
            all_data.append(last)
            i += 1
            continue
        raw_brainwaves = [float(x) for x in raw_brainwaves.split(',')]
        # print(len(raw_brainwaves))
        if len(raw_brainwaves) < 2500:
            x_new = np.linspace(0, len(raw_brainwaves) - 1, 2500)
            # 使用线性插值生成新的 300 个点
            f = interp1d(np.arange(len(raw_brainwaves)), raw_brainwaves, kind='linear')
            raw_brainwaves = f(x_new)
        # elif len(extracted_data) >150:
        #     extracted_data = extracted_data[:150]
        elif len(raw_brainwaves) > 2500:
            raw_brainwaves = raw_brainwaves[:2500]
        raw_brainwaves = np.array(raw_brainwaves)
        all_data.append(raw_brainwaves)
        i += 1

    data_np = np.vstack(all_data)
    #data_np=np.array(all_data)
    # print("data_np" + str(data_np.shape))
    # np.savetxt("data_np.txt", data_np)
    # 假设这是你3秒的音频数据，采样率500Hz，总样本数=3秒*500Hz
    sfreq_speech = 500
    duration = 3  # seconds
    samples = duration * sfreq_speech

    # 删除前n秒数据
    begin_cut = 0 * sfreq_speech
    # 一个单词n次
    end_cut = begin_cut + 1 * samples
    # 截取数据
    data_np = data_np[:, begin_cut:end_cut]
    #
    # sfreq_speech = 100
    all_extracted_speech_data = []
    all_speech_label = []
    all_range_start = []

    # 每个单词有n组
    per_num = 100
    error_num = 0
    error_num2 = 0
    length=[]
    # 数据每100个一组（每个3s[一个单词1次]）
    data1=[]
    data2=[]
    data3=[]
    for i in range(data_np.shape[0]):
        # if i<100:
        #     plt.plot(data_np[i])
        #     plt.savefig(f"data_EEG/3/data_np/ind_{i}.png")
        #     plt.close()
        # print(data_np[i].shape)
        ave = 10
        while abs(ave) > 1e-10:
            ave = np.average(data_np[i])
            data_np[i] = data_np[i] - ave
        word_data = reduce_by_max_abs(data_np[i])
        # print(word_data.shape)
        if word_data.shape[0] < 300 and word_data.shape[0] > 310:
            continue

        # word_data = word_data[:300]
        # if i<100:
        #     plt.plot(word_data)
        #     plt.savefig(f"data_EEG/3/word_data/ind_{i}.png")
        #     plt.close()
        # # print(word_data.shape)
        # 每隔3秒提取一次
        segments = np.array([word_data[i:i + samples] for i in range(0, len(word_data), samples)])
        for j in range(segments.shape[0]):
            data = segments[j]

            # 使用短时能量来检测说话部分
            window_size = int(0.1 * sfreq_speech)  # 100ms的滑动窗口
            ste = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='valid')

            # 找到能量最大的一段，假设这段是说话部分
            max_index = np.argmax(ste)
            # print('max_index:')
            # print(max_index)
            ste_threshold = np.mean(ste) + np.std(ste)  # 设置一个阈值
            speaking_indices = np.where(ste > ste_threshold)[0]

            # 数据找不到说话部位
            if len(speaking_indices) == 0:
                #print(f'数据找不到说话部位_{i}_{j}')
                all_range_start.append(-1)

                continue
            # 确定说话部分的开始和结束(在信号连续的情况下，使用这种。不连续就还得改进)
            start_speaking = speaking_indices[0]
            end_speaking = speaking_indices[-1]

            # 计算说话部分长度
            speaking_duration = end_speaking - start_speaking
            # print("speaking_duration"+str(speaking_duration))

            # 需要补充的数据长度
            needed_duration = 1.5 * sfreq_speech  # 目标时长为1.5秒
            # print("needed_duration"+str(needed_duration))
            remaining_duration = needed_duration - speaking_duration
            # print("remaining_duration"+str(remaining_duration))

            if remaining_duration < 0:
                # print(f'说话长度太长_阈值小了_{i}_{j}')
                all_range_start.append(-1)
                # print(f'默认处理_{i}_{j}，取0.5-2s')
                left = int(0.5 * sfreq_speech)
                right = int(2 * sfreq_speech)
                extracted_data = data[left:right]
                extracted_data = [float(i) for i in extracted_data]
                extracted_data = np.array(extracted_data)
                #data2.append(extracted_data)
                # if i < 100:
                #     plt.plot(extracted_data)
                #     plt.savefig(f"data_EEG/3/data2/i_{i}_j_{j}.png")
                #     plt.close()

                if extracted_data.shape[0] < 500:
                    x_new = np.linspace(0, extracted_data.shape[0]-1, 500)
                    f = interp1d(np.arange(extracted_data.shape[0]), extracted_data, kind='linear')
                    extracted_data = np.array(f(x_new))
                elif extracted_data.shape[0] >500:
                    extracted_data = extracted_data[:500]
                # if len(extracted_data) != 150:
                #     error_num2 += 1
                # length.append(len(extracted_data))
                all_extracted_speech_data.append(extracted_data)
                all_speech_label.append(i // per_num)
                all_range_start.append(left)

                continue
            #左右补充的数据量
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
            # if right - left != 150:  # debug用
            #     print("error")
            # if right > data.shape[0]:  # data.shape
            #     if data.shape[0] >= 300:
            #         left = int(300 - 1.5 * 100)
            #         right = 300
            #     else:
            #         left = data.shape[0] - 150
            #         right = data.shape[0]

            # 提取最终的1.5秒数据
            extracted_data = data[left:right]
            # data3.append(extracted_data)
            # plt.plot(extracted_data)
            # plt.savefig(f"data_EEG/3/data3/i_{i}_j_{j}.png")
            # plt.close()

            # if extracted_data.shape[0] < 150:
            #     ind1 = int(np.round((150 - extracted_data.shape[0]) / 2))
            #     ind2 = 150 - extracted_data.shape[0] - ind1
            #     array_left = np.full(ind1, extracted_data[0])
            #     # print(array_left.shape)
            #     array_right = np.full(ind2, extracted_data[-1])
            #     # print(array_right.shape)
            #     extracted_data = np.concatenate([array_left, extracted_data, array_right])

            if len(extracted_data) < 500:
                x_new = np.linspace(0, len(extracted_data)-1, 500)
                # 使用线性插值生成新的 300 个点
                f = interp1d(np.arange(len(extracted_data)), extracted_data, kind='linear')
                extracted_data = f(x_new)
            elif len(extracted_data) >500:
                extracted_data = extracted_data[:500]
            # if len(extracted_data) != 150:
            #     error_num+=1
            # print(f'完成_{i}_{j}')
            # print("extracted_data" + str(len(extracted_data)))
            length.append(len(extracted_data))
            all_extracted_speech_data.append(extracted_data)
            all_speech_label.append(i // per_num)
            all_range_start.append(left)

    # print("error" + str(error_num) + " "+str(error_num2))
    # 计算每个位置之前（包括该位置）非-1的个数
    # non_negative_ones_counts = count_non_negative_ones(all_range_start)
    #
    all_extracted_speech_data = np.array(all_extracted_speech_data)
    # all_extracted_eeg_data = all_extracted_speech_data
    # np.savetxt("data1.txt", all_extracted_speech_data)
    # np.savetxt("data2.txt", all_extracted_speech_data)
    # np.savetxt("data3.txt", all_extracted_speech_data)
    labels = np.array(all_speech_label)
    # print(f'label种类：{np.unique(labels)}')
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
        # all_extracted_eeg_data = all_extracted_eeg_data[mask]
        labels = labels[mask]

    label_stander = True

    if label_stander:
        # 重映射标签为从0开始连续的
        unique_labels = np.unique(labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        # print('label_mapping')
        # print(label_mapping)
        labels = np.vectorize(label_mapping.get)(labels)

    # print("all_extracted_speech_data")
    # print(all_extracted_speech_data.shape)
    # np.savetxt("data_EEG/data_EEG.txt", all_extracted_speech_data)
    # print("labels")
    # print(labels.shape)
    # np.savetxt("data_EEG/label_EEG.txt", labels)
    # np.savetxt("length_EGG.txt", length)

    return all_extracted_speech_data, labels


if __name__ == '__main__':
    #改成你们自己的文件路径
    root_data = 'mytest_100'
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

    # all_extracted_speech_data,labels = load_EGG()
    print(all_extracted_speech_data.shape)
    print(labels.shape)
    print(labels)
    # all_extracted_speech_data,labels = load_laryngea_520_10()

