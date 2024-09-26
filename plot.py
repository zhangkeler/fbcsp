import os
import numpy as np
import glob
from collections import Counter
import copy
import pandas as pd
from show_data import reduce_by_max_abs,count_non_negative_ones
from mne.io import concatenate_raws,read_raw_edf
import matplotlib.pyplot as plt
def read_txt_to_array(file_path):
    array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行末的换行符并添加到数组中
            array.append(line.strip())
    return array


def plot_res():
    file_path2 = 'data_np.txt'
    file_path3 = 'data1.txt'
    file_path4 = 'data2.txt'
    file_path5 = 'data3.txt'
    array = read_txt_to_array('data_EEG.txt') #全部处理后
    for i in range(100):
        plt.plot(array[i])
        plt.savefig(f"data_EEG/2/data_EEG/ind_{i}.png")
        plt.close()
    array = read_txt_to_array('data_np.txt') #原数据
    for i in range(100):
        plt.plot(array[i])
        plt.savefig(f"data_EEG/2/data_np/ind_{i}.png")
        plt.close()
    array = read_txt_to_array('data1.txt')  # 前300
    for i in range(100):
        plt.plot(array[i])
        plt.savefig(f"data_EEG/2/data1/ind_{i}.png")
        plt.close()
    array = read_txt_to_array('data2.txt')  # 提取最大能量后 remaining_duration<0
    for i in range(100):
        plt.plot(array[i])
        plt.savefig(f"data_EEG/2/data2/ind_{i}.png")
        plt.close()
    array = read_txt_to_array('data3.txt')  # 提取最大能量后 remaining_duration>0
    for i in range(100):
        plt.plot(array[i])
        plt.savefig(f"data_EEG/2/data3/ind_{i}.png")
        plt.close()

if __name__ == '__main__':
    plot_res()