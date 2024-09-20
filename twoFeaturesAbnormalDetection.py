import numpy as np
import matplotlib
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# 修改，两个特征的基础之上增加附加点
def two_feature_abnormal_detection(one_dimension_time_series):

    # 传入一维时间序列数据（自闭症脑电数据）
    data_list = one_dimension_time_series.split()  # 使用split()方法按空格分割字符串
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
    plt.figure(figsize=(12, 6))
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
    plt.xlabel('时间 t',fontsize=14)
    plt.ylabel('Z值大小',fontsize=14)

    plt.legend()
    plt.show()
    print("成功")


#测试中，传入的data_list中有116个数据点，其中max1中有36个极大值点、min1中有37个极小值点 ，一共73个极值点

