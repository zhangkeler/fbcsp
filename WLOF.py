import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import extrema
from sklearn.model_selection import StratifiedKFold
from array import array
import os
import matplotlib
import copy
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import pandas as pd

# 提取极值点特征并计算特征个数
def extrema_max_list(data):
    # num_samples = data.shape[0]
    # num_channels = data.shape[1]
    # max_count = 0
    # min_count = 0
    # print("1")
    # print(data.shape)
    #data_list = list(data)
    #print(data_list.shape)
    data = data[:,1]
    #print(data.shape)
    float_data_array = np.array(data)
    # print("data shape")
    # print(float_data_array.shape)
    # print(float_data_array)
    # 寻找极值点（极大值和极小值）
    extrema_max_points = []
    extrema_min_points = []
    extrema_points = []
    if float_data_array[0]>=float_data_array[1]:
        extrema_max_points.append((0,float_data_array[0],'startmax'))
        extrema_points.append((0,float_data_array[0],'startmax'))
    elif float_data_array[0] < float_data_array[1]:
        extrema_min_points.append((0,float_data_array[0],'startmin'))
        extrema_points.append((0,float_data_array[0],'startmin'))
    for k in range(1, len(float_data_array) - 1):
        if float_data_array[k] > float_data_array[k - 1] and float_data_array[k] > float_data_array[k + 1]:
            extrema_max_points.append((k, float_data_array[k], 'max'))
            extrema_points.append((k, float_data_array[k], 'max'))
        elif float_data_array[k] < float_data_array[k - 1] and float_data_array[k] < float_data_array[k + 1]:
            extrema_min_points.append((k, float_data_array[k], 'min'))
            extrema_points.append((k, float_data_array[k], 'min'))
    end = len(float_data_array) -1
    if float_data_array[end]>=float_data_array[end-1]:
        extrema_max_points.append((end,float_data_array[end],'startmax'))
        extrema_points.append((end,float_data_array[end],'startmax'))
    elif float_data_array[end] < float_data_array[end-1]:
        extrema_min_points.append((end,float_data_array[end],'startmin'))
        extrema_points.append((end,float_data_array[end],'startmin'))
            # 将极值点按照时间顺序排序
    extrema_max_points.sort(key=lambda x: x[0])
    extrema_min_points.sort(key=lambda x: x[0])
    extrema_points.sort(key=lambda x: x[0])

    return extrema_points

def important_point(D,extrema_points,beta,g):
    extrema_points_copy = copy.deepcopy(extrema_points)
    #global Z
    if len(extrema_points) >= math.floor(beta * (g-2)):
        EP = math.floor(beta * (g-2))
        AP = g-2-math.floor(beta*(g-2))
    else :
        EP = len(extrema_points) #10+
        AP = g -2 -len(extrema_points) #40+
    print("EP")
    print(EP)
    print("AP")
    print(AP)
    important_points = []
    start=0
    end=len(extrema_points)-1
    important_points.append(extrema_points[start])
    important_points.append(extrema_points[end])
    #del extrema_points[start]
    #del extrema_points[end-1]
    for j in range(EP):
        max_length = 0
        r =-1
        for i in range(1,len(extrema_points) - 1):
            nowdistance = abs(extrema_points[i][1] - extrema_points[i-1][1]) + abs(extrema_points[i][1]-extrema_points[i+1][1])
            if nowdistance > max_length :
                max_length = nowdistance
                r = i
        #print("r=" + str(r)+"len:extrema_points"+str(len(extrema_points)))
        if r!=-1 :
            important_points.append(extrema_points[r])
            del extrema_points[r]
    extrema_points = copy.deepcopy(extrema_points_copy)
    extrema_points = sorted(extrema_points, key=lambda x: x[0])
    important_points_2 = copy.deepcopy(important_points)
    for j in range(AP):
        # print("j"+str(j))
        max_length = 0
        a=-1
        #选出最大距离相邻重要点
        # print("len"+str(len(important_points)))
        for i in range(1,len(important_points)):
            nowdistance = abs(important_points[i][1] - important_points[i-1][1])
            t = math.floor((important_points[i - 1][0] + important_points[i][0]) / 2)
            flag = any(row[0] == t for row in important_points)
            if nowdistance > max_length and flag==False:
                max_length = nowdistance
                a = t
        t = a
        #print("t"+str(t))  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Z= D[t,1]
        #start = important_points[a-1][0]
        #end = important_points[a][0]
        Z =0
        # print("extrema_points")
        # print([row[0] for row in extrema_points])
        # print("start")
        # print(start)
        # print("end")
        # print(end)
        # while(start < end ):
        for i in range(len(extrema_points)-1):
            #if extrema_points[i][0]>=start and extrema_points[i][0]<=end:
                if extrema_points[i][0]<t and t<extrema_points[i+1][0] : #附加点在i和i+1之间
                    x= extrema_points[i+1][0] - extrema_points[i][0]
                    y= abs(extrema_points[i][1] - extrema_points[i+1][1])
                    if extrema_points[i][1] >= extrema_points[i+1][1]:
                        x1 = extrema_points[i + 1][0] - t
                    else:
                        x1 = t - extrema_points[i][0]
                    # print("x"+str(x)+"y"+str(y)+"x1"+str(x1))
                    if x ==0:
                        x = 1e-6
                    Z = y/x * x1
                    break
            # start += 1
        # print("Z"+str(Z))
        if Z!=0:
            important_points.append((t,Z,'add'))
        important_points = sorted(important_points, key=lambda x: x[0])
        #print([row[0] for row in important_points]) #!!!!!!!!!!!!!!!!
    return important_points

def max_angle_num_important(subsequence,w,important_points):
    start = subsequence[0, 0]
    end = subsequence[w - 1, 0]
    num = 0
    index = []
    # print("important_points shape")
    # print(important_points[0])
    # print(important_points[3][0])
    for i in range(len(important_points)):
        if important_points[i][0] >= start and important_points[i][0] <= end:
            num += 1
            index.append(i)
    # print(len(important_points))
    # print("num")
    # print(num)
    # print(index)
    maxangle = 0
    #print(subsequence.shape)
    for i in range(1,len(index)-1):
        A= np.array(important_points[index[i-1]][0:1])
        B=np.array(important_points[index[i]][0:1])
        C=np.array(important_points[index[i+1]][0:1])
        AB = B - A
        BC = C - B
        dot_product = np.dot(AB, BC)
        norm_AB = np.linalg.norm(AB)
        norm_BC = np.linalg.norm(BC)
        # 计算夹角的余弦值
        temp =norm_AB * norm_BC
        if temp ==0:
            temp = 1e-6
        cos_theta = dot_product / temp
        # 计算夹角（用反余弦得到角度，结果是弧度）
        angle_radians = np.arccos(cos_theta)
        # 将弧度转换为角度
        angle_degrees = np.degrees(angle_radians)
        if angle_degrees > maxangle:
            maxangle = angle_degrees

    return maxangle,num,index

def compute_feature(D,w,important_points):
    FV=[]
    # print("lenD"+str(len(D)))
    # print("w"+str(w))
    D=np.array(D)
    # print("D_shape")
    # print(D.shape)
    for i in range(D.shape[0]-w+1):
        avg = np.mean(D[i:i+w,1])
        temp = D[i:i+w,:]
        # print(i)
        # print(temp.shape)
        maxangle,numimportant,index = max_angle_num_important(temp,w,important_points)
        maxdiff = 0
        if numimportant > 1:
            for j in range(1, len(index)):
                nowdiff = abs(important_points[index[j]][1] - important_points[index[j - 1]][1])
                if nowdiff > maxdiff:
                    maxdiff = nowdiff
        FV.append((maxangle,avg,numimportant,maxdiff))
    return FV

def compute_outfactor(features,param_k):
    kwdist = []
    wdist=np.zeros((int(len(features)),int(len(features))))
    w=[]
    features = np.array(features)
    #print("feature")
    #print(features)

    for k in range(4):
        now_w = (np.sum(np.sum(features)) - np.sum(features[:, k])) / (3 * np.sum(np.sum(features)))
        w.append(now_w)
    for i in range(len(features)):
        for j in range(len(features)):
            temp = 0
            for k in range(4):
                temp += w[k]* ((features[i,k]-features[j,k])**2)
            wdist[i,j]=math.sqrt(temp)
        orderlist = np.sort(wdist[i,:])
        kwdist.append(orderlist[param_k-1])
    #print("kwdist")
    #print(kwdist)
    #print("wdist")
    #print(wdist)
    wlrdk = []
    for i in range(len(features)):
        temp = float(0)
        for j in range(len(features)):
            if j != i :
                temp += np.max([kwdist[j],wdist[i,j]])
        if temp==0:
            temp = 1e-6
        wlrdk.append(param_k/temp)
    wlrdk=np.array(wlrdk)
    #print("wlrdk")
    #print(wlrdk)

    WLOF =  (float(sum(wlrdk)) -wlrdk) /param_k / wlrdk
    return WLOF

def WLOF(D,w,g,s,beta,k):
    #归一化
    scaler = MinMaxScaler()
    second_column = D[:, 1].reshape(-1, 1)
    normalized_second_column = scaler.fit_transform(second_column)
    D[:, 1] = normalized_second_column.flatten()
    #print("normalize")
    #print(D)

    # LOWESS 进行局部加权平滑
    x = D[:, 0]
    y = D[:, 1]
    lowess = sm.nonparametric.lowess
    smoothed = lowess(y, x, frac=s)  # frac 是平滑的窗口大小，调整大小进行控制
    D[:,0] = smoothed[:, 0]
    D[:,1] = smoothed[:, 1]
    # print("smooth")
    #print(D)
    print("WLOF_D_init"+str(D.shape[0]))
    extrema_points = extrema_max_list(D)
    #print("extrema_points"+str(len(extrema_points)))
    #print(extrema_points)
    extrema_points_copy = copy.deepcopy(extrema_points)
    important_points = important_point(D,extrema_points_copy,beta,g)

    important_points_copy = copy.deepcopy(important_points)
    #print("important_points")
    #print(len(important_points))
    features = compute_feature(D,w,important_points_copy)

    outfactor = compute_outfactor(features,k)

    return extrema_points,important_points,features,outfactor

if __name__ == '__main__':
    dir = 'Earthquakes/Earthquakes_TRAIN.tsv'
    data = pd.read_csv(dir, sep='\t')
    #print(data[:,0])
    data2=np.array(data)
    print(data2.shape)
    data2=data2[:,1:]
    print(data2.shape)
    #取第几行数据

    data_index = 3
    data2=data2[data_index,:].T
    size=data2.shape[0]
    print(size)
    index_list = list(range(size))
    D = np.column_stack((index_list, data2))
    print(D)

    beta = 1/2
    #k = [5,10,15,20]
    k=5
    # s= [0.1,0.3,0.5,0.7,0.9]
    s=0.1
    #g=[40,50,60,70,80,90,100]
    g=60
    w= math.floor(0.1 * len(D))
    extrema_points,important_points,features,outfactor = WLOF(D, w, g, s, beta,k)
    print("outfactor")
    print(outfactor)
    print("extrema_points")
    print(len(extrema_points))
    print("data2")
    print(data2)
    print("D")
    print(D[:,1])

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 2)
    #ax1, ax2, ax3, ax4 = axes.flatten()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])
    # Plot A on the first subplot
    ax1.plot(data2, color='blue')
    ax2.plot(D[:,1])
    extrema_points_y = [item[1] for item in extrema_points]
    extrema_points_x = [item[0] for item in extrema_points]
    important_points_y = [item[1] for item in important_points]
    important_points_x = [item[0] for item in important_points]
    print("extrema_points")
    print(extrema_points_y)
    ax3.plot(extrema_points_x,extrema_points_y,  '-o',color='red',zorder=1)
    ax3.scatter(important_points_x,important_points_y,marker='^',s=100,zorder=2)
    ax4.plot(outfactor, color='yellow')
    #plt.plot(outfactor)
    plt.grid(True)
    plt.show()

    