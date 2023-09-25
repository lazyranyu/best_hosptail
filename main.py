# 这是一个示例 Python 脚本。
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

def kmeans_clustering(data,k):
    max_iterations = 100  # 最大迭代次数
    # 随机初始化三个坐标点：
    centers = data.sample(n=k, random_state=42)[['x', 'y']]
    for iteration in range(max_iterations):
        #计算每个点到中心点的距离：
        distance = pd.DataFrame()
        for i in range(k):
            distance[i] = np.sqrt((data['x']-centers.iloc[i]['x'])**2+(data['y']-centers.iloc[i]['y'])**2)
        # 根据距离确定每个点的所属簇
        data['cluster'] = distance.idxmin(axis=1)
        # 更新聚类中心
        new_centers = data.groupby('cluster').mean()[['x','y']]
        # 判断是否收敛
        if np.allclose(centers,new_centers):
            break

        centers = new_centers.copy()
    return data['cluster'],centers

def calculate_sum_distance(candidate,points):
    #计算目标点到其他点距离总和
    total_distance  =0
    x0 = candidate['x']
    y0 = candidate['y']
    for i in range(len(points)):
        total_distance += np.sqrt((x0-points['x'][i])**2+(y0-points['y'][i])**2)
    return total_distance

def SA(points,inital_T,cool_rate,max_counts):
    #退火算法
    #随机初始化一个点，把该点设为最优解和当前解
    points = points.reset_index()
    current_solution = points.sample(n=1)#,random_state=10
    best_solution = current_solution.copy().astype(float)
    #c初始化当前解的函数值（该点到其他点的距离和），并设为最优解
    current_distance = calculate_sum_distance(current_solution,points)
    best_distance = current_distance.copy().astype(float)

    #进行迭代
    temperature = inital_T
    iteration = 0
    while temperature > 0 and iteration < max_counts:
        #新解
        new_solutiom = points.loc[np.random.randint(len(points))]

        #新解函数值
        new_distance = calculate_sum_distance(new_solutiom,points)

        #判断是否符合标准
        if new_distance.item() < current_distance.item() or np.random.rand() < np.exp(-(current_distance.item()-new_distance.item())/temperature):
            current_solution = new_solutiom
            current_distance = new_distance
        #更新最优解
        if new_distance.item() < best_distance.item():
            best_solution = new_solutiom
            best_distance = new_distance

        temperature *= cool_rate
        iteration += 1
    return best_solution,float(best_distance)

place = pd.read_excel('data//place.xlsx')
road = pd.read_excel('data//road.xlsx')
road_start = road['start']
starts = place[place['id'].isin(road_start)][['x','y']].values
road_end = road['end']
ends = place[place['id'].isin(road_end)][['x','y']].values
print(place)
k=3
inital_T = 100
cool_rate = 0.9
max_counts = 100
cluster,centers = kmeans_clustering(place,k)
hospital_id = np.ndarray([k],dtype=int)
hospital_distance = np.ndarray([k],dtype=float)
hospital = pd.DataFrame()

for i in range(k):
    points = place[place['cluster'] == i]
    hospital0 = pd.DataFrame(columns=['cluster','id','x','y','distance'])
    best_soulution,best_distance = SA(points,inital_T,cool_rate,max_counts)
    hospital0 = pd.concat([hospital0,best_soulution.to_frame().T], ignore_index=True)
    hospital0['distance'] = best_distance
    hospital_id[i] = best_soulution['id']
    hospital_distance[i] = best_distance
    hospital = pd.concat([hospital,hospital0])

hospital = hospital.drop('index',axis=1)
print(hospital)
place.loc[97,'cluster']=2
print(place)
#place.to_excel('data//place1.xlsx',index=False)
# 绘制散点图
colors = ['yellow', 'green', 'blue']

for i in range(k):
    cluster = place[place['cluster'] == i]
    plt.scatter(cluster['x'], cluster['y'], color=colors[i], label='part{}'.format(i))

    # 标注序号
    for x, y, index in zip(cluster['x'], cluster['y'], cluster['id']):
        plt.text(x, y, str(index), ha='center', va='bottom', fontsize=10)


plt.scatter(hospital['x'],hospital['y'],color = 'red')
for i in range(len(road)):
    start_id = road.iloc[i]['start']
    end_id = road.iloc[i]['end']
    start_coords = place[place['id'] == start_id][['x', 'y']].values[0]
    end_coords = place[place['id'] == end_id][['x', 'y']].values[0]
    plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], linewidth=0.5,linestyle='-',color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('1')
#plt.legend()
plt.show()
