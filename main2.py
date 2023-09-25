import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#   cluster  id     x     y      distance
# 0       0  76  5800  7800  68053.092930
# 0       1  24  7700  2300  79721.329094
# 0       2  56  1200  5300  92827.121728
data =[[0,76,5800,7800,68053.092930],[1,24,7700,2300,79721.329094],[2,56,1200,5300,92827.121728]]
hospital = pd.DataFrame(data,columns=['cluster','id','x','y','distance'])
place = pd.read_excel('data//place1.xlsx')
roads = pd.read_excel("data//road.xlsx")
# points0 = place[place['cluster']==0].reset_index()
# points0_data = points0.loc[:,['id','x','y']]
# colors = ['yellow', 'green', 'blue']
#
# for i in range(3):
#     cluster = place[place['cluster'] == i]
#     plt.scatter(cluster['x'], cluster['y'], color=colors[i], label='part{}'.format(i))
#
#     # 标注序号
#     for x, y, index in zip(cluster['x'], cluster['y'], cluster['id']):
#         plt.text(x, y, str(index), ha='center', va='bottom', fontsize=10)
#
#
# plt.scatter(hospital['x'],hospital['y'],color = 'red')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('1')
# #plt.legend()
# plt.show()
# 创建空图
G = nx.Graph()

# 分割数据为三个部分
part0 = place[place['cluster'] == 0]
part1 = place[place['cluster'] == 1]
part2 = place[place['cluster'] == 2]

# 添加节点到图中
for _, row in place.iterrows():
    G.add_node(row['id'], x=row['x'], y=row['y'], cluster=row['cluster'])

# 添加路线到图中
for _, row in roads.iterrows():
    start_id = row['start']
    end_id = row['end']
    start_node = place[place['id'] == start_id].iloc[0]
    end_node = place[place['id'] == end_id].iloc[0]
    distance = math.sqrt((end_node['x'] - start_node['x']) ** 2 + (end_node['y'] - start_node['y']) ** 2)
    G.add_edge(start_id, end_id, weight=distance)

# 存储每个部分起始点到其他点的最短路径和距离
shortest_paths = {}
best_paths = pd.DataFrame()
shortest_paths_T = pd.DataFrame()

# 遍历每个部分
for part in [part0, part1, part2]:
    # 获取当前部分的起始点
    start_nodes = part[part['id'].isin(hospital[hospital['cluster']==part['cluster'].unique()[0]]['id'])]

    # 存储当前部分起始点到其他点的最短路径和距离
    part_shortest_paths = {}

    # 遍历起始点
    for _, start_node in start_nodes.iterrows():
        start_id = start_node['id']
        distances = nx.single_source_dijkstra_path_length(G, start_id)

        # 存储起始点到其他点的最短路径和距离
        for node_id, distance in distances.items():
            if node_id != start_id and node_id in part['id'].values:
                if node_id not in part_shortest_paths:
                    part_shortest_paths[node_id] = {'id':node_id,'path': nx.shortest_path(G, start_id, node_id),'distance': distance}

    # 将当前部分的最短路径和距离存储到总的字典中
    shortest_paths = pd.DataFrame(part_shortest_paths)
    shortest_paths_T = shortest_paths.T
    best_paths = pd.concat([best_paths,shortest_paths_T])

print(best_paths)
bestpaths = best_paths.loc[:,['path']]

print(bestpaths)
# 打印每个部分起始点到其他点的最短路径和距离
for target_id, path_info in shortest_paths.items():
    print(f"Target ID: {target_id}")
    print(f"Shortest Path: {path_info['path']}")
    print(f"Distance: {path_info['distance']}")
    print()

colors = ['yellow', 'green', 'blue']
for i in range(3):
    cluster = place[place['cluster'] == i]
    plt.scatter(cluster['x'], cluster['y'], color=colors[i], label='part{}'.format(i))

    # 标注序号
    for x, y, index in zip(cluster['x'], cluster['y'], cluster['id']):
        plt.text(x, y, str(index), ha='center', va='bottom', fontsize=10)


plt.scatter(hospital['x'],hospital['y'],color = 'red')
for i in range(len(roads)):
    start_id = roads.iloc[i]['start']
    end_id = roads.iloc[i]['end']
    start_coords = place[place['id'] == start_id][['x', 'y']].values[0]
    end_coords = place[place['id'] == end_id][['x', 'y']].values[0]
    plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], linewidth=0.5,linestyle='-',color='black')
for _, path in bestpaths.iterrows():
    path_color = colors[place[place['id'] == path['path'][0]]['cluster'].values[0]]
    path_x = place[place['id'].isin(path['path'])]['x']
    path_y = place[place['id'].isin(path['path'])]['y']
    plt.plot(path_x, path_y, color=path_color,linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('1')
#plt.legend()
plt.show()
