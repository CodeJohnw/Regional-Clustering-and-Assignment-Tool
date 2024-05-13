
# "'五个区域两个中心'"

# from pulp import LpProblem, LpMinimize, lpSum, LpVariable, value

# # 定义基本参数
# V = set(range(1, 6))  # 5个基本区域
# V_c = set(range(1, 3))  # 2个中心
# p = len(V_c)  # 中心的数量
# mu = 20  # 平均面积大小
# tau = 0.1  # 偏差容差值

# # 假设一些距离、权重和区域大小的值
# q_v = {1: 2, 2: 3, 3: 1, 4: 2, 5: 3}
# d_iv = {(1, 1): 5, (1, 2): 8, (2, 1): 7, (2, 2): 6, (3, 1): 4, (3, 2): 9, (4, 1): 3, (4, 2): 5, (5, 1): 6, (5, 2): 4}
# s_v = {1: 5, 2: 4, 3: 3, 4: 6, 5: 2}

# # 定义Lp问题
# prob = LpProblem("RegionAssignment", LpMinimize)

# # 定义变量 x_iv
# x = {(v, i): LpVariable(f'x_{v}_{i}', 0, 1, LpVariable.isBinary) for v in V for i in V_c}

# # 定义目标函数
# prob += lpSum(q_v[v] * d_iv[v, i] * x[v, i] for v in V for i in V_c)

# # 定义约束条件
# for v in V:
#     prob += lpSum(x[v, i] for i in V_c) == 1

# for i in V_c:
#     prob += (1 - tau) * mu <= lpSum(s_v[v] * x[v, i] for v in V) <= (1 + tau) * mu

# # 求解问题
# prob.solve()

# # 输出结果
# print("Status:", prob.status)
# print("Total Distance:", value(prob.objective))

# # 输出每个基本区域被分配给哪个中心
# for v in V:
#     for i in V_c:
#         if value(x[v, i]) == 1:
#             print(f"Region {v} is assigned to Center {i}")



# "'十个区域三个中心'"


# from pulp import LpProblem, LpMinimize, lpSum, LpVariable, value

# # 定义基本参数
# V = set(range(1, 11))  # 10个基本区域
# V_c = set(range(1, 4))  # 3个中心
# p = len(V_c)  # 中心的数量
# mu = 20  # 平均面积大小
# tau = 0.1  # 偏差容差值

# # 假设一些距离、权重和区域大小的值
# q_v = {1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2}
# d_iv = {(1, 1): 5, (1, 2): 8, (1, 3): 6, (2, 1): 7, (2, 2): 6, (2, 3): 9,
#         (3, 1): 4, (3, 2): 5, (3, 3): 3, (4, 1): 3, (4, 2): 5, (4, 3): 7,
#         (5, 1): 6, (5, 2): 4, (5, 3): 8, (6, 1): 2, (6, 2): 7, (6, 3): 5,
#         (7, 1): 9, (7, 2): 2, (7, 3): 4, (8, 1): 5, (8, 2): 3, (8, 3): 6,
#         (9, 1): 1, (9, 2): 6, (9, 3): 2, (10, 1): 8, (10, 2): 4, (10, 3): 1}
# s_v = {1: 5, 2: 4, 3: 3, 4: 6, 5: 2, 6: 3, 7: 5, 8: 2, 9: 4, 10: 6}

# # 定义Lp问题
# prob = LpProblem("RegionAssignment", LpMinimize)

# # 定义变量 x_iv
# x = {(v, i): LpVariable(f'x_{v}_{i}', 0, 1, LpVariable.isBinary) for v in V for i in V_c}

# # 定义目标函数
# prob += lpSum(q_v[v] * d_iv[v, i] * x[v, i] for v in V for i in V_c)

# # 定义约束条件
# for v in V:
#     prob += lpSum(x[v, i] for i in V_c) == 1

# for i in V_c:
#     prob += (1 - tau) * mu <= lpSum(s_v[v] * x[v, i] for v in V) <= (1 + tau) * mu

# # 求解问题
# prob.solve()

# # 输出结果
# print("Status:", prob.status)
# print("Total Distance:", value(prob.objective))

# # 输出每个基本区域被分配给哪个中心
# for v in V:
#     for i in V_c:
#         if value(x[v, i]) == 1:
#             print(f"Region {v} is assigned to Center {i}")


# '''封装成函数'''
# from pulp import LpProblem, LpMinimize, lpSum, LpVariable, value

# def solve_region_assignment(tau, V, V_c, q_v, d_iv, s_v):
#     p = len(V_c)  # 中心的数量
#     mu = sum(s_v.values()) / p  # 平均面积大小

#     # 定义Lp问题
#     prob = LpProblem("RegionAssignment", LpMinimize)

#     # 定义变量 x_iv
#     x = {(v, i): LpVariable(f'x_{v}_{i}', 0, 1, LpVariable.isBinary) for v in V for i in V_c}

#     # 定义目标函数
#     prob += lpSum(q_v[v] * d_iv[v, i] * x[v, i] for v in V for i in V_c)

#     # 定义约束条件
#     for v in V:
#         prob += lpSum(x[v, i] for i in V_c) == 1

#     for i in V_c:
#         prob += (1 - tau) * mu <= lpSum(s_v[v] * x[v, i] for v in V) <= (1 + tau) * mu

#     # 求解问题
#     prob.solve()

#     # 输出结果
#     status = prob.status
#     total_distance = value(prob.objective)

#     # 输出每个基本区域被分配给哪个中心
#     assignments = {v: [i for i in V_c if value(x[v, i]) == 1] for v in V}

#     return status, total_distance, assignments

# # 示例用法
# tau = 0.1
# V = set(range(1, 11))  # 10个基本区域
# V_c = set(range(1, 4))  # 3个中心
# q_v = {1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2}
# d_iv = {(1, 1): 5, (1, 2): 8, (1, 3): 6, (2, 1): 7, (2, 2): 6, (2, 3): 9,
#         (3, 1): 4, (3, 2): 5, (3, 3): 3, (4, 1): 3, (4, 2): 5, (4, 3): 7,
#         (5, 1): 6, (5, 2): 4, (5, 3): 8, (6, 1): 2, (6, 2): 7, (6, 3): 5,
#         (7, 1): 9, (7, 2): 2, (7, 3): 4, (8, 1): 5, (8, 2): 3, (8, 3): 6,
#         (9, 1): 1, (9, 2): 6, (9, 3): 2, (10, 1): 8, (10, 2): 4, (10, 3): 1}
# s_v = {1: 5, 2: 4, 3: 3, 4: 6, 5: 2, 6: 3, 7: 5, 8: 2, 9: 4, 10: 6}

# status, total_distance, assignments = solve_region_assignment(tau, V, V_c, q_v, d_iv, s_v)
# print("Status:", status)
# print("Total Distance:", total_distance)
# print("Assignments:", assignments)


'''城镇规模案例'''
from pulp import LpProblem, LpMinimize, lpSum, LpVariable, value

def solve_region_assignment(tau, V, V_c, q_v, d_iv, s_v):  #选票中心所在的小区，数量+shp文件
    p = len(V_c)  # 中心的数量
    mu = sum(s_v.values()) / p  # 平均面积大小

    # 定义Lp问题
    prob = LpProblem("RegionAssignment", LpMinimize)

    # 定义变量 x_iv
    x = {(v, i): LpVariable(f'x_{v}_{i}', 0, 1, LpVariable.isBinary) for v in V for i in V_c}

    # 定义目标函数
    prob += lpSum(q_v[v] * d_iv[v, i] * x[v, i] for v in V for i in V_c)

    # 定义约束条件
    for v in V:
        prob += lpSum(x[v, i] for i in V_c) == 1

    for i in V_c:
        prob += (1 - tau) * mu <= lpSum(s_v[v] * x[v, i] for v in V) <= (1 + tau) * mu

    # 求解问题
    prob.solve()

    # 输出结果
    status = prob.status
    total_distance = value(prob.objective)

    # 输出每个基本区域被分配给哪个中心
    assignments = {v: [i for i in V_c if value(x[v, i]) == 1] for v in V}

    return status, total_distance, assignments

'''case 1 :输出分配后的各区域总面积'''
# tau = 0.1
# V = set(range(1, 11))  # 10个基本区域
# V_c = set(range(1, 4))  # 3个中心

# # 每个区域的面积
# s_v = {1: 50000, 2: 40000, 3: 30000, 4: 60000, 5: 20000, 
#        6: 30000, 7: 50000, 8: 20000, 9: 40000, 10: 60000}

# # 每个区域的人口
# population_v = {1: 30000, 2: 25000, 3: 20000, 4: 35000, 5: 12000,
#                 6: 18000, 7: 30000, 8: 13000, 9: 25000, 10: 40000}

# # 计算平均面积大小
# p = len(V_c)  # 中心的数量
# mu = sum(s_v.values()) / p

# # 距离权重
# q_v = {v: population_v[v] for v in V}

# # 距离矩阵，这里简化为区域面积的反比例
# d_iv = {(v, i): 1 / s_v[v] for v in V for i in V_c}

# # 调用函数求解
# status, total_distance, assignments = solve_region_assignment(tau, V, V_c, q_v, d_iv, s_v)

# # 输出结果
# print("Status:", status)
# print("Total Distance:", total_distance)

# # 输出每个基本区域被分配给哪个中心和区域面积
# for v, assigned_centers in assignments.items():
#     print(f"Region {v} is assigned to Centers {assigned_centers}, Area: {s_v[v]}")



# # 输出每个基本区域被分配给哪个中心和区域面积
# for i in V_c:
#     total_area = sum(s_v[v] for v, assigned_centers in assignments.items() if i in assigned_centers)
#     print(f"Total Area for Center {i}: {total_area}")




'''case 2:100个区域8个中心'''

import random

tau = 0.1
V = set(range(1, 101))  # 100个基本区域
V_c = set(range(1, 9))   # 8个中心

# 随机生成每个区域的面积和人口
s_v = {v: random.randint(10000, 50000) for v in V}
population_v = {v: random.randint(5000, 30000) for v in V}

# 计算平均面积大小
p = len(V_c)  # 中心的数量
mu = sum(s_v.values()) / p

# 距离权重
q_v = {v: population_v[v] for v in V}

# 距离矩阵，这里简化为区域面积的反比例
d_iv = {(v, i): 1 / s_v[v] for v in V for i in V_c}

# 调用函数求解
status, total_distance, assignments = solve_region_assignment(tau, V, V_c, q_v, d_iv, s_v)

# 输出结果
print("Status:", status)
print("Total Distance:", total_distance)

# 输出每个基本区域被分配给哪个中心和区域面积
for i in V_c:
    total_area = sum(s_v[v] for v, assigned_centers in assignments.items() if i in assigned_centers)
    print(f"Total Area for Center {i}: {total_area}")

# 输出每个基本区域被分配给哪个中心和区域面积
for v, assigned_centers in assignments.items():
    print(f"Region {v} is assigned to Centers {assigned_centers}, Area: {s_v[v]}")
