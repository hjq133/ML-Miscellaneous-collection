import matplotlib.pyplot as plt
import math
import numpy as np

MIN_DISTANCE = 0.000001  # mini error


def create_data(num):  # data num
    data1 = np.random.rand(2, num) * 1
    data2 = np.random.rand(2, num) * 1 + 3
    data3 = np.random.rand(2, num) * 1 + 1.5
    data = np.concatenate((data1, data2, data3), axis=1)
    x, y = data
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.grid(True)
    plt.show()
    return np.transpose(data, [1, 0])


def gaussian_kernel(distance, bandwidth):
    m = np.shape(distance)[0]  # 样本个数
    right = np.mat(np.zeros((m, 1)))  # mX1的矩阵
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))

    gaussian_val = left * right
    return gaussian_val


def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m, n = np.shape(points)
    # 计算距离
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)

    # 计算高斯核
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)

    # 计算分母
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]

    # 均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted


def euclidean_dist(point_a, point_b):
    # 计算pointA和pointB之间的欧式距离
    total = (point_a - point_b) * (point_a - point_b).T
    return math.sqrt(total)


def group_points(mean_shift_points):
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment


def train_mean_shift(points, kernel_bandwidth):
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0  # 训练的代数
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m  # 标记是否需要漂移

    # 计算均值漂移向量
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print("\titeration : " + str(iteration))
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏移均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)  # 对样本点进行漂移
            dist = euclidean_dist(p_new, p_new_start)  # 计算该点与漂移后的点之间的距离

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:  # 不需要移动
                need_shift[i] = False

            mean_shift_points[i] = p_new
    # 计算最终的group
    group = group_points(mean_shift_points)  # 计算所属的类别
    return mean_shift_points, group


def show_result(origin_data, group, mean_shift_points):
    fig, ax = plt.subplots()
    color = ['b', 'c', 'r', 'g', 'm', 'k', 'y']
    for i in range(len(origin_data)):
        x, y = origin_data[i][0], origin_data[i][1]
        c = color[group[i]]
        ax.scatter(x, y, c=c)
    for i in range(len(mean_shift_points)):
        x, y = mean_shift_points[i, 0], mean_shift_points[i, 1]
        c = color[group[i]]
        ax.scatter(x, y, c=c, marker='*', s=100)
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    num = 15
    data = create_data(num)
    origin_data = np.copy(data)
    mean_shift_points, group = train_mean_shift(data, kernel_bandwidth=0.3)
    show_result(origin_data, group, mean_shift_points)
