from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

def cal_shannon_ent(dataset):
    """
    计算熵
    """
    # 1. 计算数据集中样本的总数
    num_entries = len(dataset)
    # 2. 创建一个字典，用于统计每个类别标签出现的次数
    labels_counts = {}
    # 3. 遍历数据集中的每条记录
    for feat_vec in dataset:
        # feat_vec[-1] 表示每条样本的最后一个元素 类别标签
        current_label = feat_vec[-1]
        # 如果该标签是第一次出现，则在字典中初始化为 0
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        # 累加该标签出现的次数
        labels_counts[current_label] += 1

        #print("类别统计：", labels_counts)
    # 4. 计算香农熵
    shannon_ent = 0.0
    # 遍历字典中的每个类别及其计数
    for key in labels_counts:
        # 计算该类别的概率
        prob = float(labels_counts[key])/num_entries
        # 根据香农熵公式累加：
        shannon_ent -= prob*log(prob, 2)
    # 5. 返回计算得到的熵值
    return shannon_ent


def create_dataSet():
    """
    熵接近 1，说明“yes”和“no”两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0,类别越集中，数据集越“纯”或“确定性越强”
    """
    dataset = [[1, 1, 'yes'],
               [1.1, 'yes'],
               [1, 1,'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels


dataset, labels = create_dataSet()
print(cal_shannon_ent(dataset))


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类
    
    参数：
        input_tree: 训练好的决策树
        feat_labels: 特征标签列表
        test_vec: 测试样本的特征向量
    
    返回：
        分类结果
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label

    # 如果测试样本的特征值不在训练树的范围内，返回None
    return None

def calculate_accuracy(tree, dataset, labels):
    """
    计算决策树在训练集上的准确率
    
    参数：
        tree: 训练好的决策树
        dataset: 训练数据集
        labels: 特征标签列表
    
    返回：
        accuracy: 准确率（0-1之间的浮点数）
    """
    correct_count = 0
    total_count = len(dataset)

    for i in range(total_count):
        test_vec = dataset[i][:-1]  # 去掉标签
        true_label = dataset[i][-1]  # 真实标签
        predicted_label = classify(tree, labels, test_vec)

        if predicted_label == true_label:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy


# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


#decision_node：定义“决策节点”的外观样式。
#boxstyle="sawtooth" 表示锯齿边框，常用于显示决策节点；
#fc='0.8'（facecolor）填充颜色为灰白色（0.8 表示灰度级）。
decision_node=dict(boxstyle="sawtooth",fc='0.8')

#leaf_node：定义“叶节点”的样式。
#boxstyle="round4" 表示圆角矩形边框；
#fc='0.8' 同样灰白填充。
leaf_node=dict(boxstyle="round4",fc='0.8')

#arrow_args：定义箭头样式。
#arrowstyle="<-" 表示箭头方向从子节点指向父节点。
arrow_args=dict(arrowstyle="<-")

# #node_txt：节点文字（显示在框中的文字，如“决策节点”、“叶节点”）。
# #center_pt：节点中心位置（子节点的位置）。
# #parent_pt：父节点位置，用于绘制箭头的起点。
# #node_type：节点样式（decision_node 或 leaf_node）。
# def plot_node(node_txt,center_pt,parent_pt,node_type): 
#     #annotate()：用于在图中添加带箭头的注释（文字+箭头）。
#     #xy=parent_pt：箭头起点（父节点位置）。
#     #xytext=center_pt：箭头终点+文字显示位置（子节点位置）。
#     #xycoords='axes fraction'：说明坐标用的是“轴的比例坐标”，即 (0,0) 是左下角，(1,1) 是右上角；
#     #bbox=node_type：节点边框样式；
#     #arrowprops=arrow_args：箭头样式；
#     #va='center'，ha='center'：文字居中对齐。
#     create_plot.ax1.annotate(node_txt,xy=parent_pt,xycoords='axes fraction',
#                              xytext=center_pt,textcoords='axes fraction',
#                              va='center',ha='center',bbox=node_type,arrowprops=arrow_args)
    

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')
    
def create_plot():
    fig=plt.figure(1,facecolor='white')  ## 新建一张图，背景白色
    fig.clf()                             # 清空之前的内容（防止重叠）
    create_plot.ax1=plt.subplot(111,frameon=False) # 创建一个子图，不显示坐标轴边框
    plot_node('决策节点',(0.5,0.1),(0.1,0.5),decision_node) # 画一个决策节点,节点位置 (0.5, 0.1)，箭头从 (0.1, 0.5) 指向节点；
    plot_node('叶节点',(0.8,0.1),(0.3,0.8),leaf_node) # 画一个叶节点,节点位置 (0.8, 0.1)，箭头从 (0.3, 0.8) 指向节点。
    plt.show()                                        # 显示图像

def get_num_leafs(my_tree):
    # my_tree 形如 {'特征A': {value1: 'yes', value2: {'特征B': {...}}}}
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    num_leafs = 0
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    max_depth = 0
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]

    num_leafs = get_num_leafs(my_tree)
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), x_off_y['y_off'])

    # 边文字（父->子取值）
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)

    # 决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)

    # 进入下一层
    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 叶子
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    # 返回上一层
    x_off_y['y_off'] += 1.0 / total_d

def create_plot(my_tree):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}

    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)

    plt.tight_layout()
    plt.show()

# ========== 运行：建树 + 绘图 ==========
# 示例数据集：天气与打球 (Play Tennis)
# weather_data = [
#     ['Sunny', 'Hot', 'High', False, 'No'],
#     ['Sunny', 'Hot', 'High', True, 'No'],
#     ['Overcast', 'Hot', 'High', False, 'Yes'],
#     ['Rain', 'Mild', 'High', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', True, 'No'],
#     ['Overcast', 'Cool', 'Normal', True, 'Yes'],
#     ['Sunny', 'Mild', 'High', False, 'No'],
#     ['Sunny', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'Normal', False, 'Yes'],
#     ['Sunny', 'Mild', 'Normal', True, 'Yes'],
#     ['Overcast', 'Mild', 'High', True, 'Yes'],
#     ['Overcast', 'Hot', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'High', True, 'No']
# ]
labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']

# 生成决策树
#tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
#create_plot(tree)

lenspath = (r"C:\Users\DELL\Documents\GitHub\tree\tree.py")

def load_data(filepath):
    data=[]
    fr = open(filepath)
    for line in fr:
        line = line.strip().split()
        data.append(line)

    return data 
labels_lenses = ['年龄','屈光','散光','泪液分泌']
dataset = load_data(lenspath)
tree = creat_tree(dataset, labels_lenses[:])
accuracy = calculate_accuracy(tree, dataset, labels_lenses)
print(f"训练集准确率: {accuracy*100:.2f}%")