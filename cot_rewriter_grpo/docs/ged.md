# 背景
目前训练：/home/luoys/cdad/cot_rewriter_grpo/scripts/grpo/start_a100.sh
包含了fsf penalty，我认为这个penalty太粗暴，并且鼓励模型输出更多的失败节点未必是好事，我希望增加一个逻辑结构奖励，比较rewrite cot和original cot的reasoning graph的reasoning graph的相似度，奖励值为1-0之间的值，相似度越高，奖励越高。

# 添加logic奖励
reasoning graph的计算方式仍然是通过fsf_judge_model获得dot_str，然后解析为图的形式，通过以下伪代码示意，计算两个图的编辑距离（Graph Edit Distance, GED）。最后通过组内均值方差归一化操作，再用sigmoid函数，将GED映射到0-1之间，作为奖励值。并且可以设置lambda权重，加入到total reward中

你参考下面逻辑进行实现：

import networkx as nx
import pydot
import numpy as np

def calculate_sigmoid_ged_reward(orig_dot_str, rew_dot_str, alpha=0.5):
    """
    基于 GED 和 Sigmoid 归一化的推理图相似度计算
    alpha: 控制奖励下降的速率。alpha 越大，对逻辑差距越敏感。
    """
    
    # 1. 将 DOT 字符串转换为 NetworkX 有向图 (DiGraph)
    def dot_to_nx(dot_str):
        graphs = pydot.graph_from_dot_data(dot_str)
        if not graphs: return nx.DiGraph()
        pydot_graph = graphs[0]
        nx_graph = nx.nx_pydot.from_pydot(pydot_graph)
        return nx_graph

    g_orig = dot_to_nx(orig_dot_str)
    g_rew = dot_to_nx(rew_dot_str)

    # 2. 定义节点匹配和编辑代价
    # 核心：保护 lightpink (反思节点)
    def node_subst_cost(n1_attrs, n2_attrs):
        c1 = n1_attrs.get('fillcolor', 'lightblue')
        c2 = n2_attrs.get('fillcolor', 'lightblue')
        if c1 == c2: return 0.0
        # 如果把反思(pink)改成了普通步骤(blue)，代价极高
        if c1 == 'lightpink' and c2 == 'lightblue': return 5.0 
        return 5.0

    def node_del_cost(n_attrs):
        # 删除反思节点的代价极高
        if n_attrs.get('fillcolor') == 'lightpink': return 5.0
        return 2.0 # 删除普通蓝色节点代价较低（允许精简）

    def node_ins_cost(n_attrs):
        return 2.0 # 增加节点的代价

    # 3. 计算有向图编辑距离 (GED)
    # 使用启发式(Heuristic)算法以适应 GRPO 的实时性要求
    ged_distance = nx.graph_edit_distance(
        g_orig, g_rew,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        timeout=5.0 # 防止极端情况下的计算超时
    )
    # 4. 组内归一化，避免信号消失
    ged_distance 组内归一化，减去均值除以方差
    # 5. 使用 Sigmoid 进行归一化映射到 [0, 1]
    # 我们使用变体：f(x) = 2 / (1 + exp(alpha * x))
    # 当 ged=0, score=1.0; 当 ged很大, score趋近于0
    similarity_score = 2.0 / (1.0 + np.exp(alpha * ged_distance))
    
    return similarity_score

# 细节要求
1. 如果某个step的fsf_judge_model解析dot_str失败，则需要跳过这个step的logic reward计算
2. 开启r_logic后，则不用开启fsf_penalty了

# Swanlab和rollout_data保存
1. swanlab reward中记录ged的max,min,mean; 还有记录ged归一化后的ged_01的max,min,mean
2. rollout_data中记录ged,ged_01的值

