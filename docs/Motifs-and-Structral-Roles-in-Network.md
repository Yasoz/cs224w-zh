## Motifs and Structral Rules in Network

### Subgraphs

在本节中，我们将介绍子图的定义。子网络或子图是网络的构建块，使我们能够表征和区分网络。

例如，在下图中，我们显示了所有大小为3的非同构有向子图。这些子图在边数或边方向上互不相同

<img src="./img/Subgraphs_example.png?style=centerme" alt="Figure 1" style="zoom:50%;" />

### Motifs

网络主题(Motifs)是网络中不断出现的重要互连模式。在此，图案意味着它是小的诱导子图。注意图 $G$ 的诱导子图是由图 $G$ 的顶点的子集 $X$ 和连接子集 $X$ 中的顶点对的所有边组成的图。

主题的重复表示它以高频率发生。我们允许Motifs重叠。

主题的重要性 (Significance) 意味着它比预期的要频繁。这里的关键思想是，我们说在实际网络中出现的子图比在随机网络中发生的频率要高得多。Significance可以使用Z-score分数定义为:
$$
Z_{i} = \frac{N_{i}^{real} - \overline N_{i}^{rand}}{std(N_{i}^{rand})}
$$

其中$N_{i}^{real}$ 是在真实网络 $G^{real}$ 中类型为 $i$ 的子图的数量 ，其中$N_{i}^{rand}$ 是在随机网络 $G^{rand}$ 中类型为 $i$ 的子图的数量 。

网络的SP(significance profile) 定义为：

$$
SP_{i} = \frac{Z_{i}}{\sqrt{\sum_{j} {Z_j^{2}}}}
$$

其中SP是归一化的Z-scores的向量。

### Configuration Model

配置模型 (configuration model) 是具有给定度数序列的随机图 $k_1$, $k_2$, ..., $k_N$ ,可以用作“空”模型，然后与真实网络进行比较。配置模型可以很容易地生成，如图所示。

<img src="./img/Configuration_Model.png?style=centerme" alt="Figure 2" style="zoom:50%;" />

另一种生成方式如下：
1) 从一个给定的图 $G$ 开始;
2) 随机选择一对边 A->B, C->D , 交换端点得到 A->D, C->B, 重复以上步骤 $Q\times \vert E\vert$ 次.
通过这种方式，我们将获得一个具有相同节点度和随机重新连接边的随机重连接图。

### Graphlets

小图 (Graphlets) 是连接的非同构子图。它使我们能够获得节点级子图度量。Graphlet 度向量( Graphlet Degree Vector, GDV)是在每个轨道位置具有节点的频率向量，它计算节点接触的 Graphlet 的数量。GDV提供了节点的本地网络拓扑的度量。

### Finding Motifs and Graphlets

找到大小为 $k$ 的 Motifs 或 Graphlet 需要我们：
1）列举所有大小为 $k$ 的相连子图；
2）计算每种子图类型的出现次数。
仅知道图中是否存在某个子图是一个艰巨的计算问题。而且，计算时间随着 Motifs 或 Graphlet 的大小增加而呈指数增长。

### ESU Algorithm
精准子图枚举算法 (Exact Subgraph Enumeration, ESU) 包含两部分, $V_{subgraph}$ 包含当前构造的子图中的节点，而 $V_{extension}$ 是一组用于扩展 motif 的候选节点。  ESU的基本思想是首先从节点 v 开始，然后将节点 u 添加到$V_{extension}$ 当节点 u 的id大于节点 v 时，u 只能与某个新添加的节点 w 相邻，而不能与 $V_{subgraph}$ 中已经存在的任何节点相邻。

ESU是作为递归函数实现的，下图显示了此算法的伪代码：
<img src="./img/Exact_Subgraph_Enumeration.png?style=centerme" alt="Figure 3" style="zoom: 50%;" />

## Structural Roles in Networks

### Roles 

我们可以将角色 (Roles) 视为网络中节点的功能，并且可以通过结构行为对其进行度量。注意角色 (roles) 与组(groups)/社区(communities)不同。角色基于节点子集之间关系的相似性。具有相同角色的节点具有相似的结构属性，但是它们不必彼此直接或间接交互。组/社区是基于邻接关系，邻近性或可达性而形成的，同一社区中的节点之间相互连接良好。

### Structural equivalence
如果节点 u 和 v 之间具有相同的关系，我们说它们在结构上是等效的。结构上等效的节点可能以许多不同的方式相似。例如，节点 u 和 v 在下图中在结构上等效，因为它们以相同的方式连接其他节点。

<img src="./img/structurally_equivalent.png?style=centerme" alt="Figure 4" style="zoom: 33%;" />

### RoIX
角色使我们能够识别网络中节点的不同属性。在这里，我们将介绍一种称为 **RolX** 的自动结构角色发现方法。这是一种没有先验知识的无监督学习方法。下图是对 RoIX 方法的概述。

<img src="./img/RoIX.png?style=centerme" alt="Figure 5" style="zoom: 33%;" />

### Recursive Feature Extraction
递归特征提取的基本思想是聚合节点的特征，并使用它们生成新的递归特征。通过这种方式，我们可以将网络连接变成结构化的功能。

节点的邻域特征的基本集合包括：

1. 局部特征，它们都是节点度的度量。
2. Egonet功能是在节点的egonet上计算的，可能包括在egonet内的边数量以及进入/离开egonet的边数量。这里节点的egonet包括节点本身，其邻居和这些节点上的诱导子图中的任何边。

为了生成递归特征，首先我们从节点特征的基本集合开始，然后使用当前节点特征的集合来生成其他特征并重复。每次递归迭代时，可能的递归特征的数量呈指数增长。

