### Spectral Clustering

在这部分，我们研究了谱方法的一个重要类别,从而在全局层次内理解网络。“谱”是指从图得出的矩阵的谱或特征值，这可以使我们深入了解图本身的结构。特别是在理解探索频谱聚类算法时，该算法利用这些工具对图中的节点进行聚类。

频谱聚类算法通常包括三个基本阶段。

1. 预处理：构造图的矩阵表示形式，例如邻接矩阵（但我们将探索其他选项）
2. 分解：计算矩阵的特征向量和特征值，并使用它们创建低维表示空间
3. 分组：根据集群在该空间中的表示将点分配给集群

### Graph Partitioning

让我们公式化我们要解决的任务。我们从无向图 $G(V, E)$ 开始。我们的目标是用某种方法将 $V$ 分为两个不相交的组 $A, B$ (即 $A \cap B = \emptyset$  且 $A \cup B = V$) ，使组内部的连接数最大化，并使两个组之间的连接数最小。

为了进一步公式化目标，下面介绍一些术语:

- **Cut(割)**: 表示两个不相交的节点集之间有多少连接。 $cut(A, B) = \sum_{i \in A, j \in B} w_{ij}$ 其中 $$w_{ij}$$ 是节点 $i$ 和 $j$ 之间边的权重。
- **Minimum cut(最小割):**  $\arg \min_{A, B} cut(A, B)$

由于我们要尽量减少 $A$ 和 $B$ 之间的连接数, 我们可能会决定以**最小割**为我们的优化目标。但是我们发现这种方式最终会产生非常不直观的集群——我们通常可以简单地设置 $A$ 为一个几乎没有传出连接的单节点，$B$ 为网络中的其它部分，从而获得一个很小的**割**。而我们需要的是一种衡量内部集群连接性的方法。

引入**传导性**(**conductance**)可以平衡组内和组间连接性的问题。我们定义传导性为 $\phi(A, B) = \frac{cut(A, B)}{min(vol(A), vol(B))}$ 其中$vol(A) = \sum_{i \in A} k_i$ 是节点 $A$ 的总（加权）度。可以粗略地认为传导性类似于表面积与体积之比：分子为 $A$ 和 $B$ 共享曲面的面积，同时分母努力确保 $A$ 和 $B$ 之间具有相似的体积。由于采取这种措施 ，选择 $A$ 和 $B$ 并且最小化它们的传导性，相比最小化割具有更均衡的分区。 由于要最大程度地减小电导是一个NP-hard问题，因此面临的挑战是如何有效地找到一个良好的分区。

### Spectral Graph Partitioning

频谱图分区是一种允许我们使用特征向量确定传导性的方法。我们将从介绍频谱图理论的一些基本技术开始。

频谱图理论的目的是分析代表图形的矩阵的“频谱”。所谓频谱是指表示图的矩阵，按照其幅值大小排序及其对应的特征值 $\lambda_{i}$ 的集合 $\Lambda = \{\lambda_1, \ldots, \lambda_n\}$ 。比如d-正则图的邻接矩阵的最大特征向量/特征值对是全一向量  $x = (1, 1, \ldots, 1)$, 并且特征值 $\lambda = d$。练习：具有两个分量（每个分量为d-regular）的不连续图的特征向量是什么？注意，根据谱定理，邻接矩阵（是实数和对称的）具有正交特征向量的完整谱。

我们可以使用频谱图理论分析哪些矩阵？

1. 邻接矩阵: 由于该矩阵与图结构直接相关，因此它是一个很好的切入点。它还具有对称的重要特性，这意味着它具有完整的实值正交特征向量谱
2. 拉普拉斯矩阵 $L$: 定义 $L = D - A$ ，其中 $D$ 是对角矩阵， $D_{ii}$ 表示节点 $i$ 的度。 $A$ 是图的邻接矩阵。拉普拉斯矩阵使我们离图的直接结构更远，但是又具有一些有趣的特性，这些特性使我们更加关注于图的更深层次结构方面的内容。我们注意到，全1向量是特征值为0的拉普拉斯矩阵的特征向量。最后，由于 $L$ 是半正定的, 这意味着它有三个等效条件：它的特征值都是非负的，对于某些矩阵 $N$ 有$L = N^T N$ 并且对于每个向量 $x$ 有 $x^T Lx \geq 0$ 。这个属性使我们可以使用线性代数工具来理解 $L$ ，从而理解原始图。

特别的， $\lambda_2$ 作为 $L$ 第二小的特征值，对它的研究使我们在理解图聚类方面取得了长足的进步。根据瑞利商理论，我们有$\lambda_2 = \min_{x: x^T w_1 = 0} \frac{x^T L x}{x^T x}$ 其中 $w_1$ 是特征值 $\lambda_1$ 对应的特征向量；换句话说，我们将向量子空间中与第一个特征向量正交的目标最小化，以便找到第二个特征向量，($L$ 是对称的，因此具有特征值的正交基)。在高层次上，瑞利商将特征向量搜索构架为一个优化问题，使我们可以运用优化技术。注意，目标值并不依赖于 $x$ 的大小，因此可以将其大小限制为1。另外请注意我们知道的 $L$ 的第一个向量 是特征值为0的全为一的向量。所以说 $x$ 正交于这个向量等于说 $\sum_i x_i = 0$。

使用 $L$ 的这些属性和定义可以写出对于 $\lambda_2$ 更具体的公式：
$$
\lambda_2 = \min_x \frac{\sum_{(i, j) \in E} (x_i - x_j)^2}{\sum_i x_i^2} \\
\text{subject to} \quad \sum_i x_i = 0
$$
如果我们另外限制 $x$ 为单位长度，目标函数将会转换为 $\min_x \sum_{(i, j) \in E} (x_i - x_j)^2$.

$\lambda_2$ 与我们找到图的最佳分割的最初目标有何关系？让我们将分区 $(A,B)$ 表示为向量 $y$ ，并且 $y_i = 1$ if $i \in A$ and $y_i = -1$ if $i \in B$。 我们先尝试在执行分区大小平衡问题 ($\vert A\vert = \vert B\vert$) 的同时尽量减少割，而不是使用传导性，这就相当于 $\sum_{i}y_{i}=0$。基于这个大小限制，可以最小化分区的割。比如寻找 $y$ 最小化 $\sum_{(i, j) \in E} (y_i - y_j)^2$ ， $y$ 的值必须是 $+1$ 或者 $-1$ ，这样会使得 $y$ 的长度是固定的。这个优化问题看起来很像 $\lambda_2$ 的定义，事实上根据上述发现，我们可以通过最小化拉普拉斯矩阵的 $\lambda_2$ 达成这一目标，并且最佳聚类 $y$ 由其对应的特征向量（称为Fiedler向量）给出。

Now that we have a link between an eigenvalue of $$L$$ and graph partitioning, let's push the connection further and see if we can get rid of the hard $$\vert A\vert = \vert B\vert$$ constraint -- maybe there is a link between the more flexible conductance measure and $$\lambda_2$$. Let's rephrase conductance here in the following way: if a graph $$G$$ is partitioned into $$A$$ and $$B$$ where $$\vert A\vert \leq \vert B\vert$$, then the conductance of the cut is defined as $$\beta = cut(A, B)/\vert A\vert$$. A result called the Cheeger inequality links $$\beta$$ to $$\lambda_2$$: in particular, $$\frac{\beta^2}{2k_{max}} \leq \lambda_2 \leq 2\beta$$ where $$k_{max}$$ is the maximum node degree in the graph. The upper bound on $$\lambda_2$$ is most useful to us for graph partitioning, since we are trying to minimize the conductance; it says that $$\lambda_2$$ gives us a good estimate of the conductance -- we never overestimate it more than by a factor of 2! The corresponding eigenvector $$x$$ is defined by $$x_i = -1/a$$ if $$i \in A$$ and $$x_j = 1/b$$ if $$i \in B$$; the signs of the entries of $$x$$ give us the partition assignments of each node.

# Spectral Partitioning Algorithm

Let's put all our findings together to state the spectral partitioning algorithm.

1. Preprocessing: build the Laplacian matrix $$L$$ of the graph
2. Decomposition: map vertices to their corresponding entries in the second eigenvector
3. Grouping: sort these entries and split the list in two to arrive at a graph partition

Some practical considerations emerge.

- How do we choose a splitting point in step 3? There's flexibility here -- we can use simple approaches like splitting at zero or the median value, or more expensive approaches like minimizing the normalized cut in one dimension.
- How do we partition a graph into more than two clusters? We could divide the graph into two clusters, then further subdivide those clusters, etc (Hagen et al '92)...but that can be inefficient and unstable. Instead, we can cluster using multiple eigenvectors, letting each node be represented by its component in these eigenvectors, then cluster these representations, e.g. through k-means (Shi-Malik '00), which is commonly used in recent papers. This method is also more principled in the sense that it approximates the optimal k-way normalized cut, emphasizes cohesive clusters and maps points to a well-separated embedded space. Furthermore, using an eigenvector basis ensures that less information is lost, since we can choose to keep the (more informative) components corresponding to bigger eigenvalues.
- How do we select the number of clusters? We can try to pick the number of clusters $$k$$ to maximize the **eigengap**, the absolute difference between two consecutive eigenvalues (ordered by descending magnitude).

# Motif-Based Spectral Clustering

What if we want to cluster by higher-level patterns than raw edges? We can instead cluster graph motifs into "modules". We can do everything in an analogous way. Let's start by proposing analogous definitions for cut, volume and conductance:

- $$cut_M(S)$$ is the number of motifs for which some nodes in the motif are in one side of the cut and the rest of the nodes are in the other cut
- $$vol_M(S)$$ is the number of motif endpoints in $$S$$ for the motif $$M$$
- We define $$\phi(S) = cut_M(S) / vol_M(S)$$

How do we find clusters of motifs? Given a motif $$M$$ and graph $$G$$, we'd like to find a set of nodes $$S$$ that minimizes $$\phi_M(S)$$. This problem is NP-hard, so we will again make use of spectral methods, namely **motif spectral clustering**:

1. Preprocessing: create a matrix $$W^{(M)}$$ defined by $$W_{ij}^{(M)}$$ equals the number of times edge $$(i, j)$$ participates in $$M$$.
2. Decomposition: use standard spectral clustering on $$W^{(M)}$$.
3. Grouping: same as standard spectral clustering

Again, we can prove a motif version of the Cheeger inequality to show that the motif conductance found by our algorithm is bounded above by $$4\sqrt{\phi_M^*}$$, where $$\phi_M^*$$ is the optimal conductance.

We can apply this method to cluster the food web (which has motifs dictated by biology) and gene regulatory networks (in which directed, signed triads pla