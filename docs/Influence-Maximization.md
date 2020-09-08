### Motivation
识别网络中有影响力的节点具有重要的实际用途。一个很好的例子是“病毒式营销”，该策略利用现有的社交网络来传播和推广产品。精心设计的病毒标记产品将识别出最具影响力的客户，说服他们采用并认可该产品，然后像病毒一样在社交网络中传播该产品。

影响力最大化的关键是如何找到最具影响力的节点集？为了回答这个问题，我们首先来看两个经典的级联模型：

- 线性阈值模型
- 独立级联模型

然后，我们将开发一种方法来找到独立级联模型中最具影响力的节点集。

### Linear Threshold Model
在线性阈值模型中，我们具有以下设定：

- 一个节点 $v$ 具有随机阈值 $$\theta_{v} \sim U[0,1]$$
- 一个节点 $v$ 受每个邻居 $w$ 的影响 ，基于节点 $v$ 和 $w$ 之间的权重 $b_{v,w}$，且

$$
\sum_{w\text{ neighbor of v }} b_{v,w}\leq 1
$$

- 一个节点 vv 当至少 θvθv其邻居的一部分是活跃的。那是
- 当节点 $v$ 至少 $\theta_{v}$ 的邻居活跃时，节点 $v$ 才活跃，即，

$$
\sum_{w\text{ active neighbor of v }} b_{v,w}\geq\theta_{v}
$$

下图演示了该过程：

![linear_threshold_model_demo](../docs/img/influence_maximization_linear_threshold_model_demo.png?style=centerme)

(A) 节点 V 被激活，且对 W 和 U 的影响分别为0.5和0.2 is； (B) W 被激活，对 X 和 U 的影响分别为0.5和0.3； (C) U 被激活并分别以0.1和0.2影响 X和 Y；(D) X 被激活并以0.2影响 Y，此时不能再激活任何节点；过程停止。

### Independent Cascade Model
在此模型中，我们根据有向图中的概率对节点的影响（激活）进行建模：

- 给定有限图 $$G=(V, E)$$
- 从新的行为开始（例如采用新产品，我们说它们是活跃的）给定一个节点集  $$S$$ 
- 每条边 $$(v, w)$$ 具有概率 $$p_{vw}$$
- 如果节点 $$v$$ 被激活, 则有机会利用概率 $p_{vw}$ 1去激活节点 $$w$$ 
- 通过网络传播激活

注意:

- 没个边仅建立一次
- 如果 $u$ 和 $v$ 都处于激活状态并且与 $w$ 相连，哪个节点先去激活 $w$ 并不重要

### Influential Maximization (of the Independent Cascade Model)

#### Definitions
- **最具影响力的集合大小 $$k$$** ($$k$$ 用户定义参数) 是包含 $k$ 个节点的集合 $S$ 。这些节点如果被激活则会产生最大预期级联大小 $$f(S)$$。[为什么是“预期的级联大小”？由于独立级联模型的随机性，节点激活是一个随机过程，因此，$$f(S)$$ 是一个随机变量。在实践中，我们通常计算许多的随机模拟从而获得期望值 $$f(S)=\frac{1}{\mid I\mid}\sum_{i\in I}f_{i}(S)$$，其中 $$I$$ 表示一组模拟]
- **节点 $u$ 的影响集 $$X_{u}$$** 是最终将被节点 $u$ 激活的节点集合，示例如下所示

<img src="./img/influence_maximization_influence_set.png?style=centerme" alt="influence_set" style="zoom:67%;" />

红色节点 $a$ 和 $b$ 都处于激活状态，两个绿色的区域分别表示由节点 $a$ 和 $b$ 激活的节点集，比如 $$X_{a}$$ 和 $$X_{b}$$。

注意：
- $$f(S)$$ 是集合 $$X_{u}$$ 的并集，即: $$f(S)=\mid\cup_{u\in S}X_{u}\mid$$。
- 如果 $$f(S)$$ 越大，表示集合 $S$ 更具有影响力

#### Problem Setup
The influential maximization problem is then an optimization problem:

$$
\max_{S \text{ of size }k}f(S)
$$

This problem is NP-hard [[Kempe et al. 2003]](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf). However, there is a greedy approximation algorithm--**Hill Climbing** that gives a solution $$S$$ with the following approximation guarantee:

$$
f(S)\geq(1-\frac{1}{e})f(OPT)
$$

where $$OPT$$ is the globally optimal solution.

### Hill Climbing
**Algorithm:** at each step $$i$$, activate and pick the node $$u$$ that has the largest marginal gain $$\max_{u}f(S_{i-1}\cup\{u\})$$:

- Start with $$S_{0}=\{\}$$
- For $$i=1...k$$
  - Activate node $$u\in V\setminus S_{i-1}$$ that $$\max_{u}f(S_{i-1}\cup\{u\})$$
  - Let $$S_{i}=S_{i-1}\cup\{u\}$$

**Claim:** Hill Climbing produces a solution that has the approximation guarantee $$f(S)\geq(1-\frac{1}{e})f(OPT)$$.

### Proof of the Approximation Guarantee of Hill Climbing
**Definition of Monotone:** if $$f(\emptyset)=0$$ and $$f(S)\leq f(T)$$ for all $$S\subseteq T$$, then $$f(\cdot)$$ is monotone.

**Definition of Submodular:** if $$f(S\cup \{u\})-f(S)\geq f(T\cup\{u\})-f(T)$$ for any node $$u$$ and any $$S\subseteq T$$, then $$f(\cdot)$$ is submodular.

**Theorem [Nemhauser et al. 1978]:**{% include sidenote.html id='note-nemhauser-theorem' note='also see this [handout](http://web.stanford.edu/class/cs224w/handouts/CS224W_Influence_Maximization_Handout.pdf)' %} if $$f(\cdot)$$ is **monotone** and **submodular**, then the $$S$$ obtained by greedily adding $$k$$ elements that maximize marginal gains satisfies

$$
f(S)\geq(1-\frac{1}{e})f(OPT)
$$

Given this theorem, we need to prove that the largest expected cascade size function $$f(\cdot)$$ is monotone and submodular.

**It is clear that the function $$f(\cdot)$$ is monotone based on the definition of $$f(\cdot)$${% include sidenote.html id='note-monotone' note='If no nodes are active, then the influence is 0. That is $$f(\emptyset)=0$$. Because activating more nodes will never hurt the influence, $$f(U)\leq f(V)$$ if $$U\subseteq V$$.' %}, and we only need to prove $$f(\cdot)$$ is submodular.**

**Fact 1 of Submodular Functions:** $$f(S)=\mid \cup_{k\in S}X_{k}\mid$$ is submodular, where $$X_{k}$$ is a set. Intuitively, the more sets you already have, the less new "area", a newly added set $$X_{k}$$ will provide.

**Fact 2 of Submodular Functions:** if $$f_{i}(\cdot)$$ are submodular and $$c_{i}\geq0$$, then $$F(\cdot)=\sum_{i}c_{i} f_{i}(\cdot)$$ is also submodular. That is a non-negative linear combination of submodular functions is a submodular function.

**Proof that $$f(\cdot)$$ is Submodular**: we run many simulations on graph G (see sidenote 1). For the simulated world $$i$$, the node $$v$$ has an activation set $$X^{i}_{v}$$, then $$f_{i}(S)=\mid\cup_{v\in S}X^{i}_{v}\mid$$ is the size of the cascades of $$S$$ for world $$i$$. Based on Fact 1, $$f_{i}(S)$$ is submodular. The expected influence set size $$f(S)=\frac{1}{\mid I\mid}\sum_{i\in I}f_{i}(S)$$ is also submodular, due to Fact 2. QED.

**Evaluation of $$f(S)$$ and Approximation Guarantee of Hill Climbing In Practice:** how to evaluate $$f(S)$$ is still an open question. The estimation achieved by simulating a number of possible worlds is a good enough evaluation [[Kempe et al. 2003]](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf):

- Estimate $$f(S)$$ by repeatedly simulating $$\Omega(n^{\frac{1}{\epsilon}})$$ possible worlds, where $$n$$ is the number of nodes and $$\epsilon$$ is a small positive real number
- It achieves $$(1\pm \epsilon)$$-approximation to $$f(S)$$
- Hill Climbing is now a $$(1-\frac{1}{e}-\epsilon)$$-approximation

### Speed-up Hill Climbing by Sketch-Based Algorithms

**Time complexity of Hill Climbing**

To find the node $$u$$ that $$\max_{u}f(S_{i-1}\cup\{u\})$$ (see the algorithm above):

- we need to evaluate the $$X_{u}$$ (the influence set) of each of the remaining nodes which has the size of $$O(n)$$ ($$n$$ is the number of nodes in $$G$$)
- for each evaluation, it takes $$O(m)$$ time to flip coins for all the edges involved ($$m$$ is the number of edges in $$G$$)
- we also need $$R$$ simulations to estimate the influence set ($$R$$ is the number of simulations/possible worlds)

We will do this $$k$$ (number of nodes to be selected) times. Therefore, the time complexity of Hill Climbing is $$O(k\cdot n \cdot m \cdot R)$$, which is slow. We can use **sketches** [[Cohen et al. 2014]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/08/skim_TR.pdf) to speed up the evaluation of $$X_{u}$$ by reducing the evaluation time from $$O(m)$$ to $$O(1)$${% include sidenote.html id='note-evaluate-influence' note='Besides sketches, there are other proposed approaches for efficiently evaluating the influence function: approximation by hypergraphs [[Borgs et al. 2012]](https://arxiv.org/pdf/1212.0884.pdf), approximating Riemann sum [[Lucier et al. 2015]](https://people.seas.harvard.edu/~yaron/papers/localApproxInf.pdf), sparsification of influence networks [[Mathioudakis et al. 2011]](https://chato.cl/papers/mathioudakis_bonchi_castillo_gionis_ukkonen_2011_sparsification_influence_networks.pdf), and heuristics, such as degree discount [[Chen et al. 2009]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf).'%}.

**Single Reachability Sketches**

- Take a possible world $$G^{i}$$ (i.e. one simulation of the graph $$G$$ using the Independent Cascade Model)
- Give each node a uniform random number $$\in [0,1]$$
- Compute the **rank** of each node $$v$$, which is the **minimum** number among the nodes that $$v$$ can reach in this world.

*Intuition: if $$v$$ can reach a large number of nodes, then its rank is likely to be small. Hence, the rank of node $$v$$ can be used to estimate the influence of node $$v$$ in $$G^{i}$$.*

However, influence estimation based on Single Reachability Sketches (i.e. single simulation of $$G$$ ) is inaccurate. To make a more accurate estimate, we need to build sketches based on many simulations{% include sidenote.html id='note-sketches' note='This is similar to take an average of $$f_{i}(S)$$ in sidenote 1, but in this case, it is achieved by using Combined Reachability Sketches.' %}, which leads to the Combined Reachability Sketches.

**Combined Reachability Sketches**

In Combined Reachability Sketches, we simulate several possible worlds and keep the smallest $$c$$ values among the nodes that $$u$$ can reach in all the possible worlds.

- Construct Combined Reachability Sketches:

  - Generate a number of possible worlds
  - For node $$u$$, assign uniformly distributed random numbers $$r^{i}_{v}\in[0,1]$$ to all $$(v, i)$$ pairs, where $$v$$ is the node in $$u$$'s reachable nodes set in the world $$i$$.
  - Take the $$c$$ smallest $$r^{i}_{v}$$ as the Combined Reachability Sketches

- Run Greedy for Influence Maximization:
  - Whenever the greedy algorithm asks for the node with the largest influence, pick node $$u$$ that has the smallest value in its sketch.
  - After $$u$$ is chosen, find its influence set $$X^{i}_{u}$$, mark the $$(v, i)$$ as infected and remove their $$r^{i}_{v}$$ from the sketches of other nodes.

Note: using Combined Reachability Sketches does not provide an approximation guarantee on the true expected influence but an approximation guarantee with respect to the possible worlds considered.
