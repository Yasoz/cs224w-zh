## Message Passing and Node Classification

### Node Classification

<img src="/img/node_classification.png?style=centerme" alt="node_classification" style="zoom: 50%;" />

节点分类是在给定一组现有节点标签的情况下，将标签分配给图中的节点的过程。此相当于半监督学习过程。虽然可以在现实环境中收集每个节点的真实标签值，但是收集这些标签非常昂贵。因此，我们依靠随机抽样来获得这些标签。然后，我们使用标签的小样本来开发模型，从而为图中的节点生成可信赖的标签预测。

集体分类是一个描述我们如何为网络中的所有节点分配标签的笼统术语。我们在网络上传播这些标签中的信息，并尝试使每个节点分配的标签稳定。之所以可以实现这一任务，是因为网络具有特殊的属性，尤其是节点之间的相关性，这使得我们可以利用它们来构建分类器。本质上，集体分类依赖于马尔可夫假设（节点 $Y_i$ 的标签取决于该节点令居节点的标签），其数学形式为：
$$
P(Y_i|i)=P(Y_i|N_i)
$$
主要有三种方法用于分类：关系分类，迭代分类和信念分类，大致按这些方法的先进程度排序。

### Correlations in a Network

在网络环境中，各个行为是相关的。这些相关性通常由三种主要现象导致：趋同性，影响力和混淆性。 

<img src="/img/graph_correlations.png?style=centerme" alt="graph_correlations" style="zoom: 50%;" />

#### Homophily

*"Birds of a feather flock together"*（物以类聚，人以群分）

趋同性通常是指个体倾向于与其相似的他人交往和联系。例如，在社交网络中，相似之处可以包括各种属性，包括年龄，性别，组织隶属关系，品味等等。例如，喜欢同一网络的个人可能会更紧密地联系在一起，因为他们可能在音乐会上或者在音乐论坛中见面和互动。这种现象通常可以反映在我们的朋友关系中，如下图所示，*Easley and Kleinberg(2010* 在种族中通过簇来展示朋友关系。

<img src="/img/homophily.png?style=centerme" alt="homophily" style="zoom:50%;" />

此外，在政治生活中我们也可以看到这种趋势。个人倾向于根据他人的政治观点来选择朋友。

<img src="/img/homophily2.png?style=centerme" alt="homophily2" style="zoom:50%;" />

#### Influence

另一个可以证明网络可以展示相关性的因素是影响。在这种情况下，形成的连接和边会影响节点本身的行为。在一个社交网络中，每个人都可能受到他们的朋友的影响。例如，一个朋友可能会推荐一种你可能感兴趣的音乐，然后你也可以将该喜好传递给你的朋友。

#### Confounding

混杂变量会导致节点表现出相似的特征。例如，我们所处的环境可能会在多个方面影响我们的相似性，从我们的语言，我们的音乐品味到我们的政治偏好。

### Leveraging Network Correlations for Classification of Network Data

#### Guilt-by-association

如果一个节点连接到具有特定标签的另一个节点，根据马尔科夫假设，这两个节点更有可能具有相同的标签。例如，如果我的朋友都是意大利人，那么根据我们上面讨论的网络相关性，我自己也更可能是意大利人。这种方法在多个域中具有广泛的应用，比如可以用于例如区分恶意网页和良性网页。恶意网页往往会相互链接以提高曝光度和可信度，并且使之在搜索引擎中排名更高。

#### Performing guilt-by-association node classification

特定节点 X 是否属于特定标签可以取决于多种因素。在本文中，最常见的包括：

* X 的特点
* X 附近对象的标签
* X 附近对象的特征

但是，如果我们仅使用以上这些因素而不使用网络属性，那么我们只会在这些特征上训练简单的分类器。为了能够实现集体分类，我们还需要考虑网络拓扑。集体分类需要以下三个组成部分：

* 一个**局部分类器**来分配初始标签
  * 这个标准分类器将基于节点属性/特征预测标签，而无需网络信息。几乎可以在这里使用任何分类器，即使kNN的表现也相当不错。
* 一个**相关分类器**捕获网络中的节点之间相关性（趋同性，影响力）。
  * 该分类器根据其邻居的标签和特征预测节点的标签。
  * 这一步需要使用网络信息 
* **集体推理** propagates the correlations through the network. Basically, we do not want to stop at the level of only using our neighbors, but through multiple iterations we want to be able to spread the contribution of other neighbors to each other.
  * This is an iterative series of steps that applies the relational classifer to each node in succession, and iterates until the inconsistency between neighboring node labels is minimized, or until we have reached our maximum iterations and run out of computational resources.
  * Node structure has a profound impact on the final predictions.

There are numerous applications for collective classification, including: 

* Document Classification 
* Part of speech tagging
* Link prediction
* Optical character recognition
* Image/3D data segmentation
* Entity resolution in sensor networks
* Spam and fraud detection

#### Example:

For the following graph, we would like to predict labels on the unlabeled, beige nodes as either (+) or (-):

![example](../assets/img/example.png?style=centerme)

To make those predictions, we will use a *Probabilistic Relational Classifier*, the basic idea of which is that the class probability of $$Y_i$$ is a weighted average of the class probabilities of its neighbors. To initialize, we will use the ground-truth labels of our labeled nodes, and for the unlabeled nodes, we will initialize Y uniformly, for instance as $$P(Y_{unlabeled}) = 0.5$$--or if you have a prior that you trust, you can use that here. After initialization, you may begin to update all nodes, in random order, until convergence conditions or you have reached the maximum number of iterations. Mathematically, each repetition will look like this:
$$
P(Y_i= c) = \frac{1}{\vert N_i\vert}{\sum_{(i,j)\in E} W_{(i,j)}} \sum W_{(i,j)} P(Y_j = c)
$$

Where $$N_i$$ is the number of neighbors of *i* and *W* is the weighted edge strength from *i* to *j*. 

We will update the nodes in *random order* until we reach convergence or our maximum number of iterations. We do not have to update in random order, but it has been shown empirically that it works very well across many cases, so we suggest random ordering. We must remember, however, that our results *will* be influenced by the order of nodes, especially for smaller graphs (larger graphs are less sensitive to that).

It should be noted, however, that there are 2 additional caveats:

* Convergence is **not** guaranteed
* Model cannot use node feature information

## *Iterative Classification*

As mentioned in the previous section, relational classifiers do not use node attributes, and so in order to leverage them we use iterative classification which allows you to classify node i based not only on the labels of its neighbors, but on its own attributes in addition. This process consists of the following steps:

* Bootstrap phase
  * create a flat vector $$a_i$$  for each node *i*.
  * Train a local classifier, our baseline, $$f(a_i)$$, (e.g. SVM, kNN) using $$a_i$$.
  * Aggregate neighbors using count, mode, proportion, mean, exists, etc. We must determine the most sensical way to aggregate our futures.

* Iteration phase
  * Repeat for each node i:
    * Update node vector $$a_i$$
    * update our label assignment $$Y_i$$ to $$f(a_i)$$ which is a hard assignment
  * Iterate until class labels stabilize or max number of iterations is reach

This is very similar to what we did before with the relational classifier, the key difference being that we now use the feature vector and once again, convergence is not guaranteed. You can find a great, real world example of this [here](https://cs.stanford.edu/~srijan/pubs/rev2-wsdm18.pdf) .



## Message Passing/Belief Propagation

### Loopy Belief Propagation

Belief propagation is a dynamic programming technique that answers conditional probabiliy queries in a graphical model. It's an iterative process in which every neighbor variables *talk* to each other, by **passing messages.** 

![message_passing](../assets/img/message_passing.png?style=centerme)

For instance, I (variable $$x_1$$) might pass a message to you (variable $$x_2$$) stating that you belong in these states with these different likelihoods. The state of the node in question doesn't depend on the belief of the node itself, but on the belief of all the nodes surrounding it. 

What message node *i* ends up sending to node *j* ultimately depends on its neighbors, *k.* Each neighbor *k* will pass a message to *i* regarding its belief of the state of *i*, and then *i* will communicate to *j*.  

![message_passing2](../assets/img/message_passing2.png?style=centerme)



When performing belief propagation, we will need the following notation:

### **Notation**:

* Label-Label potential matrix $$\psi$$ represents the dependency between a node and its neighbor. $$\psi (Y_i, Y_j)$$ is simply equal to the probability of a node j being in state $$Y_j$$ given that its neighbor *i* is in state $$Y_i$$. We have been seing this with the other methods, we just formalize it here. It basically captures what is the correlation between node *i* and *j*.
* Prior belief $$\phi$$ or $$\phi (Y_i)$$ represents the probability of node i being in state $$Y_i$$. This is capturing the node features.
* m$$_{i\to j}(Y_j)$$ is the message from *i* to *j*, which represents i's estimate of j being in state  $$Y_j$$.
* $$\mathcal{L}$$ represents the set of all states.

Once we have all notation, we can compile this all together to give us the message that *i* will send to *j* for state $$Y_j$$.

 $$ m_i\to_j(Y_j) = \alpha \sum \psi (Y_i, Y_j) \phi_i(Y_i)\Pi_{k \in \mathcal{N}\backslash  j}  m_{k\to i}$$         $$         \forall \mathcal{L}$$

This equation summarizes our task: to calculate the message from i to j, we will sum over all of our states the label-label potential multiplied by our prior, multiplied by the product of all the messages sent by neighbors from the previous rounds. To initialize, we set all of our messages equal to 1.  Then, we  calculate our message from *i* to *j*, using the formula described above. We will repeat this for each node until we reach convergence, and then we can calculate our final assignment,  *i*'s belief of being in state $$Y_i$$, or $$b_i(Y_i)$$.

![belief_propagation](../assets/img/belief_propagation.png?style=centerme)

Belief propagation has many advantages. It's easy to program and easy to parallelize. Additionally, it's very general and can apply to any graphical model with any form of potentials (higher order pairwise). However, similar to the other techiques, convergence is once again, not guaranteed. This is particularly an issue when their are many closed loops. It should be noted that we may also learn our priors. 

A good example of Belief Propagation in action is [detection of online auction fraud](http://www.cs.cmu.edu/~christos/PUBLICATIONS/netprobe-www07.pdf). 

