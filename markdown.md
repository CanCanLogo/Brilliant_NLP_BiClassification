# 实践一 酒店评论情绪正负性预测  

# 实验一：逻辑回归法  

1.原理：本实验是一个二分类问题，最简单的方式是线性回归 $\boldsymbol { y } = \boldsymbol { f } \left( \boldsymbol { w } \right) = \boldsymbol { W } ^ { T } \boldsymbol { x }$ ，将词向量作为输入 $\mathsf { x } ,$ 进行加权和偏置后得到预测值。但是这样就需要使用下式来二分类  
$$
\stackrel { \iota } { y } = \{ { 1 , } f ( x ) { > } 0 . 5
$$

这样的话函数不连续，无法进行求导和反向传播，所以引入sigmoid 函数。  

$$
  \boldsymbol { y }= \{ \begin{array} { c } { 1 , \boldsymbol { f } \left( \boldsymbol { x } \right) > 0 . 5 } \\ { 0 , \boldsymbol { f } \left( \boldsymbol { x } \right) < 0 . 5 } \end{array} \} , \ \boldsymbol { y } = \frac { 1 } { 1 + e ^ { - \left( \boldsymbol { w } ^ { T } \boldsymbol { x } + b \right) } }
$$

得： $\ln { \frac { y } { 1 - y } } = w ^ { T } x + b$ 由几率 $\ln { \big ( } { \mathrm { ~ o d d s ~ } } { \big ) } = \ln { \frac { y } { 1 - y } }$ ，可知 $P \left( \left. Y = 1 \right| x \right) = \frac { 1 } { 1 + e ^ { - \left( w ^ { T } x + b \right) } }$ 模型：建立一层神经网络，输入维度为24000 即 $3 0 0 * 8 0$ ，即词向量维度和句子定长，输出维度为1，含义是可归类为1的概率 $P ( \boldsymbol { Y = 1 } | \boldsymbol { x } ) _ { \circ }$  

正向传播为 $y = g ( f ( x ) )$ ，其中 $ f = ( \sum _ { i = 1 } ^ { n } \bigsqcup _ { \scriptstyle i } X _ { i } + b \big ) $ ，$g$为 sigmoid 函数。  

损失函数采用交叉熵 $L { = } { - } [ y \log { \left( { \mathit { p } } \right) } { + } ( 1 { - } y ) \log { \left( 1 { - } { \mathit { p } } \right) } ] _ { \circ }$ 反向传播梯度为 $d \omega _ { i } = \left[ \sigma \left( \omega x _ { i } + b \right) - y \right] x _ { i } = \left( p - y \right) x _ { i } , \ d b = \left( p - y \right)$  

2.数据平台工具：数据集为酒店评论数据集，已经进行标注，训练集比例为$80 \text{‰}$ 。  

实验在 pycharm2022.2.4上进行，安装了 pytorch3.9以及其他必要的库。数据需要进行词向量化处理，首先使用jieba 库进行中文分词，然后去掉停用词，使用库加载微博W eibo 预训练词向量作为字典，然后将酒店评论预料进行转g ensim化，其中，句子定长80 词向量维度300，训练过程采用词向量拼接而非求和。  

3.实验过程：设置超参数，学习率 0.01， BATCH_SIZE = 8， EPOCHS = 10    

4.训练结果：测试集准确率 $7 8 \%$ ,但测试集loss 收敛值远大于训练集，有待改进。

![image-20250516154442848](markdown.assets/image-20250516154442848.png)  

# 实验二：朴素贝叶斯法  

# 1.原理：  

朴素贝叶斯法是基于贝叶斯定理与特征条件独立性假设的分类方法。已知贝叶斯定理为后验概率等于先验概率乘以调整因子（似然概率除以 A 事件先验概$P \left( B \mid A \right) = { \frac { P \left( B \right) P \left( A \mid B \right) } { P \left( A \right) } } ,$  

但是特征 $A$ 相互之间也有条件概率，并不是独立分布的，在NLP 问题中，可以理解为：每一个词相互之间都有上下文关系，并不是独立出现的。如果需要考虑这个问题的话，应当采用概率有向图模型。  

而朴素贝叶斯为了简化这一问题，假设各个特征之间相互独立，基于该假设学习输入输出的联合概率分布；然后基于此模型，对给定的输入 x，利用贝叶斯定理求出后验概率最大的输出 $y _ { \circ }$  

2.模型：统计训练集，获得条件概率：P（词|积极），P(词|消极)。模型的公式，以积极为例，为：  
$$
P ( 积极 | 句子) = \frac { P \left(积极 \right) \prod P \left( \left. 词 \right| 积极 \right) } { \prod P \left( 词 \right) }
$$

先验概率 $P \left( 积极\right) = 1 / 2$ ，由于最后直接比较积极和消极，所以可以省去$\Pi { \cal P } \big ( 词 )$ 所以我们可以直接比较∏P(词∣积极)和∏P(词∣消极)的大小来进行分类。  

3.数据：不必转化词向量，对数据集进行分词操作后，划分训练集比例 $8 0 \%$ ，然后对积极和消极数据分别处理得到存储有P（词|积极），P(词|消极)的字典。  

4.实 验 过 程 和 结 果 ： 制 作 测 试 集 词 列 表 。 对 于 每 一 条 评 论 ，∏P (词∣积极)>∏P(词∣消极)则预测为积极，反之则预测为消极若词不存在  

与词频表，则P (词∣情绪)设置为0.00001，准确率 $8 3 . 2 \%$ ，模型准确率较高。  

![image-20250516154454703](markdown.assets/image-20250516154454703.png)