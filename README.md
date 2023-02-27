## Assignment 1-Parsing and POS Tagging
数据集分别提供了corpu.txt（已经分好词并使用空格隔开，用于中文分词）和corpus_POS.txt（在分词的基础上附加词性标记，用于词性标注），将二者分别按照4:1随机划分为训练集和验证集

在corpu.txt上实现并评测双向最大匹配算法

配置并使用Python版本的jieba对上述语料进行分词和评测，并与自行实现的双向最大匹配算法的结果进行比较

在corpu_POS.txt的训练集中统计出词性的初始概率分布向量、词性标签之间的转移概率矩阵和词性到词的发射概率矩阵，建立隐马尔可夫模型

在验证集上进行验证，并使用sklearn.metrics模块对得到的混淆矩阵进行评估，评价指标为精确率、召回率、F1值

### Description
- 请阅读两个main文件中的主体程序，补充完成 `./util/cut_util.py`, `./util/tag_util.py`, `./util/viterbi.py`
- 运行 `python -u main_cut.py`实现分词，`python -u main_tag.py` 实现词性预测
- 分词部分，请说明你实现的最大匹配算法`maximum_match_cut`函数的原理及时间复杂度。可以阅读`./util/trie.py`，进一步思考如何降低复杂度。
