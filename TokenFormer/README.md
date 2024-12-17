过去， 若要扩展一个Transformer模型，通常做法是增加模型中线性投影层（linear
projection）的通道维度（channel dimension）。然而，一旦维度发生变化，就需要对该线性投影层进行重新训练。这种扩展往往涉及整个模型的全部线性投影层，这意味着几乎整个模型的参数都需要重新训练，这种扩展模型方式的成本是高昂的。
为了解决这一问题，提出TokenFormer，一种具备良好可扩展性的架构。通过TokenFormer，即使在扩展模型参数后，也能够采取增量训练的方式，而非从零开始重新训练整个模型，从而大大降低了扩展成本。

---
下图左侧展示了逐步增加TokenFormer与传统Transformer模型参数量时的计算成本对比。随着横坐标的向右延伸，表示计算成本逐渐增大。当两个模型的参数量增加至同一水平时，代表TokenFormer的红点始终位于代表Transformer的蓝点左侧，这表明在扩展相同参数量的情况下，TokenFormer的计算成本更低。
下图右侧则对比了Transformer与TokenFormer的结构差异。

![](/home/lihp/hub/AINotes/TokenFormer/images/0.png)

在transformer中，input经过QKV Projecttion（3个并列的linear层）映射成QKV，然后QKV之间进行attention计算，最后再经过FFN（两个linear层）得到output。
在tokenformer中，QKV Projecttion和FFN不见了，取而代之的是Token-Param Attn层，本质上是linear层被替换成Token-Param Attn。在Token-Param Attn中，Key Param和Value Param是模型的参数，也即模型的参数以keys和values的形式存在。与transformer的keys和values的不同之处在于，这些keys和values不是数据，而是可以学习的参数。
众所周知，注意力模块中的QKV都是序列，长度是可变的，也就是说Key Param和Value Param的长度是可变的。假设已经训练了一个模型，Key Param和Value Param的序列长度为L，现在要对模型参数扩展一倍，只需在Key Param和Value Param序列的末尾上再加一截（长度也为L）参数就可以了。只有新增的参数是全新的，要从头训练，旧的参数可以只做微调甚至冻结。
这就是Tokenformer可扩展性较好的本质原因。

---
下图对tokenformer进行更详细地描述。
Token-Param Attn被简写成Pattention了。
input经过Pattention计算QKV的时候，其实是经过了三个并列的Pattention。
图右上部分展示input tokens是如何与key param、value param融合计算得到output tokens的。
右下部分展示了tokenformer的参数复用机制。
![](/home/lihp/hub/AINotes/TokenFormer/images/1.png)

