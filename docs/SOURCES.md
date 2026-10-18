# 在线核对与独立实现边界

核对日期：2026-10-18。未下载或集成完整仓库，未复制上游代码；下面机制用独立 Python/PyTorch 实现。本项目不是严格 NDS 复现，无其性能保证。

## 官方论文

André Hottung, Paula Wong-Chung, Kevin Tierney，*Neural Deconstruction Search for Vehicle Routing Problems*，TMLR 2025。

- [正式论文页](https://openreview.net/forum?id=bCmEP1Ltwq)
- [作者论文 arXiv v2](https://arxiv.org/html/2501.03715v2)，重点 §4.1–4.4。

采用的机制：以实例和当前路线结构编码节点；自回归决定删除序列；该序列控制贪心重插入顺序；完整修复后反馈；策略进入迭代搜索。未沿用其 latent、多 rollout 并行 ASA 和优化后端。本项目不继承论文结果。

## 官方源码快照

[官方仓库](https://github.com/ahottung/NDS)，在线 main 对应 commit `ca374200f70f1954ec82348d6c4e92ad4aff3168`。

- [trainer.py](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/trainer.py)：核对 `_compute_reward` / `_compute_policy_loss`。连续奖励路径截断负收益，均值 advantage 后 top-1 过滤；本项目改为未截断相对收益、leave-one-out、全 rollout。
- [model.py](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/model.py)：核对 GRU 历史状态和 masked pointer。这里只采用一般机制，编码层与参数独立实现，无 latent。
- [env.py](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/env.py)：核对 depot/已选节点 mask 与固定删除数。
- [search_sa.py](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/search_sa.py)：核对温度接受与时间停止；本项目采用简单严格改善接受，超时候选不计收益。
- [Operations.cpp](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/cpp/cvrp/Operations.cpp)：核对首个修复保持动作序列，后续尝试可随机重排；本项目固定单次修复、不随机重排。在线读取内容 SHA256：`06a08616bbcca707b47715115910c439380a5f7bcc38c38d6624170d5ac89f8e`。
- [Solution.cpp](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/src/cpp/cvrp/Solution.cpp)：核对插入增量/容量检查，并注意上游 beta 随机跳过等选项。这里无 beta，平局规则显式固定。内容 SHA256：`395e023475efbefadbe82353da136012c2725884bbfbf866bb14938bac50d099`。
- [LICENSE](https://github.com/ahottung/NDS/blob/ca374200f70f1954ec82348d6c4e92ad4aff3168/LICENSE)：MIT，copyright 2025 André Hottung。未复制源码，所以没有随包分发第三方代码；后续若复制，必须保留原 copyright 和许可文本。

论文与源码有表述/选项差异，以此记录实际读到的路径，不推断全部配置行为。纯 Python 贪心、标准 REINFORCE/GRU/attention 的独立实现与上游在奖励、rollout 使用、搜索接受、数值精度和计算性能上均有差异。
