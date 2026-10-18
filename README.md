# 关系辅助的自回归破坏与贪心修复

独立实现的 CVRP 实验系统，不依赖完整 NDS 仓库，不使用旧 Point/MLP/checkpoint。无辅助监督组称“自回归破坏＋贪心修复基线”。NDS 仅作为方法参考，出处和差异见 [SOURCES](docs/SOURCES.md)。

当前实现包括结构编码器、GRU 自回归删除、固定贪心修复、完整序列的单步 REINFORCE、真实/打乱关系辅助损失、checkpoint 恢复、无辅助头部署、单步和完整搜索评价。已有数据契约与原测试保留。没有进行真实研究实验。

先读 [算法与语义](docs/DESIGN.md)，运行步骤见 [服务器手册](docs/SERVER.md)，本地结果见 [验证记录](docs/VALIDATION.md)。神经部分需要用户已有的 PyTorch 环境；程序不安装依赖、不下载模型数据、不编译 C++。

## 源码

| 模块 | 用途 |
|---|---|
| contracts.py | 实例/参考合法性、成本重算、hash、关系和置换 |
| vrp.py | 删除 mask、固定贪心插入、普通输入白名单 |
| neural.py | 结构编码器、GRU pointer、对称关系头、移除辅助头 |
| training.py | 全 rollout LOO REINFORCE、梯度诊断、完整恢复状态 |
| search.py | 严格改善搜索、固定迭代/时间预算与超时统计 |
| evaluation.py | 无参考部署、评价、实例配对及 bootstrap |
| references.py | 可选的多起点插入＋2-opt 教师，非 HGS，质量须另验 |
| runtime.py / experiment.py | 配置 gate、只读 preflight、manifest 和独立 CLI |

## 最小合成验证

在工作区根目录运行，无依赖安装：

```bash
python -B -m unittest discover -s tests -v
```

无 PyTorch 时神经测试显式 skip，不能视为全链路通过。已有 PyTorch 的服务器必须执行：

```bash
python -B scripts/verify.py --require-neural
```

合成样例创建与只读检查：

```bash
python -B -m relation_lab.experiment make-smoke --run-dir smoke_fixture
python -B -m relation_lab.experiment preflight --task train --config smoke_fixture/relation.json --data smoke_fixture/data.json --states smoke_fixture/states.json --references smoke_fixture/references.json
```

输出目录必须不存在。正式配置从 configs/experiment_template.json 填写，null 是有意保留的未确认预算，不能直接启动。旧 configs/proposal.json 与 configs/smoke_*.json 保留供数据契约测试；旧 schema 1 不是训练配置。

## 数据与参考

数据 JSON 为实例列表：id, split, family_id, distribution, coordinates, demands, capacity, distance。depot 为索引 0，客户 1..N，距离固定 euclidean_float64，不自动取整或变更分布。训练/验证/测试共同登记在一个实例目录中；训练仅消费 train。

状态 JSON 为 [{instance_id, instance_hash, routes}, ...]；routes 是不含 depot 的客户路线。每实例至少一个当前解，同实例可有多个状态，各组共享相同状态文件。坐标、需求、容量、距离的 hash 对齐实例；同 family 变体不得跨划分。

参考 JSON 为 {train_instance_id: [reference, ...]}。每个 reference 包含 instance_id, instance_hash, status, source, solver, budget, seed, routes, cost；示例可用 make-smoke 查看。solver={name, version, binary_sha256}；budget={kind, limit}。成本需按本系统距离重算，不能把舍入打印成本冒充精确成本；外部导入的 provenance 是声明，当前不认证外部求解过程。

保留旧导入入口：

```bash
python -B -m relation_lab import-references --config CONFIG.json --data DATA.json --references REFERENCES.json --run-dir NEW_IMPORT_DIR
```

此命令接受带 schema=1 和 reference 的契约配置，只建立标签与身份产物。训练直接读取原始参考并重新校验，不信任未经绑定的标签文件。

## 当前边界

本地 stdlib 合成验证和修复吞吐已记录；神经测试需要现有 PyTorch，未运行项列在验证记录。纯 Python、串行实例和 CPU 采样优先可审计，不能预先声称 CVRP100 速度足够。需要服务器实测完整在线吞吐后再决定正式预算。

尚未执行真实参考求解、正式训练、测试集搜索或 NDS/HGS 性能比较。内置参考启发式不保证高质量；正式研究应导入或单独验证满足质量要求的参考集。实现一个可运行方法不构成关系监督有效的证据。
