# 本地验证记录

## 后续服务器反馈与修正

用户提供的服务器日志：Python 3.11.16、PyTorch 2.5.1+cu121、CUDA 12.1、Tesla V100-SXM2-16GB，CPU/CUDA 基础前向与反向通过。首次完整验证为 25 通过、1 失败、1 错误、0 跳过，不能视为全链路通过。

全链路测试用 MagicMock 替换 sys.stdout，首个错误为 encode() 的 encoding 参数收到 MagicMock；随后 Adam 初始化遇到 sympy.printing.str 缺失。已移除神经及系统测试的 stdout mock，保留真实输出流；第二个错误可能是首次依赖导入失败后的进程内残留，须新进程重跑确认，不能据此直接升级/修复共享环境。

修正只涉及 tests/test_neural.py 与 tests/test_system.py，不改变训练算法或依赖声明。本地仍无 PyTorch，服务器修正后的全链路结果待补充。PROJECT_ROOT 应显式设为日志确认的 /data/kcx/projects/select_04_NDS_HGS。

日期：2026-10-18。Windows、Python 3.11.1。未安装新依赖，未编译，未执行真实参考生成、研究训练、测试集搜索、服务器任务或 Git 提交。

## 已运行

标准库测试覆盖：参考合法性与重算成本；无向邻接/反转/排列不变性；固定置换及 RNG 隔离；实例 hash/split/family；明确 mask、不同 k 的删除修复可行性；容量和平局；相同删除集合但不同插入顺序的不同结果；LOO 数值；搜索严格改善；迟到候选不计收益；独立教师生成器在小合成数据上的确切预算和可重复性；只读 preflight；拒绝参考进入评价参数；重复目录拒绝；正式启动 gate；实例配对统计；源码 AST 解析。

最终运行 27 项测试：22 项通过，5 项因无 PyTorch 明确跳过，0 失败、0 错误。神经测试包括优化器恢复逐张量比对、辅助损失不随 rollout 数放大、shuffle 不改变策略随机流。原始结果见 `smoke_runs/verification_final/verification.json`。

`python -B scripts/verify.py --require-neural` 已实测因缺少 PyTorch 返回非零，明确不能通过服务器完整验收。缺依赖不是缺本地 NDS 的阻塞项：所有网络/训练/部署代码已经独立实现，但运行证据尚待服务器补齐。

## 尚未运行的神经验证

- 实际策略输出的 depot/重复动作 mask、精确删除数、顺序条件依赖。
- 辅助梯度到共享编码器、对称 logits、输入白名单拒绝参考字段。
- λ=0 两种组配置的更新等价、三组初始化/外生采样一致。
- 辅助损失对 rollout 数归一、shuffle 只改监督。
- 导出移除辅助头后策略输出一致。
- 保存/恢复模型、优化器、数据进度和随机状态后与未中断运行一致。
- 无参考源文件时部署、验证集评价、冻结和测试集评价全链路。

对应 `tests/test_neural.py`。服务器先用现有 PyTorch 运行严格验证，全部通过后才讨论正式预算；不能把语法解析或跳过当作这些项目通过。

## 吞吐实测的范围

`smoke_runs/repair_benchmark/benchmark.json`：N=8、删除 3 个客户、200 次纯 Python 合成修复，包含合法性检查和距离矩阵建立，共约 0.01616 秒，约 12,377 次修复/秒。这是一次极短微基准，噪声较大，不是稳定容量估计。

未测神经推理/训练吞吐、CVRP100 吞吐或 Linux/GPU 速度。不得外推为完整搜索性能。训练运行会记录 update_seconds/rollouts_per_second，评价会记录单步时间及搜索 proposals_per_second/超时量；需服务器实际运行后报告。

## 历史产物

原 `synthetic_smoke_run/` 保留，是前一数据契约阶段的产物，其 code hash 对应旧源码，不能作为本次新模块验证。新增 `test_artifacts/` 和 `smoke_runs/` 均只含明确合成数据、测试日志/manifest。未修改旧工作区。
