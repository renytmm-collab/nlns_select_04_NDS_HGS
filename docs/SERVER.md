# Linux 交付与运行

## 环境与路径

只需标准库＋PyTorch 2.6 或更新的 2.x。建议 Python 3.10+，CPU smoke 足够；CUDA 可选，首次运行不会触发编译。requirements.txt 只声明依赖，不由脚本安装。用户自行选择适合服务器 CUDA/驱动的 torch wheel，安装留待授权；不提供自动 pip/uv sync。Windows 本地路径不会写入配置。

用户先上传代码并设置真实绝对路径，本文不捏造路径：

```bash
: "${PROJECT_ROOT:?请先 export PROJECT_ROOT 为真实服务器绝对路径}"
case "$PROJECT_ROOT" in /*) ;; *) echo 'PROJECT_ROOT 必须是绝对路径'; exit 1;; esac
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
# 仅 CUDA 模式需要在进程启动前设置：
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## 完整合成 smoke

```bash
python -B scripts/verify.py --require-neural
python -B -m relation_lab.experiment make-smoke --run-dir smoke_fixture
python -B -m relation_lab.experiment preflight --task train \
  --config smoke_fixture/relation.json --data smoke_fixture/data.json \
  --states smoke_fixture/states.json --references smoke_fixture/references.json
python -B -m relation_lab.experiment train --synthetic \
  --config smoke_fixture/relation.json --data smoke_fixture/data.json \
  --states smoke_fixture/states.json --references smoke_fixture/references.json \
  --run-dir smoke_relation
```

所有目录需为新名字；重跑显式选择新目录，不删除旧结果后静默重试。verify 的全链路测试只用 N=8、2 更新的合成数据，产物在 test_artifacts/。--require-neural 缺少 PyTorch 或发生 skip 时失败，防止把跳过误认作通过。

另外两组用 baseline.json / shuffled_relation.json。baseline 不传 --references；各组共享数据、状态、初始化 seed、k、K 和预算。多 seed 需要显式创建预先登记的配置列表，不能运行后挑一个 seed。

## 正式配置与 preflight

从 configs/experiment_template.json 复制配置，明确 distribution、步数、单步重复数、搜索 seconds/iterations、M/参考预算/solver 身份。默认 CVRP100、k=10、width=64、K=8 仅为建议，未证明合适。将 budgets_confirmed 改 true 表示已核算预算；三组除 group/λ 外保持相同，seed 成对。

数据、参考、状态分别用 DATA、REFERENCES、STATES 参数指定，CONFIG 指向所选组；路径可相对 PROJECT_ROOT，不在模型输入使用路径。DATA 包含预先划分的全部实例，用于身份/泄漏检查。STATES 必须由已批准的同一初始解流程预先形成，禁止从训练参考偷带信息。

```bash
python -B -m relation_lab.experiment preflight --task train \
  --config "$CONFIG" --data "$DATA" --states "$STATES" --references "$REFERENCES"
```

baseline 或 λ=0 必须去掉 --references。preflight 不生成数据、不跑求解/推理；torch_available 只检查模块可发现性，不能替代 import/设备 smoke。

## 参考集

优先导入经过质量核验的外部启发式解，按 README 契约重算成本并填来源、版本、二进制 hash、seed 和预算。当前无外部 HGS 适配器，不调用未知二进制；系统运行不需要 HGS。

可选内置生成器为“多起点随机顺序 cheapest insertion＋有限步路线内 2-opt”。每条参考独立 seed；limit 是确切起点数，另填 max_2opt_moves_per_start。它不是 HGS，解质量不预设。只对 train 生成、独立于学生 seed，记录实际耗时/起点数；重复按配置处理，不补跑。

先读取当前生成器身份，手动填入 reference.solver，再执行：

```bash
python -B -m relation_lab.experiment solver-identity
python -B -m relation_lab.experiment preflight --task generate-references \
  --config "$CONFIG" --data "$DATA"
# 下条只由用户在批准预算后执行：
python -B -m relation_lab.experiment generate-references --confirm-real-run \
  --config "$CONFIG" --data "$DATA" --run-dir "$REFERENCE_RUN"
```

binary_sha256 对此纯 Python 求解器是 references.py 和 vrp.py 内容 hash 的组合，不声称存在二进制；外部求解器应使用真实二进制 hash。references.json 可供同一参考配置的训练组使用。参考生成成本单独报告。

## 前台、nohup 与独立状态

```bash
python -B -m relation_lab.experiment train --confirm-real-run \
  --config "$CONFIG" --data "$DATA" --states "$STATES" \
  --references "$REFERENCES" --run-dir "$RUN_DIR"
```

后台启动（关系组；RUN_DIR 父目录必须存在，目标目录不能存在）：

```bash
cd "$PROJECT_ROOT"
LOG="${RUN_DIR}.log"
( set -o noclobber
  nohup python -B -u -m relation_lab.experiment train --confirm-real-run \
    --config "$CONFIG" --data "$DATA" --states "$STATES" \
    --references "$REFERENCES" --run-dir "$RUN_DIR" > "$LOG" 2>&1 < /dev/null &
  echo "PID=$! LOG=$LOG"
)
```

baseline 的训练命令均去掉 --references；日志重定向也拒绝覆盖。单独查看：

```bash
tail -n 50 "${RUN_DIR}.log"
python -B -c 'import json,sys; m=json.load(open(sys.argv[1])); print({k:m.get(k) for k in ("status","attempted_step","committed_step","latest_checkpoint","error")})' "$RUN_DIR/manifest.json"
tail -n 5 "$RUN_DIR/metrics.jsonl"
```

指标有 update_seconds、rollouts_per_second；manifest session_seconds 含逐步落盘开销。训练、纯修复和完整在线搜索吞吐分别报告。

## 恢复与导出

--stop-after 可在已提交步骤主动停止，不修改总 steps。恢复写入新目录，原 run 不能有额外未提交尝试，不能改预算或身份：

```bash
python -B -m relation_lab.experiment resume --confirm-real-run \
  --config "$CONFIG" --data "$DATA" --states "$STATES" --references "$REFERENCES" \
  --checkpoint "$CHECKPOINT" --run-dir "$RESUMED_RUN"
python -B -m relation_lab.experiment export --checkpoint "$FINAL_CHECKPOINT" --run-dir "$EXPORT_RUN"
```

baseline 恢复仍不传参考。导出是文件转换，不启动研究计算。policy.pt 不含关系头；模型文件通过 weights_only=True 加载。

## 验证、冻结、测试

全部评价不传参考、不打开参考文件。验证集以实际求解表现选配置，checkpoint 固定最后一步：

```bash
python -B -m relation_lab.experiment evaluate --confirm-real-run \
  --config "$CONFIG" --data "$DATA" --states "$STATES" --policy "$POLICY" \
  --split validation --run-dir "$VALIDATION_RUN"
python -B -m relation_lab.experiment freeze \
  --config "$CONFIG" --data "$DATA" --states "$STATES" --policy "$POLICY" \
  --validation-results "$VALIDATION_RUN/results.json" --run-dir "$FREEZE_RUN"
python -B -m relation_lab.experiment evaluate --confirm-real-run \
  --config "$CONFIG" --data "$DATA" --states "$STATES" --policy "$POLICY" \
  --split test --freeze "$FREEZE_RUN/freeze.json" --run-dir "$TEST_RUN"
```

一次 evaluate 包含单步、固定迭代和等时间三类。总预算=该 split 状态数×eval seeds×（repeats 次单步修复＋iterations 次搜索＋seconds 时间搜索），另有不可取消单步的超时与文件 I/O。所有组/seed 的配置和选择理由一起冻结后才开始测试。

无参考部署可单独用 deploy，参数与 evaluate 相同但不传 split/freeze，仅用于部署输入验证，不能代替冻结测试。可用同一约定的新 DATA/STATES 部署。训练集关系诊断用 relation-diagnostics，传配置/数据/状态/参考/最终 checkpoint、新 run-dir 和启动开关；不参与测试选择。

多 seed 配对分析，两侧每文件对应一个匹配训练 seed：

```bash
python -B -m relation_lab.experiment compare \
  --left relation_seed1/results.json relation_seed2/results.json \
  --right baseline_seed1/results.json baseline_seed2/results.json \
  --run-dir comparison_relation_baseline
```

同样比较 relation 与 shuffle。正值表示 left 成本更低，以实例为统计单位。小合成数据 CI 没有研究解释。
