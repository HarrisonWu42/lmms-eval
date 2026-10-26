# AudioMarathon：全样本 accuracy 评估

此目录采用自定义的全样本、按样本数加权口径，不再使用官方的 F1 主分数或十任务等权平均。配置版本为 `2.0`。

九个分类子集（含 HAD、VESUS、VoxCeleb 年龄和性别）统一计算：

```text
accuracy_i = 正确回答数 / 该子集全部评估题数
```

无法解析的回答和空回答均计错，不从分母删除。每个子集输出：

- `audiomarathon_acc`：accuracy 比例值。
- `audiomarathon_score`：`100 × audiomarathon_acc`，供现有结果汇总读取。

不再输出 `audiomarathon_f1`。

LibriSpeech-long 是转写任务，继续使用 word accuracy：每条音频先计算 WER，再对全部音频取平均；空预测计为 WER=100%，不会被排除。其 `audiomarathon_acc = 1 − 平均 WER / 100`，`word_accuracy` 和 `audiomarathon_score` 均为百分数。它不是整句完全匹配率，也不是选择题正确率；沿用 `100−WER` 定义，不截断负分，插入错误使 WER 超过 100% 时 word accuracy 可以为负。

整体 `audiomarathon2` 行按实际评估记录数加权：

```text
overall_accuracy = Σ(N_i × accuracy_i) / ΣN_i
overall_score = 100 × overall_accuracy
```

`N_i` 包括无效回答，不是有效预测数。整体结果分别保存为 `audiomarathon_acc` 和 `audiomarathon_score`。完整 manifest 的总题数为 5,774；只评部分样本时使用实际评估数量。由于 ASR 项采用 word accuracy，十任务总值是包含转写指标的加权综合 accuracy，不是所有题目的二元答对比例，不能与旧版官方平均 F1 直接比较。

TAU 默认读取 `manifests/tau_corrected.jsonl`，VESUS 保留修复后的官方答案解析规则。此次计分修改不会覆盖历史 results，也不会替代 TAU 的 77 条选项变更题所需的重新推理。

## 独立启动

此包的任务名为 `audiomarathon2`，十个子任务使用 `audiomarathon2_` 前缀。这样通过 `include_path` 加载时不会与仓库内旧的 `audiomarathon` 任务重名；旧任务不受影响。

在已安装项目依赖的 CUDA 环境中，从 `code/MLLM-Comp-Platform` 根目录执行：

```bash
MODEL_PATH="Qwen/Qwen2.5-Omni-7B" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_TAG="7b-audiomarathon2-$(date +%Y%m%d-%H%M%S)" \
bash scripts/eval/lmms_eval/lmu/start_eval_aud_audiomarathon2.sh
```

默认按工作区布局查找项目根目录向上两级的 `audiomarathon2/` 和 `datasets/AudioMarathon/`。若服务器布局不同，显式设置：

```bash
export AUDIOMARATHON_TASK_DIR=/absolute/path/to/audiomarathon2
export AUDIOMARATHON_ROOT=/absolute/path/to/AudioMarathon
```

完整 SFT 权重使用 `CHECKPOINT_PATH=/absolute/path/to/checkpoint-N` 代替 `MODEL_PATH`。LoRA 使用 `ADAPTER_PATH=/absolute/path/to/adapter`，不要同时设置模型路径；正式运行会用 `swift export --merge_lora true` 导出权重，已有合并权重可通过 `MERGED_MODEL_PATH` 指定。

脚本默认使用四张卡，进程数随 `CUDA_VISIBLE_DEVICES` 的数量调整，也可设置 `EVAL_NUM_PROCESSES`。追加 `--plan_only` 只检查启动计划，不加载模型、不执行 LoRA 合并；追加 `--sample_limit 2` 可先对每个子集试跑两题。默认端口为 `29638`，可用 `AUDIOMARATHON2_MAIN_PROCESS_PORT` 修改。

输出默认在 `runs/lmu/eval/$RUN_TAG/audio_eval/audiomarathon2/`，可用 `OUTPUT_DIR` 覆盖。另支持 `PYTHON_BIN`、`SWIFT_BIN` 和 `DATASET_ROOT`（其下应有 `AudioMarathon/`）。运行会读取全部任务的新计分规则，不会把旧日志直接冒充 TAU 修正版的推理结果。
