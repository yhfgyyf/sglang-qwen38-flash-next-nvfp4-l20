# 双 L20 运行 Qwen3.8 Flash Next NVFP4

本仓库以 [SGLang v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)（上游提交 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`）为基础，合入了 4 处 Python 源码补丁。当前服务在 2 张 NVIDIA L20 上运行，模型目录为 `/home/dgd/damoxing/qwen3.8next`，对外模型名为 `Qwen3.8-flash`。该模型的 `config.json` 声明了 ModelOpt `NVFP4` 量化和 `Qwen4ExpForConditionalGeneration` 架构。

这里保存的是服务代码和说明，不包含模型权重、DSH 会话、日志或访问凭据。运行时仍需安装与本仓库对应的 SGLang 依赖及 CUDA 扩展，并提供模型目录。若只执行 `pip install sglang==0.5.20`，安装到的是未包含以下 4 处补丁的官方发行包。

## 源码修复

### 1. L20 上 Qwen Sparse Attention 的 varlen 后备内核选择

- **现象：**L20 属于 CUDA SM89。Qwen Sparse Attention（QSA）在目标验证阶段进行 CUDA Graph 捕获时，通用的 FA4 CuTe `flash_attn_varlen_func` 后备路径无法编译该模型使用的 packed decode 形状，导致服务启动失败。
- **原因：**原 `_resolve_flash_attn_varlen_func()` 只对 SM121 选择专用实现；其他设备先尝试 `flash_attn`，再回退到 FA4 CuTe，没有为 SM89 的该形状选用 SGLang 已有的兼容内核。
- **修复：**在 [qwen_sparse_attn_backend.py](python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py) 中增加 SM89 分支，返回 `sgl_kernel.flash_attn.flash_attn_varlen_func`。其他架构的选择路径不变。

### 2. NVFP4 MoE 在 Marlin 路径上分配了不用的 swizzled scale

- **现象：**双 L20 加载 NVFP4 MoE 时，GPU 内存占用和碎片化增加，模型加载余量被无用的 scale 张量消耗。
- **原因：**SM89 自动选择 Marlin 后备路径，但 `create_weights()` 仍给每个 MoE 层的 `w13`、`w2` scale 做 `swizzle_blockscale()` 并分配新参数。随后 Marlin 的权重处理直接返回，不会读取这些 swizzled 参数。
- **修复：**在 [modelopt_quant.py](python/sglang/srt/layers/quantization/modelopt_quant.py) 中保存 `use_marlin_fallback`，对该路径不创建 `w13_blockscale_swizzled` 和 `w2_blockscale_swizzled`；需要它们的其他后端仍照常创建。

### 3. Marlin FP4 重打包时替换 Parameter 对象

- **现象：**Marlin 权重重打包会替换已注册的权重和 scale `Parameter` 对象；这会让持有旧对象引用的权重更新或图复用流程面临引用失效、读到旧数据的风险。
- **原因：**原实现对 `w13_weight`、`w2_weight` 及 4 个 scale 字段直接赋值新的 `torch.nn.Parameter`。
- **修复：**在 [marlin_utils_fp4.py](python/sglang/srt/layers/quantization/marlin_utils_fp4.py) 中统一调用已有的 `copy_or_rebind_param()`。它在可行时更新原参数的数据，保持模块参数对象的身份稳定。

### 4. Responses 历史回合被 `phase` 拆成多个 assistant 消息

- **现象：**重放 DSH 的约 5 万 token 会话时，开启 `preserve_thinking` 的 Responses 请求会间歇返回只有换行的最终正文。修复前，原始 182 个 Responses 输入项被转换成 174 条内部 Chat 消息；同一 assistant 回合的推理、正文、工具调用被拆开。
- **原因：**`_merge_consecutive_assistant_messages()` 原先只合并 `phase` 相同的相邻 assistant 消息。Responses 的 reasoning 与 function_call 项没有 `phase`，而正文项标有 `commentary` 或 `final_answer`，导致同一回合无法合并，聊天模板读到多个连续的 assistant 消息。
- **修复：**在 [serving_responses.py](python/sglang/srt/entrypoints/openai/serving_responses.py) 中允许没有 `phase` 的片段与相邻 assistant 消息合并，并继承正文的明确阶段；两个明确而不同的阶段仍保持分开。相同 DSH 历史现转换为 98 条内部 Chat 消息。对应的 [回归测试](test/registered/unit/entrypoints/openai/test_serving_responses.py) 同时覆盖推理、工具调用、工具结果与不同阶段边界。

## DSH Responses 空白正文：历史启动配置与本次源码修复

此前曾用启动参数 `--default-chat-template-kwargs '{"preserve_thinking":false}'` 避免反复携带历史 reasoning：在当时的完整 DSH 案例中，保留历史 thinking 的 6 次有 4 次空白，关闭后 8 次均有正文。这是绕开症状的配置调整。进一步对照发现，Responses 转换器按 `phase` 拆开同一 assistant 回合才是更具体的可修复问题：修复前以原始 DSH 输入、开启历史 thinking 重放 5 次，3 次在产生推理后以只有 `\n\n` 的正文结束。

源码修复后的测试直接调用当前 SGLang 服务，重放同一 DSH 会话的 182 个 Responses 输入项和 27 个工具定义：

- **原始最终回答步骤，显式开启历史 thinking：**输入 49,961 token，10/10 次都有推理；9 次直接给出非空最终正文，1 次继续调用工具；空白最终正文 0/10。
- **长度对照：**保留原始 `phase`，仅给最早的用户消息补充中性背景，使输入达到 50,640 token；5/5 次有推理和非空最终正文。
- **当前启动命令的默认设置：**不显式传 `preserve_thinking`，原始最终回答步骤输入 49,961 token；3/3 次有推理和非空最终正文。
- **同一会话中的工具调用步骤：**从触发工具调用的 DSH 历史位置重放，输入 38,991 token；3/3 次有推理和结构化 `bash` 工具调用。测试没有执行模型给出的命令。
- **代码回归：**`test_serving_responses.py` 的 66 个测试全部通过，其中包括新增的跨无 `phase` 片段合并测试和原有的明确阶段边界测试。

以上是重复采样结果，不能保证未来任何输入都不会出现空白正文；它表明原始故障案例在本次修复后没有复现。当前服务没有设置 `preserve_thinking=false` 的默认覆盖参数。

## 当前服务启动命令

以下按当前运行进程的参数记录。`--model-path` 是本机路径，迁移到另一台机器时需要替换；`--host 0.0.0.0` 会监听所有网络接口。

```bash
/home/dgd/anaconda3/envs/sglang/bin/sglang serve \
  --model-path /home/dgd/damoxing/qwen3.8next \
  --mem-fraction-static 0.96 \
  --chunked-prefill-size 8192 \
  --tp 2 \
  --speculative-algorithm NEXTN \
  --served-model-name Qwen3.8-flash \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --host 0.0.0.0 \
  --port 10002 \
  --ple-offload-embedding \
  --disable-custom-all-reduce
```
