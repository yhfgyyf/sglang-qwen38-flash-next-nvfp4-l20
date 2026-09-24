# 双 L20 运行 Qwen3.8 Flash Next NVFP4

本仓库以 [SGLang v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)（上游提交 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`）为基础，合入了当前运行环境中实际使用的 3 处 Python 源码补丁。当前服务在 2 张 NVIDIA L20 上运行，模型目录为 `/home/dgd/damoxing/qwen3.8next`，对外模型名为 `Qwen3.8-flash`。该模型的 `config.json` 声明了 ModelOpt `NVFP4` 量化和 `Qwen4ExpForConditionalGeneration` 架构。

这里保存的是服务代码和说明，不包含模型权重、DSH 会话、日志或访问凭据。运行时仍需安装与本仓库对应的 SGLang 依赖及 CUDA 扩展，并提供模型目录。若只执行 `pip install sglang==0.5.20`，安装到的是未包含以下 3 处补丁的官方发行包。

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

## DSH Responses 空白正文：启动配置修复

这是**启动配置**调整，不属于上面 3 处源码改动。模型的 `chat_template.jinja` 在 `preserve_thinking` 未指定时，默认把历史 assistant reasoning 再次渲染进后续请求。DSH 长会话重放中，这会明显增大输入并与正文空白现象相关：此前默认设置的完整案例 6 次有 4 次空白；为请求设置 `preserve_thinking=false` 后，8 次均有正文。因此服务统一使用 `--default-chat-template-kwargs '{"preserve_thinking":false}'`，让模板保留最近查询的推理流程，但不反复携带较早轮次的 reasoning。空白现象与这一模板行为之间的逐 token 因果链尚未单独证明；对照测试支持采用该配置。

重启后的直接流式 Responses 测试重放了 DSH 历史和 27 个工具定义，额外加入背景文本，连续做了 3 组“推理 → 工具调用 → 工具结果 → 正文”：每次请求的输入量为 52,447–53,170 token；6/6 次有 reasoning，3/3 次有结构化 `bash` 工具调用，3/3 次工具结果后的正文非空并正确使用了模拟结果 `56`。其中 2 次正文在正确答案后又提及旧会话主题，说明内容相关性仍有改进空间。此测试直接调用 SGLang 接口，工具结果由脚本模拟，没有执行模型提出的命令。

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
  --disable-custom-all-reduce \
  --default-chat-template-kwargs '{"preserve_thinking":false}'
```
