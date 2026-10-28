# 自训练 teacher → reflow：可选后续实验

本模块建立可执行的配对数据生成与 student 训练流程。**尚未在本项目的大模型上运行 reflow，也没有证据表明本项目的 1–4 步结果已具备可用质量，或在同等质量下更快。** 它不会自动启动、不会接管当前训练，也不会修改 `model.py`、`train.py`、`sample.py` 或本地服务。

## 方法依据与本项目的边界

Rectified Flow 原论文使用直线路径上的速度回归；reflow 再用前一轮 ODE 的起点与终点配对训练下一轮。作者的官方代码也先保存 `z0/z1`，再选择均匀时间训练 reflow。参见 [原论文](https://arxiv.org/abs/2209.03003)、[作者的方法说明](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)、[官方配对生成实现](https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/run_lib_reflow.py) 与 [官方损失实现](https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/losses.py)。这些来源说明算法，不验证本项目的训练效果。

本实现采用当前项目的时间方向：

```text
x0 ~ N(0, I)
x1 = teacher 的多步 Euler ODE(x0, prompt), t: 0 → 1
xt = (1-t) * x0 + t * x1,  t ~ Uniform[0,1)
loss = mean((student(xt, t, prompt) - (x1-x0))²)
```

训练必须使用保存的同一个 `x0`。重新抽一个独立噪声会破坏 teacher 建立的配对。本实现不在配对终点做显示用的 `[-1,1]` 裁剪；`x0/x1` 以 float32 保存，不转成 PNG。这里使用有限步 Euler 近似，teacher 的数值误差也可能进入 student。

默认从本项目**自己从随机权重训练的 teacher** 拷贝初始化 student，以保留已有能力。`--student-init random` 则重新随机初始化 student。前一种做法不是使用外部预训练模型，也不能称为“student 本次全新随机初始化”。检查点和模型卡分别记录两者，以及所有权重的本地随机来源链。

没有加入外部预训练文本编码器、VAE、感知损失或图片回退。弱 teacher 可能把噪声、空白和错误布局教给 student。reflow 不是补足基础训练或文字学习的保证。

## 输入检查与互斥

teacher 必须是项目格式的 `inference.pt`，且满足：

- `checkpoint_kind == "inference-ema"`，已经完成至少 1 个优化步骤。
- `provenance.random_initialization is True`。
- `provenance.pretrained_weights is False`。

这是格式与来源声明检查，不是对权重来源的加密认证。只加载自己生成并信任的本地检查点；PyTorch 恢复文件不是接收任意第三方文件的安全格式。

`pairs/train` 会用 `model_access` 同时锁住当前项目的 `neural/checkpoints`、teacher 所在目录，以及自己的数据/输出目录。即使 teacher 在其他位置，也保留 base 目录作为 GPU 使用互斥入口。若基础训练、采样或评估正在使用模型，命令会立即失败，不排队争抢 GPU。只有很小的 CPU smoke 使用它自己创建的临时锁目录。

输出路径不得与 base、teacher、配对数据路径相同、互为父子目录。默认位置为：

```text
neural/reflow-data/          # teacher 噪声-终点对、caption/token、manifest、status
neural/reflow-checkpoints/   # student latest.pt、inference.pt、EMA、模型卡、日志
```

不要把 `--checkpoints` 指向 base。当前工作台仍读取原来的 base 检查点；本模块不会自动切换模型。

## 后续实验命令

以下命令**仅供之后明确安排实验时使用**，不是已完成的 GPU 训练记录。在项目根目录执行。先正常暂停基础训练并确认它释放模型锁，再保存一个固定 teacher 快照；不要在基础训练仍写文件时复制检查点。快照的内容 SHA256 会写入数据 manifest，student 训练时必须匹配。若 base 后来继续训练，应继续使用这个固定快照，而不是变化中的 `inference.pt`。

准备一个 UTF-8 `prompts.txt`，每行一条 caption；也支持 JSONL，每行含 `prompt` 字段。每条 caption 必须适配 teacher tokenizer 的字节预算；超长会报错，不会静默截断。配对图像分辨率来自 teacher 的实际 `native_size`，不通过放大增加细节。

```powershell
# 仅在基础训练已经正常暂停后操作；目标快照目录应独立保存。
# 此工具不自动复制或暂停正在训练的 base。

.\.venv\Scripts\python.exe -m neural.reflow pairs `
  --teacher .\neural\frozen-teacher\inference.pt `
  --prompts .\neural\reflow-prompts.txt `
  --output .\neural\reflow-data `
  --count 1024 --teacher-steps 64 --batch-size 2 --shard-size 32

.\.venv\Scripts\python.exe -m neural.reflow train `
  --pairs .\neural\reflow-data `
  --checkpoints .\neural\reflow-checkpoints `
  --student-init teacher --steps 5000 --batch-size 2 --accum 4
```

64 teacher 步、1024 对与 5000 student 步只是可修改的实验起点，不是已验证的配方。默认 GPU BF16；确需 FP32 可加 `--fp32`。生产 CPU 运行必须显式 `--device cpu --allow-cpu`，通常仅适合很小的诊断。

每个 256×256 样本的两个 float32 像素张量约占 512 KiB，1024 对仅像素数据约 512 MiB，另有 tokens、caption 与元数据。`x0` 使用按样本序号派生的独立 CPU Torch Generator，因此保存种子与实际噪声均可检查。批大小、分片大小、teacher 步数、来源 hash、prompt 内容、设备类型与精度也固定在生成合同中。

注意：现有 `neural.sample` 在 GPU 上使用 GPU Generator，相同整数 seed 不意味着它复现 CPU Generator 保存的那一张 `x0`。若核对某个配对终点，应读取保存的 `x0` 调用本模块的 `teacher_endpoint`。独立比较 teacher/student 时，可以让二者均使用同一个现有采样器、相同设备与 seed。

## 暂停与恢复

向相应输出目录写入 `STOP`，或发送 Ctrl+C。配对生成在当前 batch 后提交完整的已生成部分；训练在当前优化步骤后保存。工具不擅自删除 `STOP`；恢复前由操作者移除它。

```powershell
# 暂停 student；只影响 reflow 输出目录。
New-Item -ItemType File .\neural\reflow-checkpoints\STOP

# 移除 STOP 后，用原来的设置恢复；可增加总目标步数。
.\.venv\Scripts\python.exe -m neural.reflow train `
  --pairs .\neural\reflow-data `
  --checkpoints .\neural\reflow-checkpoints `
  --student-init teacher --steps 10000 --batch-size 2 --accum 4 --resume
```

配对生成恢复是在相同 `pairs` 命令上增加 `--resume`。对数、种子、batch/shard、teacher 和 prompt 内容不能改变。即使暂停发生在第一对提交前也能恢复。

student `latest.pt` 保留模型、EMA、AdamW、Python/NumPy/Torch/CUDA RNG，以及配对抽样器 RNG。恢复时核对配置、数据 manifest、teacher hash、精度和设备类型，允许增加总训练步数。学习率仅由固定 warmup 和已完成步数决定，不随总目标变化。EMA 从当前 student 初始化，使用固定 `--ema-decay`，不会在恢复时重新初始化。

数据按原子 manifest 提交，文件带 SHA256。student 文件复用基础训练的原子写入工具，但使用自己的保存函数，模型卡的 `training_data` 和 provenance 的 `dataset` 指向实际 **reflow 配对目录**，不会误写原始合成数据路径。

前向/反向异常会回滚未完成步骤的抽样/时间 RNG 与计数后尝试保存。若 Adam 或 EMA 更新过程出错，无法证明状态完整，工具保留此前的原子检查点而不保存半更新模型。训练日志可能包含恢复后重复的同一步记录，分析时按检查点和 step 处理。GPU 数值运算不承诺逐位复现。

`status.json` 提供真实 state、阶段、步数、损失与时间；`training.jsonl` 记录 student 更新。它们不表示图像质量已验证。进程被强制结束时，恢复会回到最后一次完整提交，可能重做未提交工作。

## 之后怎样评估 1–4 步

可使用现有采样入口读取独立的 student 推理检查点，例如：

```powershell
.\.venv\Scripts\python.exe -m neural.sample "黑白聊天窗口，输入框和发送按钮" `
  --checkpoint .\neural\reflow-checkpoints\inference.pt `
  --steps 4 --seed 123 --output .\neural\reflow-evaluation\student-4.png
```

应固定未参与配对生成的 prompt/seed，对比 teacher 多步、student 1/2/4/16 步和相同条件下的延迟，检查内容遵循、线条、布局、可读文字、空白比例及不同 seed 的变化。先预热再测延迟，并记录设备、原生尺寸、检查点与样本数；不能用一张图或训练损失替代结论。本工具没有自动给出 OCR、FID 或“已达到四步质量”评分。

## 已执行的验证范围

只在 CPU 上用临时的 **252,513 参数、32×32 小网络** 测试，未读取生产大权重，未申请 GPU；所有临时数据位于指定 work 目录并在结束后清理：

```powershell
.\.venv\Scripts\python.exe -m neural.reflow smoke-test `
  --work ".\work" `
  --report ".\work\reflow-smoke-report.json"
```

该小测试中，固定配对 loss 在 14 次实际梯度更新内从约 `0.009222` 降至 `0.000291`；同时检查原始噪声保存、端点公式、分片/校验、配对恢复、STOP 暂停、完整与推理检查点加载、数据来源路径。还验证了 1/4 步采样能输出有限数值、全新随机 student 选项、零对暂停恢复、来源声明拒绝、共享锁冲突拒绝与输出路径保护。CPU 上连续训练与暂停恢复后的模型/EMA 逐位一致。**这些结果只验证实现路径，不验证生产模型的画质、提示词理解或少步速度收益。**
