# K-Buffer测试工具

这个测试工具用于验证K-Buffer渲染核心（renderForward和renderBackward kernel）修改的正确性，基于真实的TreeHill数据集进行端到端测试。支持前向渲染和反向传播两种测试模式。

## 快速开始

### 1. 训练TreeHill模型

首先需要训练一个TreeHill数据集的模型：

```bash
# 训练TreeHill模型，使用8x降采样图片
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/mipnerf360/treehill \
    out_dir=runs \
    experiment_name=treehill_3dgut_15k_8x \
    n_iterations=15000 \
    dataset.downsample_factor=8

# 训练TreeHill模型
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/mipnerf360/treehill \
    out_dir=runs \
    experiment_name=treehill_3dgut_15k \
    n_iterations=15000 \
```

### 2. 捕获基准数据

#### 前向渲染测试（renderForward kernel）

在修改renderForward代码之前，先捕获原始版本的输出作为基准：

```bash
cd threedgut_tracer/tests

# 使用原始分辨率训出来的模型推理原始分辨率
python test_kbuffer_modification.py --mode capture \
    --checkpoint ../../runs/treehill_3dgut_15k/treehill-0809_183156/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_15k.pt

# 使用原始分辨率训出来的模型推理原始8x降采样分辨率
python test_kbuffer_modification.py --mode capture \
    --checkpoint ../../runs/treehill_3dgut_15k/treehill-0809_183156/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_15k_8x.pt


# 新的baseline，8x降采样图片训出来的checkpoint
python test_kbuffer_modification.py --mode capture \
    --checkpoint ../../runs/treehill_3dgut_15k_8x/treehill-0809_183156/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_forward.pt
```

#### 反向传播测试（renderBackward kernel）

如果要测试renderBackward kernel，需要捕获反向传播基准：

```bash
python test_kbuffer_modification.py --mode capture-backward \
    --checkpoint ../../runs/treehill_3dgut_onestep/treehill-0809_183156/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_backward.pt
```

### 3. 修改K-Buffer代码

根据测试目标修改相应的文件：

- **renderForward kernel**: 在 `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh` 等文件中修改前向渲染逻辑
- **renderBackward kernel**: 修改反向传播相关的CUDA kernel代码

### 4. 重新编译

```bash
cd ../../threedgut_tracer
python setup_3dgut.py build_ext --inplace
```

### 5. 验证修改

#### 验证前向渲染修改

```bash
cd tests

python test_kbuffer_modification.py --mode verify \
    --baseline baselines/treehill_forward.pt
```

#### 验证反向传播修改

```bash
python test_kbuffer_modification.py --mode verify-backward \
    --baseline baselines/treehill_backward.pt
```

## 命令行参数

### 前向渲染捕获模式 (capture)

- `--checkpoint, -c`: 训练好的TreeHill模型checkpoint路径 **(必需)**
- `--dataset, -d`: TreeHill数据集路径 (默认: `data/mipnerf360/treehill`)
- `--output, -o`: 基准文件输出路径 (默认: `baselines/treehill_onestep.pt`)
- `--view-id`: 测试视角ID (默认: 5)

### 前向渲染验证模式 (verify)

- `--baseline, -b`: 基准文件路径 (默认: `baselines/treehill_onestep.pt`)
- `--tolerance-rgb`: RGB差异容忍度 (默认: 1e-3)
- `--tolerance-depth`: 深度差异容忍度 (默认: 1e-3)
- `--show-gaussian-stats`: 显示每个像素处理的高斯点数量统计
- `--save-heatmap`: 保存hits_count热力图到指定路径

### 反向传播捕获模式 (capture-backward)

- `--checkpoint, -c`: 训练好的TreeHill模型checkpoint路径 **(必需)**
- `--dataset, -d`: TreeHill数据集路径 (默认: `data/mipnerf360/treehill`)
- `--output, -o`: 反向传播基准文件输出路径
- `--view-id`: 测试视角ID (默认: 5)

### 反向传播验证模式 (verify-backward)

- `--baseline, -b`: 反向传播基准文件路径
- `--tolerance-grad`: 梯度差异容忍度 (默认: 1e-5)

### 通用参数

- `--config`: 配置文件路径 (默认: `apps/colmap_3dgut.yaml`)
- `--downsample-factor`: 图像下采样因子 (1/2/4/8, 默认: 1)
- `--verbose, -v`: 详细日志输出

## 示例输出

### 前向渲染捕获基准成功

```
🎯 K-Buffer基准捕获模式
--------------------------------------------------
📂 Loading TreeHill dataset from data/mipnerf360/treehill
🏋️ Loading checkpoint from runs/treehill_3dgut/ckpt_last.pt
✅ Checkpoint loaded
🔄 Running inference and capturing trace inputs/outputs...

✅ Baseline saved: baselines/treehill_forward.pt
   📊 Resolution: (1267, 831)
   🔬 Gaussians: 42000
   💾 File size: 12.8 MB
```

### 反向传播捕获基准成功

```
🎯 K-Buffer反向传播基准捕获模式
--------------------------------------------------
📂 Loading TreeHill dataset from data/mipnerf360/treehill
🏋️ Loading checkpoint from runs/treehill_3dgut/ckpt_last.pt
✅ MixtureOfGaussians model initialized with gradients enabled
🔄 Running forward pass...
🔄 Running backward pass...
🔍 RGB MSE Loss: 0.234567
✅ Backward pass completed

✅ 反向传播基准捕获完成!
📂 保存位置: baselines/treehill_backward.pt
📊 场景信息: (1267, 831) 分辨率
📉 损失值: 0.234567
🎯 Gradients: 254321/254321 params have gradients
💾 File size: 24.3 MB
```

### 前向渲染验证成功

```
🔍 K-Buffer修改验证模式
--------------------------------------------------
📂 Loaded baseline: baselines/treehill_forward.pt
   🏷️  Version: original_trace_function
🔄 Running modified trace function...

🌳 ============================================================
🌳 K-Buffer修改验证报告
🌳 ============================================================
🏷️  基准版本: original_trace_function
📅 测试时间: 2025-01-08 15:30:45
🖥️  GPU: NVIDIA RTX 4090
📊 分辨率: (1267, 831)
------------------------------------------------------------
🎨 RGB输出       | ✅ 通过
   最大误差: 1.23e-06 (阈值: 1.00e-03)
   平均误差: 2.45e-08

📏 深度输出       | ✅ 通过
   最大误差: 5.67e-05 (阈值: 1.00e-03)
   平均误差: 8.90e-07

🎯 命中计数       | ✅ 通过
   最大误差: 0.00e+00 (阈值: 1.00e-06)
   平均误差: 0.00e+00

🏆 🎉 整体通过
🌳 ============================================================
```

### 反向传播验证成功

```
🔍 K-Buffer反向传播修改验证模式
--------------------------------------------------
📂 Loaded backward baseline: baselines/treehill_backward.pt
   🏷️  Version: original_backward_function
   📉 Expected loss: 0.234567
🔄 Running modified forward pass...
🔄 Running modified backward pass...
🔍 Actual loss: 0.234567
✅ Modified backward pass completed

🌳 ======================================================================
🌳 K-Buffer反向传播修改验证报告
🌳 ======================================================================
🏷️  基准版本: original_backward_function
📅 测试时间: 2025-01-08 15:30:45
🖥️  GPU: NVIDIA RTX 4090
📊 分辨率: (1267, 831)
----------------------------------------------------------------------
🎨 前向输出验证 (应与原版本完全一致):
   RGB输出:      ✅ 通过 (最大差异: 0.00e+00)
   深度输出:     ✅ 通过 (最大差异: 0.00e+00)
   命中计数:     ✅ 通过 (最大差异: 0.00e+00)

📉 损失值验证:   ✅ 通过
   期望损失: 0.23456789
   实际损失: 0.23456789
   损失差异: 0.00e+00

🎯 梯度验证:     ✅ 通过
   最大梯度差异: 3.45e-06 (阈值: 1.00e-05)
   平均梯度差异: 5.67e-08

📊 各参数梯度对比:
   positions            ✅ (最大差异: 3.45e-06, 期望范数: 2.34e+02, 实际范数: 2.34e+02)
   rotation             ✅ (最大差异: 1.23e-06, 期望范数: 1.56e+02, 实际范数: 1.56e+02)
   scale                ✅ (最大差异: 2.11e-06, 期望范数: 8.90e+01, 实际范数: 8.90e+01)
   density              ✅ (最大差异: 1.89e-06, 期望范数: 4.56e+01, 实际范数: 4.56e+01)
   features_albedo      ✅ (最大差异: 2.78e-06, 期望范数: 3.21e+02, 实际范数: 3.21e+02)
   features_specular    ✅ (最大差异: 1.45e-06, 期望范数: 1.98e+02, 实际范数: 1.98e+02)

🏆 🎉 整体通过
🌳 ======================================================================
```

## 故障排除

### 前向渲染测试调试

1. **使用详细模式**
   ```bash
   python test_kbuffer_modification.py --mode verify \
       --baseline baselines/treehill_forward.pt --verbose
   ```

2. **调整容忍度**
   ```bash
   python test_kbuffer_modification.py --mode verify \
       --baseline baselines/treehill_forward.pt \
       --tolerance-rgb 1e-4 --tolerance-depth 1e-4
   ```

### 反向传播测试调试

1. **使用详细模式**
   ```bash
   python test_kbuffer_modification.py --mode verify-backward \
       --baseline baselines/treehill_backward.pt --verbose
   ```

2. **调整梯度容忍度**
   ```bash
   python test_kbuffer_modification.py --mode verify-backward \
       --baseline baselines/treehill_backward.pt \
       --tolerance-grad 1e-4
   ```

3. **快速测试（使用下采样）**
   ```bash
   # 先捕获8x下采样版本
python test_kbuffer_modification.py --mode capture-backward \
    --checkpoint ../../runs/treehill_3dgut_15k/treehill-0809_184751/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_backward_8x.pt \
    --downsample-factor 8
   
   # 快速验证
   python test_kbuffer_modification.py --mode verify-backward \
       --baseline baselines/treehill_backward_8x.pt \
       --downsample-factor 8
   ```

### 常见问题

1. **前向输出不一致**: 检查renderForward kernel是否被意外修改
2. **梯度差异过大**: 
   - 检查renderBackward kernel的修改逻辑
   - 确认浮点运算精度
   - 验证梯度累积逻辑
3. **内存不足**: 使用更大的下采样因子（4x或8x）
4. **基准文件类型错误**: 确保使用正确的基准文件（forward vs backward）

## 文件结构

```
threedgut_tracer/tests/
├── README.md                          # 这个文件
├── kbuffer_tester.py                  # 核心测试器类
├── test_kbuffer_modification.py       # 主测试脚本
├── utils.py                           # 工具函数 (已简化)
└── baselines/                         # 基准数据目录
    └── treehill_onestep.pt            # 捕获的基准文件
```

## 核心原理

这个测试工具支持两种测试模式：前向渲染测试和反向传播测试。

### 前向渲染测试 (renderForward kernel)

核心思想是对比修改前后的前向渲染输出：

1. **捕获阶段**: 运行原始的render函数，保存输入参数和输出结果
   - 使用真实的ColmapDataset加载TreeHill数据集
   - 从checkpoint中提取训练好的高斯粒子参数
   - 保存render函数的完整输入输出数据

2. **验证阶段**: 使用相同的输入参数运行修改后的render函数，对比输出差异
   - 重新加载保存的输入参数
   - 运行修改后的renderForward代码
   - 确保使用相同的确定性环境

3. **结果对比**: 计算RGB、深度、命中计数等输出的最大和平均差异
   - RGB输出 (RGBA channels)
   - 深度输出 (hit distance)
   - 命中计数 (hits count per pixel)
   - 粒子可见性 (Gaussian visibility)

4. **通过判定**: 根据预设的容忍度判断修改是否正确
   - RGB容忍度: 1e-3 (默认)
   - 深度容忍度: 1e-3 (默认)
   - 命中计数: 完全一致 (1e-6)

### 反向传播测试 (renderBackward kernel)

核心思想是对比修改前后的梯度计算结果：

1. **捕获阶段**: 运行完整的前向+反向传播，保存所有参数的梯度
   - 启用requires_grad=True对所有Gaussian参数
   - 运行前向传播得到渲染结果
   - 构造损失函数（MSE loss with ground truth）
   - 执行backward()计算所有参数的梯度
   - 保存完整的前向输出和梯度信息

2. **验证阶段**: 使用相同的输入运行修改后的反向传播，对比梯度差异
   - 重建相同的模型状态和输入数据
   - 运行修改后的renderBackward代码
   - 使用相同的损失函数进行反向传播
   - 确保相同的确定性环境

3. **结果对比**: 分别验证前向输出和梯度
   - 前向输出应完全一致（防止意外修改renderForward）
   - 损失值应完全一致
   - 对比所有参数的梯度差异（positions, rotation, scale, density, features等）

4. **通过判定**: 
   - 前向输出: 完全一致 (1e-6)
   - 损失值: 完全一致 (1e-6)
   - 梯度容忍度: 1e-5 (默认，可调整)

### 技术细节

- **数据集**: 使用真实的MipNeRF360 TreeHill数据集，通过ColmapDataset加载
- **模型**: 支持从checkpoint自动提取训练好的MixtureOfGaussians参数
- **确定性**: 通过固定随机种子和CUDA设置确保结果可重现
- **内存管理**: 避开复杂的CUDA内存句柄，直接在PyTorch层面操作
- **兼容性**: 支持不同的checkpoint格式和数据集下采样

这种方法避开了复杂的CUDA内存管理，直接在PyTorch层面进行端到端验证，既简单又可靠。

ncu -gencode arch=compute_90,code=sm_90, 



```python
python test_kbuffer_modification.py --mode capture \
   --checkpoint ../../runs/treehill_3dgut_15k/treehill-0809_184751/ckpt_last.pt \
   --dataset ../../data/mipnerf360/treehill \
   --output baselines/treehill_15k_8x.pt \
   --downsample-factor 8


# 运行8x版本的验证和统计
python test_kbuffer_modification.py --mode verify \
    --baseline baselines/treehill_15k_8x.pt \
    --downsample-factor 8 \
    --show-gaussian-stats \
    --save-heatmap analysis/treehill_8x_heatmap.png
```