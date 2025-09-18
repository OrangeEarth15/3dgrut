# Rendering Optimization Test Framework

This test framework validates rendering optimization strategies for k_buffer_size=0 (unsorted mode), including load balancing and other performance optimizations. It provides end-to-end testing based on the real TreeHill dataset, supporting both forward rendering and backward propagation test modes.

**Important**: This framework is specifically designed for unsorted rendering mode (k_buffer_size=0) optimization validation, not for K-Buffer sorted mode testing.

## Quick Start

### 1. Train TreeHill Model

First, you need to train a TreeHill dataset model:

```bash
# # Train TreeHill model with 8x downsampled images
# python train.py --config-name apps/colmap_3dgut.yaml \
#     path=data/mipnerf360/treehill \
#     out_dir=runs \
#     experiment_name=treehill_3dgut_15k_8x \
#     n_iterations=15000 \
#     dataset.downsample_factor=8

# Train TreeHill model for forward testing checkpoint
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/mipnerf360/treehill \
    out_dir=runs \
    experiment_name=treehill_3dgut_15k \
    n_iterations=15000

# Example checkpoint path: /home/scratch.sarawang_ent/3dgrut/runs/treehill_3dgut_15k/treehill-0809_184751/ckpt_last.pt
```

### 2. Capture Baseline Data

#### Forward Rendering Optimization Testing

Before modifying rendering optimization code, capture the original version's output as baseline:

```bash
cd threedgut_tracer/test_optimization

# # Use original resolution trained model for original resolution inference
# python test_optimization.py --mode capture \
#     --checkpoint ../../runs/treehill_3dgut_15k/treehill-1709_073359/ckpt_last.pt \
#     --dataset ../../data/mipnerf360/treehill \
#     --output baselines/treehill_15k.pt

# Use original resolution trained model for 8x downsampled resolution inference
python test_optimization.py --mode capture \
    --checkpoint ../../runs/treehill_3dgut_15k/treehill-1709_073359/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_15k_8x.pt --downsample-factor 8

# python test_optimization.py --mode capture \
#     --checkpoint ../../runs/treehill_3dgut_15k/treehill-1709_073359/ckpt_last.pt \
#     --dataset ../../data/mipnerf360/treehill \
#     --output baselines/treehill_15k_8x_view10.pt --downsample-factor 8 --view-id 10

# # New baseline: 8x downsampled trained checkpoint for 8x downsampled resolution inference
# python test_optimization.py --mode capture \
#     --checkpoint ../../runs/treehill_3dgut_15k_8x/treehill-0809_183156/ckpt_last.pt \
#     --dataset ../../data/mipnerf360/treehill \
#     --output baselines/treehill_8x_forward_8x.pt --downsample-factor 8
```

#### Backward Propagation Optimization Testing

For testing backward propagation optimizations, capture backward baseline:

```bash
# # 8x downsampled trained checkpoint backward propagation for 8x downsampled resolution
# python test_optimization.py --mode capture-backward \
#     --checkpoint ../../runs/treehill_3dgut_15k_8x/treehill-0809_183156/ckpt_last.pt \
#     --dataset ../../data/mipnerf360/treehill \
#     --output baselines/treehill_8x_backward_8x.pt --downsample-factor 8

# Original resolution trained checkpoint backward propagation for 8x downsampled resolution
python test_optimization.py --mode capture-backward \
    --checkpoint ../../runs/treehill_3dgut_15k/treehill-1709_073359/ckpt_last.pt \
    --dataset ../../data/mipnerf360/treehill \
    --output baselines/treehill_backward_8x.pt --downsample-factor 8
```


### 3. Verify Modifications

#### Verify Forward Rendering Optimization

```bash
# Verify new baseline
python test_optimization.py --mode verify \
    --baseline baselines/treehill_forward.pt

# Verify original resolution version
python test_optimization.py --mode verify \
    --baseline baselines/treehill_15k.pt

# Verify 8x downsampled version (--downsample-factor 8 is optional, baseline file contains resolution info)
python test_optimization.py --mode verify \
    --baseline baselines/treehill_15k_8x.pt
```

#### Verify Backward Propagation Optimization

```bash
# Verify original resolution checkpoint backward propagation 8x downsampled version
python test_optimization.py --mode verify-backward \
    --baseline baselines/treehill_backward_8x.pt

# Verify 8x downsampled checkpoint backward propagation 8x downsampled version
python test_optimization.py --mode verify-backward \
    --baseline baselines/treehill_8x_backward_8x.pt
```

## Command Line Arguments

### Forward Rendering Optimization Capture Mode (capture)

- `--checkpoint, -c`: Trained TreeHill model checkpoint path **(required)**
- `--dataset, -d`: TreeHill dataset path (default: `data/mipnerf360/treehill`)
- `--output, -o`: Baseline file output path (default: `baselines/treehill_onestep.pt`)
- `--view-id`: Test view ID (default: 5)

### Forward Rendering Optimization Verification Mode (verify)

- `--baseline, -b`: Baseline file path (default: `baselines/treehill_onestep.pt`)
- `--tolerance-rgb`: RGB difference tolerance (default: 1e-3)
- `--tolerance-depth`: Depth difference tolerance (default: 1e-3)
- `--show-gaussian-stats`: Show Gaussian processing statistics per pixel
- `--save-heatmap`: Save hits_count heatmap to specified path

### Backward Propagation Optimization Capture Mode (capture-backward)

- `--checkpoint, -c`: Trained TreeHill model checkpoint path **(required)**
- `--dataset, -d`: TreeHill dataset path (default: `data/mipnerf360/treehill`)
- `--output, -o`: Backward propagation baseline file output path
- `--view-id`: Test view ID (default: 5)

### Backward Propagation Optimization Verification Mode (verify-backward)

- `--baseline, -b`: Backward propagation baseline file path
- `--tolerance-grad`: Gradient difference tolerance (default: 1e-5)

### Common Parameters

- `--config`: Config file path (default: `apps/colmap_3dgut.yaml`)
- `--downsample-factor`: Image downsample factor (1/2/4/8, default: 1) **Only needed for capture mode, optional for verify mode**
- `--verbose, -v`: Verbose logging output


## Analysis and Profiling

### Heatmap Analysis

```bash
python test_optimization.py --mode capture \
   --checkpoint ../../runs/treehill_3dgut_15k/treehill-0809_184751/ckpt_last.pt \
   --dataset ../../data/mipnerf360/treehill \
   --output baselines/treehill_15k_8x.pt \
   --downsample-factor 8

# Run 8x version verification and statistics
python test_optimization.py --mode verify \
    --baseline baselines/treehill_15k_8x.pt \
    --show-gaussian-stats \
    --save-heatmap analysis/treehill_8x_heatmap.png
```

### NCU Performance Analysis

**Note**: The `-lineinfo` compilation flag enables mapping between SASS assembly code and source code, allowing NCU to provide detailed source-level performance analysis including:
- Line-by-line execution statistics
- Register usage per source line
- Memory access patterns mapped to source code
- Warp efficiency analysis at source level

setup_3dgut.py has been updated with `-lineinfo` option for this functionality.

```bash
# Original full resolution version forward analysis
ncu --set full \
    --target-processes all \
    --kernel-name-base function \
    -k regex:render \
    --import-source on \
    --source-folders include,src \
    --export test_optimization/render_full_analysis \
    --force-overwrite \
    python test_optimization/test_optimization.py --mode verify --baseline test_optimization/baselines/treehill_15k.pt

# Complete source-level analysis (8x downsampled, most detailed):
ncu -f \
    -o /home/scratch.sarawang_ent/3dgrut/threedgut_tracer/test_optimization/render_8x_balanced.ncu-rep \
    --set full \
    --target-processes all \
    --kernel-name-base function \
    -k regex:render \
    --import-source on \
    --source-folders include,src \
    python test_optimization/test_optimization.py --mode verify --baseline test_optimization/baselines/treehill_15k_8x.pt
```

### NCU Backward Propagation Analysis

```bash
# use .ncu-rep format output
ncu -f \
    -o /home/scratch.sarawang_ent/3dgrut/threedgut_tracer/test_optimization/render_backward_8x.ncu-rep \
    --set full \
    --target-processes all \
    --kernel-name-base function \
    -k regex:render \
    --import-source on \
    --source-folders include,src \
    python test_optimization/test_optimization.py --mode verify-backward --baseline test_optimization/baselines/treehill_backward_8x.pt
```

## Framework Overview

This testing framework is specifically designed for validating rendering optimizations in **k_buffer_size=0 (unsorted mode)**, including:

- **Load balancing strategy optimizations**
- **Rendering performance optimizations** 
- **Other algorithmic improvements for unsorted mode**

The framework captures baseline data before optimization, then verifies that optimized versions produce consistent results within acceptable tolerances.

### Test Views

The framework uses TreeHill dataset test views selected with `test_split_interval=8`:
- Test view 0: _DSC8874.JPG (original index 0)
- Test view 1: _DSC8882.JPG (original index 8)
- Test view 2: _DSC8890.JPG (original index 16)
- Test view 3: _DSC8898.JPG (original index 24)
- Test view 4: _DSC8906.JPG (original index 32)
- **Test view 5: _DSC8914.JPG (original index 40)** ← Default test view
- Test view 6: _DSC8922.JPG (original index 48)
- ...and so on (total 18 test views)

> **Notes**:
> - The `--downsample-factor` parameter is optional in verify mode since baseline files contain correct resolution data. However, capture mode must specify the correct downsample parameter.
> - This test framework is specifically for k_buffer_size=0 (unsorted mode) rendering optimization validation. Ensure k_buffer_size is set to 0 in config files.
> - Forward output differences are acceptable in backward verification mode when using load balancing optimizations - the key metric is gradient consistency.
