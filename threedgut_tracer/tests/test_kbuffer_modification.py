#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import os
import logging
import torch
from pathlib import Path
from typing import Optional

# 添加项目路径到Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加threedgut_tracer路径到Python path
threedgut_tracer_root = Path(__file__).parent.parent
sys.path.insert(0, str(threedgut_tracer_root))

try:
    from kbuffer_tester import TreeHillKBufferTester
except ImportError:
    try:
        from tests.kbuffer_tester import TreeHillKBufferTester
    except ImportError:
        from threedgut_tracer.tests.kbuffer_tester import TreeHillKBufferTester


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def _print_detailed_gaussian_analysis(stats):
    """打印详细的高斯点处理分析"""
    import numpy as np
    
    # 性能分析
    total_hits = stats['total_gaussian_hits']
    valid_pixels = stats['valid_pixels']
    total_pixels = stats['total_pixels']
    
    print("🔍 性能分析:")
    print(f"   计算效率: {valid_pixels/total_pixels:.1%} (有效像素比例)")
    print(f"   渲染复杂度: {total_hits/1000000:.1f}M 高斯点击中")
    if valid_pixels > 0:
        avg_complexity = total_hits / valid_pixels
        print(f"   场景复杂度: {avg_complexity:.1f} 高斯点/有效像素")
    print()
    
    # 渲染质量分析
    print("🎨 渲染质量分析:")
    max_hits = stats['hits_per_pixel']['max']
    mean_hits = stats['valid_hits_per_pixel']['mean']
    
    if max_hits > 0:
        coverage_quality = min(mean_hits / max_hits, 1.0)
        print(f"   覆盖均匀性: {coverage_quality:.1%} (越高越均匀)")
    
    std_hits = stats['valid_hits_per_pixel']['std']
    if mean_hits > 0:
        variation_coeff = std_hits / mean_hits
        print(f"   变异系数: {variation_coeff:.2f} (越低越稳定)")
    
    # 内存使用分析
    print()
    print("💾 内存访问分析:")
    if stats.get('resolution'):
        resolution = stats['resolution']
        total_memory_access = total_hits * 64  # 假设每次击中64字节数据访问
        print(f"   估算内存访问: {total_memory_access/1024/1024:.1f} MB")
        print(f"   访问密度: {total_hits/resolution[0]/resolution[1]:.1f} 次/像素")
    print()
    
    # 分布分析
    histogram = stats['histogram']
    if histogram['counts'] and len(histogram['counts']) > 0:
        counts = np.array(histogram['counts'])
        total_count = counts.sum()
        
        print("📊 分布特征:")
        # 找到主要分布区间
        if total_count > 0:
            cumsum = np.cumsum(counts) / total_count
            p50_idx = np.argmax(cumsum >= 0.5)
            p95_idx = np.argmax(cumsum >= 0.95)
            
            bins = histogram['bin_edges']
            if p50_idx < len(bins) - 1:
                print(f"   50%像素处理: ≤{bins[p50_idx + 1]:.0f} 个高斯点")
            if p95_idx < len(bins) - 1:
                print(f"   95%像素处理: ≤{bins[p95_idx + 1]:.0f} 个高斯点")
        
        # 识别渲染模式
        zero_hits = counts[0] if len(counts) > 0 else 0
        background_ratio = zero_hits / total_count if total_count > 0 else 0
        
        if background_ratio > 0.3:
            print(f"   渲染模式: 主要为背景渲染 ({background_ratio:.1%} 背景像素)")
        elif mean_hits < 10:
            print(f"   渲染模式: 简单场景 (平均{mean_hits:.1f}次击中)")
        elif mean_hits < 50:
            print(f"   渲染模式: 中等复杂度 (平均{mean_hits:.1f}次击中)")
        else:
            print(f"   渲染模式: 高复杂度场景 (平均{mean_hits:.1f}次击中)")


def _save_hits_heatmap(raw_outputs, stats, output_path):
    """保存hits_count热力图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        import os
        
        # 解决中文字符显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 从原始输出获取hits_count张量数据
        hits_count_tensor = raw_outputs['hits_count']
        hits_count = hits_count_tensor.cpu().numpy()
        
        # 更健壮的维度处理，避免squeeze错误
        original_shape = hits_count.shape
        print(f"🔍 处理hits_count张量，原始维度: {original_shape}")
        
        # 递归处理维度直到得到2D数组
        while len(hits_count.shape) > 2:
            # 找到第一个size为1的维度
            size_1_dims = [i for i, size in enumerate(hits_count.shape) if size == 1]
            if size_1_dims:
                hits_count = np.squeeze(hits_count, axis=size_1_dims[0])
            else:
                # 如果没有size为1的维度，取第一个slice
                if len(hits_count.shape) == 4:  # [B, C, H, W]
                    hits_count = hits_count[0, 0]
                elif len(hits_count.shape) == 3:  # [B, H, W] 或 [C, H, W]
                    hits_count = hits_count[0]
                else:
                    break
        
        print(f"🔍 处理后hits_count维度: {hits_count.shape}")
        
        # 确保输出目录存在
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 确定图像尺寸 (H, W)
        height, width = hits_count.shape[0], hits_count.shape[1]
        print(f"🖼️ 生成热力图: {height} × {width} (高 × 宽)")
        
        # 左图：原始hits_count (默认origin='upper'，图像顶部对应数组第0行)
        im1 = axes[0].imshow(hits_count, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Gaussian Hits per Pixel\nResolution: {width} x {height}')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Height (pixels)')
        plt.colorbar(im1, ax=axes[0], label='Hit Count')
        
        # 右图：对数尺度（突出低值区域）
        hits_count_log = np.log10(hits_count + 1)  # +1避免log(0)
        im2 = axes[1].imshow(hits_count_log, cmap='plasma', aspect='auto')
        axes[1].set_title(f'Hit Count (Log Scale)\nlog10(hits + 1)')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Height (pixels)')
        plt.colorbar(im2, ax=axes[1], label='log10(Hit Count + 1)')
        
        # 添加统计信息
        stats_text = f"""Statistics:
Mean: {stats['hits_per_pixel']['mean']:.1f}
Max: {stats['hits_per_pixel']['max']:.0f}
Total Hits: {stats['total_gaussian_hits']:,.0f}
Valid Pixels: {stats['valid_pixel_ratio']:.1%}"""
        
        fig.text(0.02, 0.98, stats_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("⚠️ matplotlib未安装，无法保存热力图")
        print("💡 安装命令: pip install matplotlib")
    except Exception as e:
        raise Exception(f"保存热力图时出错: {e}")


def capture_baseline(args):
    """捕获基准模式"""
    print("🎯 K-Buffer基准捕获模式")
    print("-" * 50)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return False
    
    # 检查数据集路径
    if args.dataset and not os.path.exists(args.dataset):
        print(f"❌ 数据集路径不存在: {args.dataset}")
        return False
    
    tester = TreeHillKBufferTester(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    print(f"🌳 使用TreeHill数据集: {args.dataset}")
    print(f"🏋️ 使用Checkpoint: {args.checkpoint}")
    downsample_factor = getattr(args, 'downsample_factor', 1)
    if downsample_factor > 1:
        print(f"📐 下采样因子: {downsample_factor}x (分辨率将为原始的1/{downsample_factor})")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 捕获基准
    try:
        baseline = tester.capture_baseline_outputs(
            output_path=args.output,
            test_view_id=args.view_id
        )
        
        print("\n✅ 基准捕获完成!")
        print(f"📂 保存位置: {args.output}")
        print(f"📊 场景信息: {baseline['metadata']['resolution']} 分辨率")
        print("\n💡 下一步: 修改evalKBuffer代码后运行验证模式")
        print(f"   python {sys.argv[0]} --mode verify --baseline {args.output}")
        return True
        
    except Exception as e:
        print(f"❌ 基准捕获失败: {e}")
        return False


def capture_backward_baseline(args):
    """捕获反向传播基准模式"""
    print("🎯 K-Buffer反向传播基准捕获模式")
    print("-" * 50)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return False
    
    # 检查数据集路径
    if args.dataset and not os.path.exists(args.dataset):
        print(f"❌ 数据集路径不存在: {args.dataset}")
        return False
    
    tester = TreeHillKBufferTester(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    print(f"🌳 使用TreeHill数据集: {args.dataset}")
    print(f"🏋️ 使用Checkpoint: {args.checkpoint}")
    downsample_factor = getattr(args, 'downsample_factor', 1)
    if downsample_factor > 1:
        print(f"📐 下采样因子: {downsample_factor}x (分辨率将为原始的1/{downsample_factor})")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 捕获反向传播基准
    try:
        baseline = tester.capture_backward_baseline(
            output_path=args.output,
            test_view_id=args.view_id
        )
        
        print("\n✅ 反向传播基准捕获完成!")
        print(f"📂 保存位置: {args.output}")
        print(f"📊 场景信息: {baseline['metadata']['resolution']} 分辨率")
        print(f"📉 损失值: {baseline['loss']:.6f}")
        print("\n💡 下一步: 修改renderBackward代码后运行反向验证模式")
        print(f"   python {sys.argv[0]} --mode verify-backward --baseline {args.output}")
        return True
        
    except Exception as e:
        print(f"❌ 反向传播基准捕获失败: {e}")
        return False


def verify_backward_modification(args):
    """验证反向传播修改模式"""
    print("🔍 K-Buffer反向传播修改验证模式") 
    print("-" * 50)
    
    if not os.path.exists(args.baseline):
        print(f"❌ 基准文件不存在: {args.baseline}")
        print("💡 请先运行捕获反向基准模式创建基准文件")
        return False
    
    # 从基准文件中读取原始的checkpoint和dataset路径
    try:
        baseline = torch.load(args.baseline, map_location='cpu', weights_only=False)
        
        # 检查是否为backward基准文件
        if baseline['metadata'].get('test_type') != 'backward':
            print(f"❌ 基准文件不是反向传播基准文件")
            print(f"   文件类型: {baseline['metadata'].get('test_type', 'unknown')}")
            print("💡 请使用 --mode capture-backward 创建反向传播基准文件")
            return False
            
        checkpoint_path = baseline['metadata']['checkpoint_path']
        dataset_path = baseline['metadata']['dataset_path']
        
        print(f"📂 从基准文件获取路径:")
        print(f"   🏋️ Checkpoint: {checkpoint_path}")
        print(f"   🌳 Dataset: {dataset_path}")
        
    except Exception as e:
        print(f"❌ 无法读取基准文件: {e}")
        return False
    
    tester = TreeHillKBufferTester(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    # 验证反向传播修改
    try:
        passed, results = tester.verify_backward_modification(
            args.baseline,
            tolerance_grad=args.tolerance_grad
        )
        
        if passed:
            print("\n🎉 恭喜! K-Buffer反向传播修改验证通过!")
            print("✅ 所有梯度都在允许的误差范围内")
            return True
        else:
            print("\n⚠️ K-Buffer反向传播修改验证失败!")
            print("❌ 发现超出容忍度的梯度差异")
            print("💡 建议:")
            print("   1. 检查renderBackward的修改逻辑")
            print("   2. 确认浮点运算精度问题")
            print("   3. 调整梯度容忍度参数（如果差异很小）")
            print("   4. 确保renderForward kernel未被修改（前向输出应完全一致）")
            return False
            
    except Exception as e:
        print(f"❌ 反向传播验证过程出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def verify_modification(args):
    """验证修改模式"""
    print("🔍 K-Buffer修改验证模式")
    print("-" * 50)
    
    if not os.path.exists(args.baseline):
        print(f"❌ 基准文件不存在: {args.baseline}")
        print("💡 请先运行捕获模式创建基准文件")
        return False
    
    # 从基准文件中读取原始的checkpoint和dataset路径
    try:
        baseline = torch.load(args.baseline, map_location='cpu', weights_only=False)
        checkpoint_path = baseline['metadata']['checkpoint_path']
        dataset_path = baseline['metadata']['dataset_path']
        
        print(f"📂 从基准文件获取路径:")
        print(f"   🏋️ Checkpoint: {checkpoint_path}")
        print(f"   🌳 Dataset: {dataset_path}")
        
    except Exception as e:
        print(f"❌ 无法读取基准文件: {e}")
        return False
    
    tester = TreeHillKBufferTester(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    # 验证修改
    try:
        passed, results = tester.verify_modification(
            args.baseline,
            tolerance_rgb=args.tolerance_rgb,
            tolerance_depth=args.tolerance_depth
        )
        
        # 如果请求显示统计信息，打印额外的高斯点处理统计
        if args.show_gaussian_stats and 'gaussian_processing_stats' in results:
            print("\n" + "="*60)
            print("📊 详细高斯点处理统计")
            print("="*60)
            stats = results['gaussian_processing_stats']
            _print_detailed_gaussian_analysis(stats)
        
        # 如果请求保存热力图
        if args.save_heatmap and 'raw_outputs' in results:
            try:
                _save_hits_heatmap(results['raw_outputs'], results['gaussian_processing_stats'], args.save_heatmap)
                print(f"\n📊 热力图已保存: {args.save_heatmap}")
            except Exception as e:
                print(f"\n⚠️ 保存热力图失败: {e}")
        
        if passed:
            print("\n🎉 恭喜! K-Buffer修改验证通过!")
            print("✅ 所有输出都在允许的误差范围内")
            return True
        else:
            print("\n⚠️ K-Buffer修改验证失败!")
            print("❌ 发现超出容忍度的差异")
            print("💡 建议:")
            print("   1. 检查evalKBuffer的修改逻辑")
            print("   2. 确认浮点运算精度问题")
            print("   3. 调整容忍度参数（如果差异很小）")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="K-Buffer修改测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 捕获前向基准 (需要先训练好TreeHill模型)
  python test_kbuffer_modification.py --mode capture \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --dataset data/mipnerf360/treehill \\
    --output baselines/treehill_onestep.pt
  
  # 验证前向修改 (修改renderForward代码后)
  python test_kbuffer_modification.py --mode verify \\
    --baseline baselines/treehill_onestep.pt
  
  # 捕获反向传播基准 (用于测试renderBackward kernel)
  python test_kbuffer_modification.py --mode capture-backward \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --dataset data/mipnerf360/treehill \\
    --output baselines/treehill_backward.pt
  
  # 验证反向传播修改 (修改renderBackward代码后)
  python test_kbuffer_modification.py --mode verify-backward \\
    --baseline baselines/treehill_backward.pt \\
    --tolerance-grad 1e-6
  
  # 详细验证 
  python test_kbuffer_modification.py --mode verify \\
    --baseline baselines/treehill_onestep.pt \\
    --tolerance-rgb 1e-4 --tolerance-depth 1e-4 --verbose
    
  # 带高斯点统计和热力图
  python test_kbuffer_modification.py --mode verify \\
    --baseline baselines/treehill_onestep.pt \\
    --show-gaussian-stats --save-heatmap heatmaps/gaussian_hits.png
    
  # 8倍下采样版本 (快速测试) - 前向和反向
  python test_kbuffer_modification.py --mode capture \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --output baselines/treehill_onestep_8x.pt \\
    --downsample-factor 8
    
  python test_kbuffer_modification.py --mode capture-backward \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --output baselines/treehill_backward_8x.pt \\
    --downsample-factor 8
    
  python test_kbuffer_modification.py --mode verify-backward \\
    --baseline baselines/treehill_backward_8x.pt \\
    --downsample-factor 8 --tolerance-grad 1e-5
    
注意: 
  - 需要先训练好TreeHill数据集的模型
  - 只支持真实数据集测试，确保数据集路径正确
        """
    )
    
    # 基本参数
    parser.add_argument('--mode', 
                       choices=['capture', 'verify', 'capture-backward', 'verify-backward'], 
                       required=True,
                       help='运行模式: capture(捕获前向基准), verify(验证前向修改), capture-backward(捕获反向基准), verify-backward(验证反向修改)')
    
    parser.add_argument('--config',
                       type=str,
                       default=None,
                       help='配置文件路径 (默认: apps/colmap_3dgut.yaml)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='详细日志输出')
    
    # 捕获模式参数
    capture_group = parser.add_argument_group('捕获模式参数')
    capture_group.add_argument('--checkpoint', '-c',
                              type=str,
                              help='训练好的TreeHill模型checkpoint路径 (capture模式必需)')
    
    capture_group.add_argument('--dataset', '-d',
                              type=str,
                              default='data/mipnerf360/treehill',
                              help='TreeHill数据集路径')
    
    capture_group.add_argument('--output', '-o',
                              type=str,
                              default='baselines/treehill_onestep.pt',
                              help='基准文件输出路径')
    
    capture_group.add_argument('--view-id',
                              type=int,
                              default=5,
                              help='测试视角ID')
    
    # 验证模式参数
    verify_group = parser.add_argument_group('验证模式参数')
    verify_group.add_argument('--baseline', '-b',
                             type=str,
                             default='baselines/treehill_onestep.pt',
                             help='基准文件路径')
    
    verify_group.add_argument('--tolerance-rgb',
                             type=float,
                             default=1e-3,
                             help='RGB差异容忍度')
    
    verify_group.add_argument('--tolerance-depth',
                             type=float,
                             default=1e-3,
                             help='深度差异容忍度')
    
    verify_group.add_argument('--show-gaussian-stats',
                             action='store_true',
                             help='显示每个像素处理的高斯点数量统计')
    
    verify_group.add_argument('--save-heatmap',
                             type=str,
                             help='保存hits_count热力图到指定路径 (例: heatmaps/hits_visualization.png)')
    
    # 反向传播测试参数
    backward_group = parser.add_argument_group('反向传播测试参数')
    backward_group.add_argument('--tolerance-grad',
                               type=float,
                               default=1e-5,
                               help='梯度差异容忍度 (verify-backward模式)')
    
    # 通用参数 - 适用于所有模式
    parser.add_argument('--downsample-factor',
                        type=int,
                        default=1,
                        choices=[1, 2, 4, 8],
                        help='图像下采样因子: 1=原分辨率(5068×3326), 2=1/2分辨率, 4=1/4分辨率, 8=1/8分辨率')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode in ['capture', 'capture-backward'] and not args.checkpoint:
        print(f"❌ {args.mode}模式需要提供--checkpoint参数")
        print("💡 示例: --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt")
        sys.exit(1)
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        if args.mode == 'capture':
            success = capture_baseline(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'verify':
            success = verify_modification(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'capture-backward':
            success = capture_backward_baseline(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'verify-backward':
            success = verify_backward_modification(args)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 执行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# NCU性能分析命令
# 
# 1. 基础性能分析:
# ncu --set full \
#     --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --export tests/render_full_analysis \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k.pt
#
# 2. 负载均衡专项分析:
# ncu --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --metrics sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,smsp__sass_branch_targets_threads_diverged.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warps_active.avg.pct_of_peak_sustained_active,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
#     --page details \
#     --export tests/render_load_balance_analysis \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k.pt
#
# 3. 带源码的详细分析:
# ncu --set full \
#     --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --import-source on \
#     --source-folders include,src \
#     --export tests/render_full_analysis \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k.pt

# ========== 8倍下采样版本NCU分析命令 (快速分析) ==========
#
# 1. 基础性能分析 (8x下采样，快速):
# ncu --set full \
#     --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --export tests/render_8x_analysis \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k_8x.pt --downsample-factor 8
#
# 2. 负载均衡专项分析 (8x下采样):
# ncu --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --metrics sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,smsp__sass_branch_targets_threads_diverged.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warps_active.avg.pct_of_peak_sustained_active,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,smsp__warps_eligible.avg.pct_of_peak_sustained_active \
#     --page details \
#     --export tests/render_8x_load_balance \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k_8x.pt --downsample-factor 8
#
# 3. 内存访问模式分析 (8x下采样):
# ncu --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --metrics l1tex__t_bytes.sum,l1tex__t_requests.sum,l1tex__t_sectors.sum,dram__bytes.sum,dram__sectors.sum,l1tex__t_sector_hit_rate.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
#     --page details \
#     --export tests/render_8x_memory \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k_8x.pt --downsample-factor 8
#
# 4. 计算utilization分析 (8x下采样):
# ncu --target-processes all \
#     --kernel-name-base function \
#     -k regex:render \
#     --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
#     --page details \
#     --export tests/render_8x_compute \
#     --force-overwrite \
#     python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k_8x.pt --downsample-factor 8
#
# 5. 完整源码级分析 (8x下采样，最详细):
# ncu -f -o /home/scratch.sarawang_ent/3dgrut/threedgut_tracer/tests/render_8x_balanced.ncu-rep --set full --target-processes all --kernel-name-base function -k regex:render --import-source on --source-folders include,src python tests/test_kbuffer_modification.py --mode verify --baseline tests/baselines/treehill_15k_8x.pt
