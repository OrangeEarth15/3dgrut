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

# Add project paths to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add threedgut_tracer path to Python path
threedgut_tracer_root = Path(__file__).parent.parent
sys.path.insert(0, str(threedgut_tracer_root))

try:
    from render_tester import TreeHillRenderOptimizationTester
except ImportError:
    try:
        from test_optimization.render_tester import TreeHillRenderOptimizationTester
    except ImportError:
        from threedgut_tracer.test_optimization.render_tester import TreeHillRenderOptimizationTester


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def _print_detailed_gaussian_analysis(stats):
    """Print detailed Gaussian processing analysis"""
    import numpy as np
    
    # Performance analysis
    total_hits = stats['total_gaussian_hits']
    valid_pixels = stats['valid_pixels']
    total_pixels = stats['total_pixels']
    
    print("🔍 Performance Analysis:")
    print(f"   Computation Efficiency: {valid_pixels/total_pixels:.1%} (valid pixel ratio)")
    print(f"   Rendering Complexity: {total_hits/1000000:.1f}M gaussian hits")
    if valid_pixels > 0:
        avg_complexity = total_hits / valid_pixels
        print(f"   Scene Complexity: {avg_complexity:.1f} gaussians/valid pixel")
    print()
    
    # Rendering quality analysis
    print("🎨 Rendering Quality Analysis:")
    max_hits = stats['hits_per_pixel']['max']
    mean_hits = stats['valid_hits_per_pixel']['mean']
    
    if max_hits > 0:
        coverage_quality = min(mean_hits / max_hits, 1.0)
        print(f"   Coverage Uniformity: {coverage_quality:.1%} (higher is more uniform)")
    
    std_hits = stats['valid_hits_per_pixel']['std']
    if mean_hits > 0:
        variation_coeff = std_hits / mean_hits
        print(f"   Variation Coefficient: {variation_coeff:.2f} (lower is more stable)")
    
    # Memory usage analysis
    print()
    print("💾 Memory Access Analysis:")
    if stats.get('resolution'):
        resolution = stats['resolution']
        total_memory_access = total_hits * 64  # Assume 64 bytes per hit
        print(f"   Estimated Memory Access: {total_memory_access/1024/1024:.1f} MB")
        print(f"   Access Density: {total_hits/resolution[0]/resolution[1]:.1f} times/pixel")
    print()
    
    # Distribution analysis
    histogram = stats['histogram']
    if histogram['counts'] and len(histogram['counts']) > 0:
        counts = np.array(histogram['counts'])
        total_count = counts.sum()
        
        print("📊 Distribution Characteristics:")
        # Find main distribution ranges
        if total_count > 0:
            cumsum = np.cumsum(counts) / total_count
            p50_idx = np.argmax(cumsum >= 0.5)
            p95_idx = np.argmax(cumsum >= 0.95)
            
            bins = histogram['bin_edges']
            if p50_idx < len(bins) - 1:
                print(f"   50% pixels process: ≤{bins[p50_idx + 1]:.0f} gaussians")
            if p95_idx < len(bins) - 1:
                print(f"   95% pixels process: ≤{bins[p95_idx + 1]:.0f} gaussians")
        
        # Identify rendering patterns
        zero_hits = counts[0] if len(counts) > 0 else 0
        background_ratio = zero_hits / total_count if total_count > 0 else 0
        
        if background_ratio > 0.3:
            print(f"   Rendering Mode: Mainly background rendering ({background_ratio:.1%} background pixels)")
        elif mean_hits < 10:
            print(f"   Rendering Mode: Simple scene (average {mean_hits:.1f} hits)")
        elif mean_hits < 50:
            print(f"   Rendering Mode: Medium complexity (average {mean_hits:.1f} hits)")
        else:
            print(f"   Rendering Mode: High complexity scene (average {mean_hits:.1f} hits)")


def _save_hits_heatmap(raw_outputs, stats, output_path):
    """Save hits_count heatmap visualization"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        import os
        
        # Setup matplotlib for clean rendering
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Get hits_count tensor data from raw outputs
        hits_count_tensor = raw_outputs['hits_count']
        hits_count = hits_count_tensor.cpu().numpy()
        
        # Robust dimension handling to avoid squeeze errors
        original_shape = hits_count.shape
        print(f"🔍 Processing hits_count tensor, original shape: {original_shape}")
        
        # Recursively handle dimensions until we get a 2D array
        while len(hits_count.shape) > 2:
            # Find first dimension with size 1
            size_1_dims = [i for i, size in enumerate(hits_count.shape) if size == 1]
            if size_1_dims:
                hits_count = np.squeeze(hits_count, axis=size_1_dims[0])
            else:
                # If no size-1 dimensions, take first slice
                if len(hits_count.shape) == 4:  # [B, C, H, W]
                    hits_count = hits_count[0, 0]
                elif len(hits_count.shape) == 3:  # [B, H, W] or [C, H, W]
                    hits_count = hits_count[0]
                else:
                    break
        
        print(f"🔍 Processed hits_count shape: {hits_count.shape}")
        
        # Ensure output directory exists
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create heatmap visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Determine image dimensions (H, W)
        height, width = hits_count.shape[0], hits_count.shape[1]
        print(f"🖼️ Generating heatmap: {height} × {width} (height × width)")
        
        # Left plot: Original hits_count (default origin='upper', image top corresponds to array row 0)
        im1 = axes[0].imshow(hits_count, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Gaussian Hits per Pixel\nResolution: {width} x {height}')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Height (pixels)')
        plt.colorbar(im1, ax=axes[0], label='Hit Count')
        
        # Right plot: Log scale (highlight low-value regions)
        hits_count_log = np.log10(hits_count + 1)  # +1 to avoid log(0)
        im2 = axes[1].imshow(hits_count_log, cmap='plasma', aspect='auto')
        axes[1].set_title(f'Hit Count (Log Scale)\nlog10(hits + 1)')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Height (pixels)')
        plt.colorbar(im2, ax=axes[1], label='log10(Hit Count + 1)')
        
        # Add statistics information
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
        print("⚠️ matplotlib not installed, cannot save heatmap")
        print("💡 Install command: pip install matplotlib")
    except Exception as e:
        raise Exception(f"Error saving heatmap: {e}")


def capture_baseline(args):
    """Baseline capture mode"""
    print("🎯 Rendering Optimization Baseline Capture Mode")
    print("-" * 50)
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file does not exist: {args.checkpoint}")
        return False
    
    # Check dataset path
    if args.dataset and not os.path.exists(args.dataset):
        print(f"❌ Dataset path does not exist: {args.dataset}")
        return False
    
    tester = TreeHillRenderOptimizationTester(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    print(f"🌳 Using TreeHill dataset: {args.dataset}")
    print(f"🏋️ Using Checkpoint: {args.checkpoint}")
    downsample_factor = getattr(args, 'downsample_factor', 1)
    if downsample_factor > 1:
        print(f"📐 Downsample factor: {downsample_factor}x (resolution will be 1/{downsample_factor} of original)")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Capture baseline
    try:
        baseline = tester.capture_baseline_outputs(
            output_path=args.output,
            test_view_id=args.view_id
        )
        
        print("\n✅ Baseline capture completed!")
        print(f"📂 Save location: {args.output}")
        print(f"📊 Scene info: {baseline['metadata']['resolution']} resolution")
        print("\n💡 Next step: Run verification mode after modifying rendering optimization code")
        print(f"   python {sys.argv[0]} --mode verify --baseline {args.output}")
        return True
        
    except Exception as e:
        print(f"❌ Baseline capture failed: {e}")
        return False


def capture_backward_baseline(args):
    """Backward baseline capture mode"""
    print("🎯 Rendering Optimization Backward Baseline Capture Mode")
    print("-" * 50)
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file does not exist: {args.checkpoint}")
        return False
    
    # Check dataset path
    if args.dataset and not os.path.exists(args.dataset):
        print(f"❌ Dataset path does not exist: {args.dataset}")
        return False
    
    tester = TreeHillRenderOptimizationTester(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    print(f"🌳 Using TreeHill dataset: {args.dataset}")
    print(f"🏋️ Using Checkpoint: {args.checkpoint}")
    downsample_factor = getattr(args, 'downsample_factor', 1)
    if downsample_factor > 1:
        print(f"📐 Downsample factor: {downsample_factor}x (resolution will be 1/{downsample_factor} of original)")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Capture backward baseline
    try:
        baseline = tester.capture_backward_baseline(
            output_path=args.output,
            test_view_id=args.view_id
        )
        
        print("\n✅ Backward baseline capture completed!")
        print(f"📂 Save location: {args.output}")
        print(f"📊 Scene info: {baseline['metadata']['resolution']} resolution")
        print(f"📉 Loss value: {baseline['loss']:.6f}")
        print("\n💡 Next step: Run backward verification mode after modifying backward optimization code")
        print(f"   python {sys.argv[0]} --mode verify-backward --baseline {args.output}")
        return True
        
    except Exception as e:
        print(f"❌ Backward baseline capture failed: {e}")
        return False


def verify_backward_modification(args):
    """Backward modification verification mode"""
    print("🔍 Rendering Optimization Backward Verification Mode") 
    print("-" * 50)
    
    if not os.path.exists(args.baseline):
        print(f"❌ Baseline file does not exist: {args.baseline}")
        print("💡 Please run capture backward baseline mode first to create baseline file")
        return False
    
    # Read original checkpoint and dataset paths from baseline file
    try:
        baseline = torch.load(args.baseline, map_location='cpu', weights_only=False)
        
        # Check if it's a backward baseline file
        if baseline['metadata'].get('test_type') != 'backward':
            print(f"❌ Baseline file is not a backward baseline file")
            print(f"   File type: {baseline['metadata'].get('test_type', 'unknown')}")
            print("💡 Please use --mode capture-backward to create backward baseline file")
            return False
            
        checkpoint_path = baseline['metadata']['checkpoint_path']
        dataset_path = baseline['metadata']['dataset_path']
        
        print(f"📂 Getting paths from baseline file:")
        print(f"   🏋️ Checkpoint: {checkpoint_path}")
        print(f"   🌳 Dataset: {dataset_path}")
        
    except Exception as e:
        print(f"❌ Cannot read baseline file: {e}")
        return False
    
    tester = TreeHillRenderOptimizationTester(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    # Verify backward modification
    try:
        passed, results = tester.verify_backward_modification(
            args.baseline,
            tolerance_grad=args.tolerance_grad
        )
        
        if passed:
            print("\n🎉 Congratulations! Rendering optimization backward verification passed!")
            print("✅ Gradients are within tolerance range")
            print("💡 Note: Forward output differences are acceptable when using load balancing optimizations")
            return True
        else:
            print("\n⚠️ Rendering optimization backward verification failed!")
            print("❌ Found gradient differences exceeding tolerance")
            print("💡 Suggestions:")
            print("   1. Check backward optimization modification logic")
            print("   2. Verify floating point precision issues")
            print("   3. Adjust gradient tolerance parameter (if differences are small)")
            print("   4. Forward output differences are acceptable, focus on gradient consistency")
            return False
            
    except Exception as e:
        print(f"❌ Error during backward verification process: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def verify_modification(args):
    """Modification verification mode"""
    print("🔍 Rendering Optimization Verification Mode")
    print("-" * 50)
    
    if not os.path.exists(args.baseline):
        print(f"❌ Baseline file does not exist: {args.baseline}")
        print("💡 Please run capture mode first to create baseline file")
        return False
    
    # Read original checkpoint and dataset paths from baseline file
    try:
        baseline = torch.load(args.baseline, map_location='cpu', weights_only=False)
        checkpoint_path = baseline['metadata']['checkpoint_path']
        dataset_path = baseline['metadata']['dataset_path']
        
        print(f"📂 Getting paths from baseline file:")
        print(f"   🏋️ Checkpoint: {checkpoint_path}")
        print(f"   🌳 Dataset: {dataset_path}")
        
    except Exception as e:
        print(f"❌ Cannot read baseline file: {e}")
        return False
    
    tester = TreeHillRenderOptimizationTester(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        config_path=args.config,
        downsample_factor=getattr(args, 'downsample_factor', 1)
    )
    
    # Verify modification
    try:
        passed, results = tester.verify_modification(
            args.baseline,
            tolerance_rgb=args.tolerance_rgb,
            tolerance_depth=args.tolerance_depth
        )
        
        # Show additional Gaussian processing statistics if requested
        if args.show_gaussian_stats and 'gaussian_processing_stats' in results:
            print("\n" + "="*60)
            print("📊 Detailed Gaussian Processing Statistics")
            print("="*60)
            stats = results['gaussian_processing_stats']
            _print_detailed_gaussian_analysis(stats)
        
        # Save heatmap if requested
        if args.save_heatmap and 'raw_outputs' in results:
            try:
                _save_hits_heatmap(results['raw_outputs'], results['gaussian_processing_stats'], args.save_heatmap)
                print(f"\n📊 Heatmap saved: {args.save_heatmap}")
            except Exception as e:
                print(f"\n⚠️ Failed to save heatmap: {e}")
        
        if passed:
            print("\n🎉 Congratulations! Rendering optimization verification passed!")
            print("✅ All outputs are within tolerance range")
            return True
        else:
            print("\n⚠️ Rendering optimization verification failed!")
            print("❌ Found differences exceeding tolerance")
            print("💡 Suggestions:")
            print("   1. Check rendering optimization modification logic")
            print("   2. Verify floating point precision issues")
            print("   3. Adjust tolerance parameters (if differences are small)")
            return False
            
    except Exception as e:
        print(f"❌ Error during verification process: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Rendering Optimization Test Framework (for k_buffer_size=0 load balancing optimizations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Capture forward baseline (requires trained TreeHill model)
  python test_optimization.py --mode capture \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --dataset data/mipnerf360/treehill \\
    --output baselines/treehill_onestep.pt
  
  # Verify forward optimization (after modifying load balancing code)
  python test_optimization.py --mode verify \\
    --baseline baselines/treehill_onestep.pt
  
  # Capture backward baseline (for testing backward optimizations)
  python test_optimization.py --mode capture-backward \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --dataset data/mipnerf360/treehill \\
    --output baselines/treehill_backward.pt
  
  # Verify backward optimization (after modifying backward optimization code)
  python test_optimization.py --mode verify-backward \\
    --baseline baselines/treehill_backward.pt \\
    --tolerance-grad 1e-6
  
  # Detailed verification 
  python test_optimization.py --mode verify \\
    --baseline baselines/treehill_onestep.pt \\
    --tolerance-rgb 1e-4 --tolerance-depth 1e-4 --verbose
    
  # With Gaussian statistics and heatmap
  python test_optimization.py --mode verify \\
    --baseline baselines/treehill_onestep.pt \\
    --show-gaussian-stats --save-heatmap heatmaps/gaussian_hits.png
    
  # 8x downsampled version (fast testing) - forward and backward
  python test_optimization.py --mode capture \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --output baselines/treehill_onestep_8x.pt \\
    --downsample-factor 8
    
  python test_optimization.py --mode capture-backward \\
    --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt \\
    --output baselines/treehill_backward_8x.pt \\
    --downsample-factor 8
    
  python test_optimization.py --mode verify-backward \\
    --baseline baselines/treehill_backward_8x.pt \\
    --downsample-factor 8 --tolerance-grad 1e-5
    
Notes: 
  - Requires trained TreeHill dataset model
  - Only supports real dataset testing, ensure correct dataset path
  - Specifically for k_buffer_size=0 (unsorted mode) rendering optimizations
        """
    )
    
    # Basic parameters
    parser.add_argument('--mode', 
                       choices=['capture', 'verify', 'capture-backward', 'verify-backward'], 
                       required=True,
                       help='Run mode: capture(capture forward baseline), verify(verify forward optimization), capture-backward(capture backward baseline), verify-backward(verify backward optimization)')
    
    parser.add_argument('--config',
                       type=str,
                       default=None,
                       help='Config file path (default: apps/colmap_3dgut.yaml)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose logging output')
    
    # Capture mode parameters
    capture_group = parser.add_argument_group('Capture mode parameters')
    capture_group.add_argument('--checkpoint', '-c',
                              type=str,
                              help='Trained TreeHill model checkpoint path (required for capture mode)')
    
    capture_group.add_argument('--dataset', '-d',
                              type=str,
                              default='data/mipnerf360/treehill',
                              help='TreeHill dataset path')
    
    capture_group.add_argument('--output', '-o',
                              type=str,
                              default='baselines/treehill_onestep.pt',
                              help='Baseline file output path')
    
    capture_group.add_argument('--view-id',
                              type=int,
                              default=5,
                              help='Test view ID')
    
    # Verification mode parameters
    verify_group = parser.add_argument_group('Verification mode parameters')
    verify_group.add_argument('--baseline', '-b',
                             type=str,
                             default='baselines/treehill_onestep.pt',
                             help='Baseline file path')
    
    verify_group.add_argument('--tolerance-rgb',
                             type=float,
                             default=1e-3,
                             help='RGB difference tolerance')
    
    verify_group.add_argument('--tolerance-depth',
                             type=float,
                             default=1e-3,
                             help='Depth difference tolerance')
    
    verify_group.add_argument('--show-gaussian-stats',
                             action='store_true',
                             help='Show Gaussian processing statistics per pixel')
    
    verify_group.add_argument('--save-heatmap',
                             type=str,
                             help='Save hits_count heatmap to specified path (e.g. heatmaps/hits_visualization.png)')
    
    # Backward optimization test parameters
    backward_group = parser.add_argument_group('Backward optimization test parameters')
    backward_group.add_argument('--tolerance-grad',
                               type=float,
                               default=1e-5,
                               help='Gradient difference tolerance (verify-backward mode)')
    
    # Common parameters - applicable to all modes
    parser.add_argument('--downsample-factor',
                        type=int,
                        default=1,
                        choices=[1, 2, 4, 8],
                        help='Image downsample factor: 1=original resolution(5068×3326), 2=1/2 resolution, 4=1/4 resolution, 8=1/8 resolution')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.mode in ['capture', 'capture-backward'] and not args.checkpoint:
        print(f"❌ {args.mode} mode requires --checkpoint parameter")
        print("💡 Example: --checkpoint runs/treehill_3dgut_onestep/treehill-*/ckpt_last.pt")
        sys.exit(1)
    
    # Setup logging
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
        print("\n❌ User interrupted")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 Execution error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
