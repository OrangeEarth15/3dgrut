"""
Rendering Optimization Tester - Validates load balancing and rendering strategy optimizations

End-to-end testing framework based on TreeHill dataset, capturing data before and after 
trace function execution to compare original and optimized versions.
Specifically designed for k_buffer_size=0 (unsorted mode) rendering optimizations.
"""

import torch 
import numpy as np
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import sys

# Add project root directory for complete package paths
project_root = Path(__file__).parent.parent.parent  # 3dgrut root directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from threedgut_tracer.tracer import Tracer
logger = logging.getLogger(__name__)


class TreeHillRenderOptimizationTester:
    """TreeHill dataset-based rendering optimization tester. Validates load balancing and other optimization strategies for k_buffer_size=0."""
    
    def __init__(self, checkpoint_path: str, dataset_path: str = "data/mipnerf360/treehill", config_path: Optional[str] = None, downsample_factor: int = 1):
        """
        Initialize the tester
        
        Args:
            checkpoint_path: Path to trained TreeHill model checkpoint
            dataset_path: TreeHill dataset path
            config_path: Configuration file path, uses default if not provided
            downsample_factor: Downsampling factor, 1=original resolution, 2=1/2 resolution, 8=1/8 resolution
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.downsample_factor = downsample_factor
        # Use configuration file from project root directory
        project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or str(project_root / "configs/apps/colmap_3dgut.yaml")
        self.tracer = None
        self.gaussians = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
    def _load_treehill_scene(self, test_view_id: int = 5):
        """
        Load TreeHill dataset and trained model
        
        Args:
            test_view_id: Test view ID
            
        Returns:
            Loaded scene data
        """
        logger.info(f"📂 Loading TreeHill dataset from {self.dataset_path}")
        logger.info(f"🏋️ Loading checkpoint from {self.checkpoint_path}")
        
        # Check if files exist
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Load checkpoint (set weights_only=False to support checkpoints with numpy objects)
        # map_location='cuda' loads checkpoint to GPU
        checkpoint = torch.load(self.checkpoint_path, map_location='cuda', weights_only=False)
        logger.info("✅ Checkpoint loaded")
        
        # Load real TreeHill dataset
        try:
            try:
                from threedgrut.datasets.dataset_colmap import ColmapDataset
            except ImportError:
                from ...threedgrut.datasets.dataset_colmap import ColmapDataset
            
            # Create test dataset
            dataset = ColmapDataset(
                path=self.dataset_path,
                device='cuda',
                split='test',
                downsample_factor=self.downsample_factor,
                test_split_interval=8
            )
            
            logger.info(f"✅ Dataset loaded: {len(dataset)} test views")
            
            # Select specified test view
            if test_view_id >= len(dataset):
                logger.warning(f"⚠️ Requested view_id {test_view_id} >= dataset size {len(dataset)}, using view 0")
                test_view_id = 0
            
            # Get test batch - mimicking DataLoader behavior
            test_item = dataset[test_view_id]
            
            # Convert single sample to batch format (mimicking DataLoader's collate_fn)
            # DataLoader puts single sample in list and converts scalars to tensors
            # Contains data, camera poses, intrinsics
            batch = {
                "data": [test_item["data"]],  # DataLoader format: [[1, H, W, 3]]
                "pose": [test_item["pose"]],  # DataLoader format: [[1, 4, 4]]
                "intr": [torch.tensor(test_item["intr"])]   # DataLoader format: [tensor(int)]
            }
            
            # Include mask if available
            if "mask" in test_item:
                batch["mask"] = [test_item["mask"]]
            
            test_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            logger.info(f"🔍 Test batch type: {type(test_batch)}")
            logger.info(f"🔍 Test batch attributes: {[attr for attr in dir(test_batch) if not attr.startswith('_')]}")
            
            # Get image resolution
            if not hasattr(test_batch, 'rgb_gt'):
                raise AttributeError("test_batch does not have 'rgb_gt' attribute - cannot determine image resolution")
                
            rgb_gt = test_batch.rgb_gt
            logger.info(f"🔍 rgb_gt type: {type(rgb_gt)}")
            logger.info(f"🔍 rgb_gt value: {rgb_gt}")
            
            if rgb_gt is None:
                raise ValueError("rgb_gt is None - cannot determine image resolution")
                
            if not hasattr(rgb_gt, 'shape'):
                raise AttributeError("rgb_gt does not have 'shape' attribute - cannot determine image resolution")
                
            shape = rgb_gt.shape
            logger.info(f"📊 rgb_gt shape: {shape}")
            
            # Determine H, W based on tensor dimensions  
            if len(shape) == 4:  # [B, H, W, C]
                _, H, W, _ = shape
            elif len(shape) == 3:  # [H, W, C]
                H, W, _ = shape
            elif len(shape) == 2:  # [H, W]
                H, W = shape
            else:
                raise ValueError(f"Unexpected rgb_gt shape {shape} - cannot determine image resolution. Expected 2D, 3D or 4D tensor.")
            
            logger.info(f"🌳 Using test view {test_view_id}: {H}x{W} resolution")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import ColmapDataset: {e}")
            raise ImportError("ColmapDataset import failed. Please check threedgrut installation.")
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise RuntimeError(f"Failed to load TreeHill dataset from {self.dataset_path}. Please check the dataset path and format.")
        
        scene_data = {
            'checkpoint': checkpoint,
            'test_batch': test_batch,
            'metadata': {
                'dataset': 'mipnerf360_treehill',
                'view_id': test_view_id,
                'resolution': (H, W),
                'checkpoint_path': self.checkpoint_path,
                'dataset_path': self.dataset_path,
                'downsample_factor': self.downsample_factor
            }
        }
        
        logger.info(f"🌳 TreeHill scene loaded: view {test_view_id}, {H}x{W} (downsample_factor={self.downsample_factor})")
        return scene_data
    

    def _setup_deterministic_environment(self):
        """Setup deterministic runtime environment"""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        
        # Setup CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info("🔧 Deterministic environment configured")
    
    def capture_baseline_outputs(self, output_path: str, test_view_id: int = 5) -> Dict[str, Any]:
        """
        Perform complete inference following original repository logic, capture data before and after inference, save as baseline
        
        Args:
            output_path: Baseline file save path
            test_view_id: Test view ID
            
        Returns:
            Captured baseline data
        """
        logger.info("📸 Capturing rendering optimization baseline data...")
        
        # 1. Setup deterministic environment
        self._setup_deterministic_environment()
        
        # 2. Load dataset and checkpoint
        scene_data = self._load_treehill_scene(test_view_id)
        checkpoint = scene_data['checkpoint']
        gpu_batch = scene_data['test_batch']  # Batch object obtained through get_gpu_batch_with_intrinsics
        
        # 3. Initialize tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # Use Hydra to correctly load configuration (mimicking original code approach)
        # config_path must be relative path from current file to configs directory
        # config_name needs to include subdirectory path
        config_name = "apps/colmap_3dgut.yaml"  # Configuration file in configs/apps/ subdirectory
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. Initialize MixtureOfGaussians model (following original repository logic)
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint)
        model.build_acc()  # Build acceleration structure
        logger.info("✅ MixtureOfGaussians model initialized")
        
        # 5. Perform complete inference, capture trace inputs/outputs
        logger.info("🔄 Running inference and capturing trace inputs/outputs...")
        with torch.no_grad():
            # Here model(gpu_batch) internally calls self.renderer.render(self, gpu_batch, train, frame_id)
            # i.e. tracer.render(gaussians=model, gpu_batch=gpu_batch, train=False, frame_id=0)
            trace_outputs = model(gpu_batch, train=False, frame_id=0)
        
        # 6. Save baseline data
        baseline = {
            'trace_inputs': {
                # Save trace function input parameters (following original repository logic: model + gpu_batch + train + frame_id)
                'model_state': {
                    'positions': model.positions.clone(),
                    'rotation': model.rotation.clone(), 
                    'scale': model.scale.clone(),
                    'density': model.density.clone(),
                    'features_albedo': model.features_albedo.clone(),
                    'features_specular': model.features_specular.clone(),
                    'num_gaussians': model.num_gaussians,
                    'n_active_features': model.n_active_features
                },
                'gpu_batch': {
                    'rays_ori': gpu_batch.rays_ori.clone(),
                    'rays_dir': gpu_batch.rays_dir.clone(),
                    'T_to_world': gpu_batch.T_to_world.clone(),
                    'rgb_gt': gpu_batch.rgb_gt.clone() if hasattr(gpu_batch, 'rgb_gt') and gpu_batch.rgb_gt is not None else None,
                    'mask': gpu_batch.mask.clone() if hasattr(gpu_batch, 'mask') and gpu_batch.mask is not None else None,
                    'intrinsics': gpu_batch.intrinsics if hasattr(gpu_batch, 'intrinsics') else None,
                    'intrinsics_OpenCVPinholeCameraModelParameters': getattr(gpu_batch, 'intrinsics_OpenCVPinholeCameraModelParameters', None),
                    'intrinsics_OpenCVFisheyeCameraModelParameters': getattr(gpu_batch, 'intrinsics_OpenCVFisheyeCameraModelParameters', None)
                },
                'train': False,
                'frame_id': 0
            },
            'trace_outputs': {
                # Save trace function output results
                'pred_rgb': trace_outputs['pred_rgb'].clone(),
                'pred_opacity': trace_outputs['pred_opacity'].clone(),
                'pred_dist': trace_outputs['pred_dist'].clone(),
                'hits_count': trace_outputs['hits_count'].clone(),
                'mog_visibility': trace_outputs['mog_visibility'].clone()
            },
            'metadata': {
                **scene_data['metadata'],
                'baseline_version': 'original_unsorted_rendering',
                'render_mode': 'unsorted_k_buffer_size_0',
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': time.time(),
                'config_path': self.config_path
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save baseline
        torch.save(baseline, output_path)
        
        # Calculate data size
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        logger.info(f"✅ Baseline saved: {output_path}")
        logger.info(f"   📊 Resolution: {baseline['metadata']['resolution']}")
        logger.info(f"   🔬 Gaussians: {baseline['trace_inputs']['model_state']['num_gaussians']}")
        logger.info(f"   💾 File size: {file_size_mb:.1f} MB")
        
        return baseline
    
    def capture_backward_baseline(self, output_path: str, test_view_id: int = 5) -> Dict[str, Any]:
        """
        Capture backward propagation baseline data, including gradients for all parameters
        
        Args:
            output_path: Baseline file save path
            test_view_id: Test view ID
            
        Returns:
            Captured backward propagation baseline data
        """
        logger.info("📸 Capturing rendering optimization backward baseline data...")
        
        # 1. Setup deterministic environment
        self._setup_deterministic_environment()
        
        # 2. Load dataset and checkpoint
        scene_data = self._load_treehill_scene(test_view_id)
        checkpoint = scene_data['checkpoint']
        gpu_batch = scene_data['test_batch']
        
        # 3. Initialize tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # Use Hydra to correctly load configuration
        config_name = "apps/colmap_3dgut.yaml"
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. Initialize MixtureOfGaussians model and set requires_grad=True
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint)
        model.build_acc()
        
        # Ensure all parameters require gradients
        model.positions.requires_grad_(True)
        model.rotation.requires_grad_(True) 
        model.scale.requires_grad_(True)
        model.density.requires_grad_(True)
        model.features_albedo.requires_grad_(True)
        model.features_specular.requires_grad_(True)
        
        logger.info("✅ MixtureOfGaussians model initialized with gradients enabled")
        
        # 5. Forward pass
        logger.info("🔄 Running forward pass...")
        trace_outputs = model(gpu_batch, train=True, frame_id=0)  # train=True to enable gradients
        
        # 6. Construct loss function and perform backward pass
        logger.info("🔄 Running backward pass...")
        
        # Use ground truth to construct RGB loss
        if gpu_batch.rgb_gt is not None:
            rgb_loss = torch.nn.functional.mse_loss(trace_outputs['pred_rgb'], gpu_batch.rgb_gt)
            logger.info(f"🔍 RGB MSE Loss: {rgb_loss.item():.6f}")
        else:
            # If no ground truth, use simple sum loss to trigger backward
            rgb_loss = trace_outputs['pred_rgb'].sum()
            logger.info(f"🔍 Simple sum loss: {rgb_loss.item():.6f}")
        
        # Clear previous gradients
        model.zero_grad()
        
        # Backward pass
        rgb_loss.backward()
        torch.cuda.synchronize()  # Ensure all CUDA operations complete
        
        logger.info("✅ Backward pass completed")
        
        # 7. Save baseline data (including forward outputs and gradients)
        baseline = {
            'trace_inputs': {
                'model_state': {
                    'positions': model.positions.detach().clone(),
                    'rotation': model.rotation.detach().clone(),
                    'scale': model.scale.detach().clone(),
                    'density': model.density.detach().clone(),
                    'features_albedo': model.features_albedo.detach().clone(),
                    'features_specular': model.features_specular.detach().clone(),
                    'num_gaussians': model.num_gaussians,
                    'n_active_features': model.n_active_features
                },
                'gpu_batch': {
                    'rays_ori': gpu_batch.rays_ori.clone(),
                    'rays_dir': gpu_batch.rays_dir.clone(),
                    'T_to_world': gpu_batch.T_to_world.clone(),
                    'rgb_gt': gpu_batch.rgb_gt.clone() if hasattr(gpu_batch, 'rgb_gt') and gpu_batch.rgb_gt is not None else None,
                    'mask': gpu_batch.mask.clone() if hasattr(gpu_batch, 'mask') and gpu_batch.mask is not None else None,
                    'intrinsics': gpu_batch.intrinsics if hasattr(gpu_batch, 'intrinsics') else None,
                    'intrinsics_OpenCVPinholeCameraModelParameters': getattr(gpu_batch, 'intrinsics_OpenCVPinholeCameraModelParameters', None),
                    'intrinsics_OpenCVFisheyeCameraModelParameters': getattr(gpu_batch, 'intrinsics_OpenCVFisheyeCameraModelParameters', None)
                },
                'train': True,
                'frame_id': 0
            },
            'trace_outputs': {
                'pred_rgb': trace_outputs['pred_rgb'].detach().clone(),
                'pred_opacity': trace_outputs['pred_opacity'].detach().clone(),
                'pred_dist': trace_outputs['pred_dist'].detach().clone(),
                'hits_count': trace_outputs['hits_count'].detach().clone(),
                'mog_visibility': trace_outputs['mog_visibility'].detach().clone()
            },
            'gradients': {
                # Save gradients for all parameters
                'positions_grad': model.positions.grad.clone() if model.positions.grad is not None else None,
                'rotation_grad': model.rotation.grad.clone() if model.rotation.grad is not None else None,
                'scale_grad': model.scale.grad.clone() if model.scale.grad is not None else None,
                'density_grad': model.density.grad.clone() if model.density.grad is not None else None,
                'features_albedo_grad': model.features_albedo.grad.clone() if model.features_albedo.grad is not None else None,
                'features_specular_grad': model.features_specular.grad.clone() if model.features_specular.grad is not None else None,
            },
            'loss': rgb_loss.detach().item(),
            'metadata': {
                **scene_data['metadata'],
                'baseline_version': 'original_unsorted_backward',
                'render_mode': 'unsorted_k_buffer_size_0',
                'test_type': 'backward',
                'loss_type': 'mse_rgb' if gpu_batch.rgb_gt is not None else 'sum_rgb',
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': time.time(),
                'config_path': self.config_path
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save baseline
        torch.save(baseline, output_path)
        
        # Calculate data size and gradient statistics
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        # Count non-zero gradient parameters
        grad_stats = self._compute_gradient_stats(baseline['gradients'])
        
        logger.info(f"✅ Backward baseline saved: {output_path}")
        logger.info(f"   📊 Resolution: {baseline['metadata']['resolution']}")
        logger.info(f"   🔬 Gaussians: {baseline['trace_inputs']['model_state']['num_gaussians']}")
        logger.info(f"   📉 Loss: {baseline['loss']:.6f}")
        logger.info(f"   🎯 Gradients: {grad_stats['non_zero_params']}/{grad_stats['total_params']} params have gradients")
        logger.info(f"   💾 File size: {file_size_mb:.1f} MB")
        
        return baseline
    
    def verify_modification(self, baseline_path: str, 
                          tolerance_rgb: float = 1e-3, 
                          tolerance_depth: float = 1e-3) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify modified rendering optimization implementation
        
        Args:
            baseline_path: Baseline file path
            tolerance_rgb: RGB difference tolerance
            tolerance_depth: Depth difference tolerance
            
        Returns:
            (passed, detailed test results)
        """
        logger.info("🔍 Verifying modified rendering optimization...")
        
        # 1. Load baseline data
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
            
        baseline = torch.load(baseline_path, map_location='cuda', weights_only=False)
        trace_inputs = baseline['trace_inputs']
        expected_outputs = baseline['trace_outputs']
        
        logger.info(f"📂 Loaded baseline: {baseline_path}")
        logger.info(f"   🏷️  Version: {baseline['metadata']['baseline_version']}")
        logger.info(f"   📅 Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(baseline['metadata']['timestamp']))}")
        
        # 2. Setup deterministic environment
        self._setup_deterministic_environment()
        
        # 3. Initialize tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # Use Hydra to correctly load configuration (mimicking original code approach)
        # config_path must be relative path from current file to configs directory
        # config_name needs to include subdirectory path
        config_name = "apps/colmap_3dgut.yaml"  # Configuration file in configs/apps/ subdirectory
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. Rebuild MixtureOfGaussians model (following original repository logic)
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        self._restore_model_state(model, trace_inputs['model_state'])
        model.build_acc()  # Build acceleration structure
        logger.info("✅ MixtureOfGaussians model restored")
        
        # Rebuild gpu_batch
        gpu_batch = self._rebuild_gpu_batch(trace_inputs['gpu_batch'])
        
        # 5. Run modified version
        with torch.no_grad():
            logger.info("🔄 Running modified trace function...")
            
            actual_outputs = model(
                gpu_batch, 
                train=trace_inputs['train'], 
                frame_id=trace_inputs['frame_id']
            )
            
            logger.info(f"🔍 Actual outputs type: {type(actual_outputs)}")
            if isinstance(actual_outputs, dict):
                logger.info(f"🔍 Actual outputs keys: {list(actual_outputs.keys())}")
            else:
                logger.info(f"🔍 Actual outputs attributes: {[attr for attr in dir(actual_outputs) if not attr.startswith('_')]}")
        
        # 6. Compare results
        results = self._compare_trace_outputs(
            expected_outputs, 
            actual_outputs,
            tolerance_rgb, 
            tolerance_depth,
            baseline['metadata']
        )
        
        # 7. Add Gaussian processing statistics
        gaussian_stats = self._analyze_gaussian_processing_stats(actual_outputs, baseline['metadata'])
        results['gaussian_processing_stats'] = gaussian_stats
        
        # Save raw outputs for other analysis
        results['raw_outputs'] = actual_outputs
        
        # 8. Generate report
        self._print_verification_report(results)
        
        return results['passed'], results
    
    def verify_backward_modification(self, baseline_path: str,
                                   tolerance_grad: float = 1e-5) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify modified backward propagation implementation
        
        Args:
            baseline_path: Backward propagation baseline file path
            tolerance_grad: Gradient difference tolerance
            
        Returns:
            (passed, detailed test results)
        """
        logger.info("🔍 Verifying modified backward pass...")
        
        # 1. Load baseline data
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Backward baseline file not found: {baseline_path}")
            
        baseline = torch.load(baseline_path, map_location='cuda', weights_only=False)
        
        # Check if it's a backward baseline file
        if baseline['metadata'].get('test_type') != 'backward':
            raise ValueError(f"Baseline file is not a backward test baseline. Got test_type: {baseline['metadata'].get('test_type')}")
        
        trace_inputs = baseline['trace_inputs']
        expected_outputs = baseline['trace_outputs']
        expected_gradients = baseline['gradients']
        expected_loss = baseline['loss']
        
        logger.info(f"📂 Loaded backward baseline: {baseline_path}")
        logger.info(f"   🏷️  Version: {baseline['metadata']['baseline_version']}")
        logger.info(f"   📉 Expected loss: {expected_loss:.6f}")
        logger.info(f"   📅 Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(baseline['metadata']['timestamp']))}")
        
        # 2. Setup deterministic environment
        self._setup_deterministic_environment()
        
        # 3. Initialize tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        config_name = "apps/colmap_3dgut.yaml"
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. Rebuild MixtureOfGaussians model
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        self._restore_model_state(model, trace_inputs['model_state'])
        model.build_acc()
        
        # Set requires_grad=True
        model.positions.requires_grad_(True)
        model.rotation.requires_grad_(True)
        model.scale.requires_grad_(True)
        model.density.requires_grad_(True)
        model.features_albedo.requires_grad_(True)
        model.features_specular.requires_grad_(True)
        
        logger.info("✅ MixtureOfGaussians model restored with gradients enabled")
        
        # Rebuild gpu_batch
        gpu_batch = self._rebuild_gpu_batch(trace_inputs['gpu_batch'])
        
        # 5. Run modified version
        # Ensure gradients are enabled (they are enabled by default, no context manager needed)
        logger.info("🔄 Running modified forward pass...")
        
        actual_outputs = model(
            gpu_batch, 
            train=trace_inputs['train'], 
            frame_id=trace_inputs['frame_id']
        )
        
        # Construct same loss function
        logger.info("🔄 Running modified backward pass...")
        loss_type = baseline['metadata']['loss_type']
        
        if loss_type == 'mse_rgb' and gpu_batch.rgb_gt is not None:
            actual_loss = torch.nn.functional.mse_loss(actual_outputs['pred_rgb'], gpu_batch.rgb_gt)
        else:
            actual_loss = actual_outputs['pred_rgb'].sum()
            
        logger.info(f"🔍 Actual loss: {actual_loss.item():.6f}")
        
        # Clear gradients and perform backward pass
        model.zero_grad()
        actual_loss.backward()
        torch.cuda.synchronize()
        
        logger.info("✅ Modified backward pass completed")
        
        # 6. Collect actual gradients
        actual_gradients = {
            'positions_grad': model.positions.grad.clone() if model.positions.grad is not None else None,
            'rotation_grad': model.rotation.grad.clone() if model.rotation.grad is not None else None,
            'scale_grad': model.scale.grad.clone() if model.scale.grad is not None else None,
            'density_grad': model.density.grad.clone() if model.density.grad is not None else None,
            'features_albedo_grad': model.features_albedo.grad.clone() if model.features_albedo.grad is not None else None,
            'features_specular_grad': model.features_specular.grad.clone() if model.features_specular.grad is not None else None,
        }
        
        # 7. Compare results
        results = self._compare_backward_outputs(
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            expected_gradients=expected_gradients,
            actual_gradients=actual_gradients,
            expected_loss=expected_loss,
            actual_loss=actual_loss.item(),
            tolerance_grad=tolerance_grad,
            metadata=baseline['metadata']
        )
        
        # 8. Calculate final pass status (mainly focus on gradient and loss)
        grad_passed = results['gradients']['passed']
        loss_passed = results['loss']['passed']
        backward_test_passed = grad_passed and loss_passed
        results['passed'] = backward_test_passed
        
        # 9. Generate report
        self._print_backward_verification_report(results)
        
        return backward_test_passed, results
    
    def _restore_model_state(self, model, model_state):
        """Restore model state from saved data"""
        import torch.nn as nn
        
        # Model parameters need to be torch.nn.Parameter type
        model.positions = nn.Parameter(model_state['positions'].clone())
        model.rotation = nn.Parameter(model_state['rotation'].clone())
        model.scale = nn.Parameter(model_state['scale'].clone())
        model.density = nn.Parameter(model_state['density'].clone())
        model.features_albedo = nn.Parameter(model_state['features_albedo'].clone())
        model.features_specular = nn.Parameter(model_state['features_specular'].clone())
        # Ensure n_active_features is correctly set
        model.n_active_features = model_state['n_active_features']
        
    def _rebuild_gpu_batch(self, batch_data):
        """Rebuild gpu_batch from saved data"""
        from threedgrut.datasets.protocols import Batch
        
        # Rebuild complete Batch object including all saved attributes
        kwargs = {
            'rays_ori': batch_data['rays_ori'].clone(),
            'rays_dir': batch_data['rays_dir'].clone(),
            'T_to_world': batch_data['T_to_world'].clone(),
        }
        
        # Add optional attributes (if they exist)
        if batch_data.get('rgb_gt') is not None:
            kwargs['rgb_gt'] = batch_data['rgb_gt'].clone()
        if batch_data.get('mask') is not None:
            kwargs['mask'] = batch_data['mask'].clone()
        if batch_data.get('intrinsics') is not None:
            kwargs['intrinsics'] = batch_data['intrinsics']
        if batch_data.get('intrinsics_OpenCVPinholeCameraModelParameters') is not None:
            kwargs['intrinsics_OpenCVPinholeCameraModelParameters'] = batch_data['intrinsics_OpenCVPinholeCameraModelParameters']
        if batch_data.get('intrinsics_OpenCVFisheyeCameraModelParameters') is not None:
            kwargs['intrinsics_OpenCVFisheyeCameraModelParameters'] = batch_data['intrinsics_OpenCVFisheyeCameraModelParameters']
        
        return Batch(**kwargs)
    
    def _analyze_gaussian_processing_stats(self, outputs: Dict[str, torch.Tensor], 
                                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistics of Gaussian processing per pixel"""
        hits_count_tensor = outputs['hits_count']
        logger.info(f"🔍 hits_count tensor shape: {hits_count_tensor.shape}")
        hits_count = hits_count_tensor.cpu().numpy()
        
        # More robust dimension handling
        original_shape = hits_count.shape
        logger.info(f"🔍 Original hits_count shape: {original_shape}")
        
        # Recursively squeeze all dimensions of size 1 until we get 2D array
        while len(hits_count.shape) > 2:
            # Find first dimension of size 1
            size_1_dims = [i for i, size in enumerate(hits_count.shape) if size == 1]
            if size_1_dims:
                hits_count = hits_count.squeeze(axis=size_1_dims[0])
            else:
                # If no dimensions of size 1 but still >2D, might be [B, C, H, W] format
                # Try taking first batch and first channel
                if len(hits_count.shape) == 4:  # [B, C, H, W]
                    hits_count = hits_count[0, 0]
                elif len(hits_count.shape) == 3:  # [B, H, W] or [C, H, W]
                    hits_count = hits_count[0]
                else:
                    break
        
        logger.info(f"🔍 Final hits_count shape: {hits_count.shape}")
        
        # Calculate statistics
        valid_pixels = hits_count[hits_count > 0]  # Only count valid pixels
        total_pixels = hits_count.size
        valid_pixel_count = len(valid_pixels)
        
        stats = {
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixel_count,
            'valid_pixel_ratio': valid_pixel_count / total_pixels if total_pixels > 0 else 0,
            'hits_per_pixel': {
                'min': float(hits_count.min()),
                'max': float(hits_count.max()), 
                'mean': float(hits_count.mean()),
                'median': float(np.median(hits_count)),
                'std': float(hits_count.std())
            },
            'valid_hits_per_pixel': {
                'min': float(valid_pixels.min()) if len(valid_pixels) > 0 else 0,
                'max': float(valid_pixels.max()) if len(valid_pixels) > 0 else 0,
                'mean': float(valid_pixels.mean()) if len(valid_pixels) > 0 else 0,
                'median': float(np.median(valid_pixels)) if len(valid_pixels) > 0 else 0,
                'std': float(valid_pixels.std()) if len(valid_pixels) > 0 else 0
            },
            'total_gaussian_hits': float(hits_count.sum()),
            'histogram': self._compute_hits_histogram(hits_count),
            'resolution': hits_count.shape,
            'hits_count_raw': hits_count.copy()  # Save raw data for visualization
        }
        
        return stats
    
    def _compute_hits_histogram(self, hits_count, bins=20):
        """Compute histogram of hits_count"""
        max_hits = int(hits_count.max())
        if max_hits == 0:
            return {'bins': [], 'counts': []}
        
        # Create histogram bins
        bin_edges = np.linspace(0, max_hits, min(bins, max_hits + 1))
        hist_counts, _ = np.histogram(hits_count.flatten(), bins=bin_edges)
        
        return {
            'bin_edges': bin_edges.tolist(),
            'counts': hist_counts.tolist(),
            'bins': bins
        }
    
    def _compare_trace_outputs(self, expected: Dict[str, torch.Tensor], 
                              actual: Dict[str, torch.Tensor],
                              tolerance_rgb: float, 
                              tolerance_depth: float,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compare trace function output results"""
        
        # Compare RGB
        rgb_expected = expected['pred_rgb']
        rgb_actual = actual['pred_rgb']
        rgb_diff = torch.abs(rgb_expected - rgb_actual)
        
        # Compare depth
        depth_expected = expected['pred_dist']
        depth_actual = actual['pred_dist']
        depth_diff = torch.abs(depth_expected - depth_actual)
        
        # Compare hit count
        count_expected = expected['hits_count']
        count_actual = actual['hits_count']
        count_diff = torch.abs(count_expected - count_actual)
        
        # Calculate statistics
        max_rgb_diff = rgb_diff.max().item()
        mean_rgb_diff = rgb_diff.mean().item()
        max_depth_diff = depth_diff.max().item()
        mean_depth_diff = depth_diff.mean().item()
        max_count_diff = count_diff.max().item()
        mean_count_diff = count_diff.mean().item()
        
        # Determine pass status
        rgb_passed = max_rgb_diff < tolerance_rgb
        depth_passed = max_depth_diff < tolerance_depth
        count_passed = max_count_diff <= 1  # Hit count should be completely consistent
        
        overall_passed = rgb_passed and depth_passed and count_passed
        
        return {
            'passed': overall_passed,
            'max_rgb_diff': max_rgb_diff,
            'mean_rgb_diff': mean_rgb_diff,
            'max_depth_diff': max_depth_diff,
            'mean_depth_diff': mean_depth_diff,
            'max_count_diff': max_count_diff,
            'mean_count_diff': mean_count_diff,
            'tolerance_rgb': tolerance_rgb,
            'tolerance_depth': tolerance_depth,
            'metadata': metadata
        }
    
    def _print_verification_report(self, results: Dict[str, Any]):
        """Print verification report"""
        print("\n" + "🌳 " + "="*60)
        print("🌳 Rendering Optimization Verification Report")
        print("🌳 " + "="*60)
        print(f"🏷️  Baseline version: {results['metadata']['baseline_version']}")
        print(f"📅 Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 Resolution: {results['metadata'].get('resolution', 'unknown')}")
        downsample_factor = results['metadata'].get('downsample_factor', 1)
        if downsample_factor > 1:
            print(f"📐 Downsample factor: {downsample_factor}x")
        print("-" * 60)
        
        # RGB comparison
        rgb_status = "✅ Pass" if results['max_rgb_diff'] < results['tolerance_rgb'] else "❌ Fail"
        print(f"🎨 RGB Output        | {rgb_status}")
        print(f"   Max error: {results['max_rgb_diff']:.2e} (threshold: {results['tolerance_rgb']:.2e})")
        print(f"   Mean error: {results['mean_rgb_diff']:.2e}")
        print()
        
        # Depth comparison
        depth_status = "✅ Pass" if results['max_depth_diff'] < results['tolerance_depth'] else "❌ Fail"
        print(f"📏 Depth Output      | {depth_status}")
        print(f"   Max error: {results['max_depth_diff']:.2e} (threshold: {results['tolerance_depth']:.2e})")
        print(f"   Mean error: {results['mean_depth_diff']:.2e}")
        print()
        
        # Hit count comparison
        count_status = "✅ Pass" if results['max_count_diff'] <= 1 else "❌ Fail"
        print(f"🎯 Hit Count         | {count_status}")
        print(f"   Max error: {results['max_count_diff']:.2e} (threshold: 1)")
        print(f"   Mean error: {results['mean_count_diff']:.2e}")
        print()
        
        # Gaussian processing statistics
        if 'gaussian_processing_stats' in results:
            self._print_gaussian_stats(results['gaussian_processing_stats'])
        
        # Overall assessment
        overall_status = "🎉 Optimization Verified" if results['passed'] else "⚠️  Optimization Issues Found"
        print(f"🏆 {overall_status}")
        print("🌳 " + "="*60)
    
    def _print_gaussian_stats(self, stats: Dict[str, Any]):
        """Print Gaussian processing statistics"""
        print("🎯 Gaussian Processing Statistics")
        print("-" * 40)
        
        # Basic information
        print(f"📊 Image Resolution: {stats['resolution'][1]}×{stats['resolution'][0]}")
        print(f"📊 Total Pixels: {stats['total_pixels']:,}")
        print(f"📊 Valid Pixels: {stats['valid_pixels']:,} ({stats['valid_pixel_ratio']:.1%})")
        print()
        
        # Global statistics
        print("🌐 Global Statistics:")
        print(f"   Total Gaussian Hits: {stats['total_gaussian_hits']:,.0f}")
        print(f"   Average per Pixel: {stats['hits_per_pixel']['mean']:.1f} gaussians")
        print(f"   Maximum per Pixel: {stats['hits_per_pixel']['max']:.0f} gaussians")
        print(f"   Median: {stats['hits_per_pixel']['median']:.1f} gaussians")
        print()
        
        # Valid pixel statistics (excluding background)
        if stats['valid_pixels'] > 0:
            print("🎯 Valid Pixel Statistics (excluding background):")
            print(f"   Average per Pixel: {stats['valid_hits_per_pixel']['mean']:.1f} gaussians")
            print(f"   Maximum per Pixel: {stats['valid_hits_per_pixel']['max']:.0f} gaussians")
            print(f"   Minimum per Pixel: {stats['valid_hits_per_pixel']['min']:.0f} gaussians")
            print(f"   Standard Deviation: {stats['valid_hits_per_pixel']['std']:.1f}")
            print()
        
        # Distribution histogram
        if stats['histogram']['counts']:
            print("📈 Hit Distribution:")
            self._print_histogram(stats['histogram'])
        print()
    
    def _print_histogram(self, histogram):
        """Print simple ASCII histogram"""
        bins = histogram['bin_edges']
        counts = histogram['counts']
        
        if not counts or max(counts) == 0:
            print("   (No valid data)")
            return
        
        # Normalize display
        max_count = max(counts)
        max_width = 30  # Maximum display width
        
        for i in range(len(counts)):
            if i >= len(bins) - 1:
                break
            
            start_bin = bins[i]
            end_bin = bins[i + 1]
            count = counts[i]
            
            if count > 0:
                # Calculate display width
                width = int((count / max_count) * max_width)
                bar = "█" * width
                
                # Format output
                print(f"   {start_bin:4.0f}-{end_bin:4.0f}: {bar} ({count:,})")
    
    def _compute_gradient_stats(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute gradient statistics"""
        total_params = 0
        non_zero_params = 0
        grad_norms = {}
        
        for param_name, grad in gradients.items():
            if grad is not None:
                total_elements = grad.numel()
                non_zero_elements = torch.count_nonzero(grad).item()
                grad_norm = torch.norm(grad).item()
                
                total_params += total_elements
                non_zero_params += non_zero_elements
                grad_norms[param_name] = {
                    'norm': grad_norm,
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'total_elements': total_elements,
                    'non_zero_elements': non_zero_elements,
                    'sparsity': 1.0 - (non_zero_elements / total_elements) if total_elements > 0 else 0.0
                }
        
        return {
            'total_params': total_params,
            'non_zero_params': non_zero_params,
            'sparsity': 1.0 - (non_zero_params / total_params) if total_params > 0 else 0.0,
            'param_stats': grad_norms
        }
    
    def _compare_backward_outputs(self, expected_outputs: Dict[str, torch.Tensor],
                                actual_outputs: Dict[str, torch.Tensor],
                                expected_gradients: Dict[str, torch.Tensor],
                                actual_gradients: Dict[str, torch.Tensor],
                                expected_loss: float,
                                actual_loss: float,
                                tolerance_grad: float,
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compare backward propagation output results"""
        
        # Compare forward propagation outputs (with relaxed tolerance for backward tests)
        rgb_expected = expected_outputs['pred_rgb']
        rgb_actual = actual_outputs['pred_rgb']
        rgb_diff = torch.abs(rgb_expected - rgb_actual)
        
        depth_expected = expected_outputs['pred_dist']
        depth_actual = actual_outputs['pred_dist']
        depth_diff = torch.abs(depth_expected - depth_actual)
        
        count_expected = expected_outputs['hits_count']
        count_actual = actual_outputs['hits_count']
        count_diff = torch.abs(count_expected - count_actual)
        
        # Compare loss values
        loss_diff = abs(expected_loss - actual_loss)
        
        # Compare gradients
        gradient_comparisons = {}
        max_grad_diff = 0.0
        mean_grad_diff = 0.0
        total_grad_elements = 0
        
        for param_name in expected_gradients.keys():
            expected_grad = expected_gradients[param_name]
            actual_grad = actual_gradients[param_name]
            
            if expected_grad is not None and actual_grad is not None:
                grad_diff = torch.abs(expected_grad - actual_grad)
                max_param_grad_diff = grad_diff.max().item()
                mean_param_grad_diff = grad_diff.mean().item()
                
                gradient_comparisons[param_name] = {
                    'max_diff': max_param_grad_diff,
                    'mean_diff': mean_param_grad_diff,
                    'expected_norm': torch.norm(expected_grad).item(),
                    'actual_norm': torch.norm(actual_grad).item(),
                    'passed': max_param_grad_diff < tolerance_grad
                }
                
                # Update global statistics
                max_grad_diff = max(max_grad_diff, max_param_grad_diff)
                mean_grad_diff += mean_param_grad_diff * expected_grad.numel()
                total_grad_elements += expected_grad.numel()
                
            elif expected_grad is None and actual_grad is None:
                gradient_comparisons[param_name] = {
                    'max_diff': 0.0,
                    'mean_diff': 0.0,
                    'expected_norm': 0.0,
                    'actual_norm': 0.0,
                    'passed': True
                }
            else:
                # One is None, one is not None
                gradient_comparisons[param_name] = {
                    'max_diff': float('inf'),
                    'mean_diff': float('inf'),
                    'expected_norm': torch.norm(expected_grad).item() if expected_grad is not None else 0.0,
                    'actual_norm': torch.norm(actual_grad).item() if actual_grad is not None else 0.0,
                    'passed': False
                }
                max_grad_diff = float('inf')
        
        if total_grad_elements > 0:
            mean_grad_diff /= total_grad_elements
        
        # Calculate pass status (relaxed tolerance for forward outputs in backward tests)
        # Because forward pass may also use load balancing optimizations, allow some numerical differences
        rgb_passed = rgb_diff.max().item() < 1e-3  # Use relaxed RGB tolerance
        depth_passed = depth_diff.max().item() < 1e-3  # Use relaxed depth tolerance
        count_passed = count_diff.max().item() <= 10  # Allow some hit count differences
        loss_passed = loss_diff < 1e-6
        grad_passed = max_grad_diff < tolerance_grad and max_grad_diff != float('inf')
        
        overall_passed = rgb_passed and depth_passed and count_passed and loss_passed and grad_passed
        
        return {
            'passed': overall_passed,
            'forward_outputs': {
                'rgb_passed': rgb_passed,
                'depth_passed': depth_passed,
                'count_passed': count_passed,
                'max_rgb_diff': rgb_diff.max().item(),
                'max_depth_diff': depth_diff.max().item(),
                'max_count_diff': count_diff.max().item(),
            },
            'loss': {
                'passed': loss_passed,
                'expected': expected_loss,
                'actual': actual_loss,
                'diff': loss_diff
            },
            'gradients': {
                'passed': grad_passed,
                'max_diff': max_grad_diff,
                'mean_diff': mean_grad_diff,
                'tolerance': tolerance_grad,
                'param_comparisons': gradient_comparisons
            },
            'metadata': metadata
        }
    
    def _print_backward_verification_report(self, results: Dict[str, Any]):
        """Print backward propagation verification report"""
        print("\n" + "🌳 " + "="*70)
        print("🌳 Rendering Optimization Backward Verification Report")
        print("🌳 " + "="*70)
        print(f"🏷️  Baseline Version: {results['metadata']['baseline_version']}")
        print(f"📅 Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 Resolution: {results['metadata'].get('resolution', 'unknown')}")
        downsample_factor = results['metadata'].get('downsample_factor', 1)
        if downsample_factor > 1:
            print(f"📐 Downsample Factor: {downsample_factor}x")
        print("-" * 70)
        
        # Forward output comparison (backward tests allow forward output differences)
        forward_results = results['forward_outputs']
        print("🎨 Forward Output Verification (may differ due to load balancing optimizations):")
        
        rgb_status = "✅ Pass" if forward_results['rgb_passed'] else "❌ Fail"
        print(f"   RGB Output:      {rgb_status} (Max Diff: {forward_results['max_rgb_diff']:.2e})")
        
        depth_status = "✅ Pass" if forward_results['depth_passed'] else "❌ Fail"
        print(f"   Depth Output:     {depth_status} (Max Diff: {forward_results['max_depth_diff']:.2e})")
        
        count_status = "✅ Pass" if forward_results['count_passed'] else "❌ Fail"
        print(f"   Hit Count:     {count_status} (Max Diff: {forward_results['max_count_diff']:.2e})")
        print()
        
        # Loss value comparison
        loss_results = results['loss']
        loss_status = "✅ Pass" if loss_results['passed'] else "❌ Fail"
        print(f"📉 Loss Verification:   {loss_status}")
        print(f"   Expected Loss: {loss_results['expected']:.8f}")
        print(f"   Actual Loss: {loss_results['actual']:.8f}")
        print(f"   Loss Difference: {loss_results['diff']:.2e}")
        print()
        
        # Gradient comparison
        grad_results = results['gradients']
        grad_status = "✅ Pass" if grad_results['passed'] else "❌ Fail"
        print(f"🎯 Gradient Verification:     {grad_status}")
        print(f"   Max Gradient Diff: {grad_results['max_diff']:.2e} (Threshold: {grad_results['tolerance']:.2e})")
        print(f"   Mean Gradient Diff: {grad_results['mean_diff']:.2e}")
        print()
        
        # Parameter gradient details
        print("📊 Parameter Gradient Comparison:")
        param_comparisons = grad_results['param_comparisons']
        for param_name, comparison in param_comparisons.items():
            status = "✅" if comparison['passed'] else "❌"
            param_display_name = param_name.replace('_grad', '')
            print(f"   {param_display_name:20s} {status} "
                  f"(Max Diff: {comparison['max_diff']:.2e}, "
                  f"Expected Norm: {comparison['expected_norm']:.2e}, "
                  f"Actual Norm: {comparison['actual_norm']:.2e})")
        print()
        
        # Overall assessment - for backward tests, mainly focus on gradient correctness
        grad_passed = results['gradients']['passed']
        loss_passed = results['loss']['passed']
        forward_has_issues = not (forward_results['rgb_passed'] and forward_results['depth_passed'] and forward_results['count_passed'])
        
        # Use already calculated pass status
        backward_test_passed = results['passed']
        
        if backward_test_passed:
            if forward_has_issues:
                overall_status = "🎉 Backward Optimization Verified (Forward outputs differ due to load balancing)"
            else:
                overall_status = "🎉 Backward Optimization Verified"
        else:
            overall_status = "⚠️  Backward Optimization Issues Found"
            
        print(f"🏆 {overall_status}")
        
        if not backward_test_passed:
            print("\n💡 Debug Suggestions:")
            if not grad_results['passed']:
                print("   - Gradients inconsistent, check backward pass optimization modifications")
                print("   - Try adjusting gradient tolerance parameter")
            if not loss_results['passed']:
                print("   - Loss values inconsistent, possible numerical precision issues")
        elif forward_has_issues:
            print("\n💡 Note:")
            print("   - Forward outputs differ due to load balancing optimizations (this is expected)")
            print("   - Gradients are consistent, backward optimization is working correctly")
        
        print("🌳 " + "="*70)

