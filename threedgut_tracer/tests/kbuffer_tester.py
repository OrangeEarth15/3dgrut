"""
K-Buffer测试器 - 用于验证evalKBuffer修改的正确性

基于TreeHill数据集的端到端测试方案，在trace函数前后捕获数据，对比原始版本和修改版本的输出。
"""

import torch 
import numpy as np
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import sys

# 添加项目根目录，使用完整包路径
project_root = Path(__file__).parent.parent.parent  # 3dgrut根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from threedgut_tracer.tracer import Tracer
logger = logging.getLogger(__name__)


class TreeHillKBufferTester:
    """基于TreeHill数据集的K-Buffer测试器"""
    
    def __init__(self, checkpoint_path: str, dataset_path: str = "data/mipnerf360/treehill", config_path: Optional[str] = None, downsample_factor: int = 1):
        """
        初始化测试器
        
        Args:
            checkpoint_path: 训练好的treehill模型checkpoint路径
            dataset_path: treehill数据集路径
            config_path: 配置文件路径，如果不提供则使用默认配置
            downsample_factor: 下采样因子，1=原分辨率，2=1/2分辨率，8=1/8分辨率
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.downsample_factor = downsample_factor
        # 使用项目根目录下的配置文件
        project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or str(project_root / "configs/apps/colmap_3dgut.yaml")
        self.tracer = None
        self.gaussians = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
    def _load_treehill_scene(self, test_view_id: int = 5):
        """
        加载TreeHill数据集和训练好的模型
        
        Args:
            test_view_id: 测试视角ID
            
        Returns:
            加载的场景数据
        """
        logger.info(f"📂 Loading TreeHill dataset from {self.dataset_path}")
        logger.info(f"🏋️ Loading checkpoint from {self.checkpoint_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # 加载checkpoint (设置weights_only=False以支持包含numpy对象的checkpoint)
        # map_location='cuda' 将checkpoint加载到GPU上
        checkpoint = torch.load(self.checkpoint_path, map_location='cuda', weights_only=False)
        logger.info("✅ Checkpoint loaded")
        
        # 加载真实的TreeHill数据集
        try:
            try:
                from threedgrut.datasets.dataset_colmap import ColmapDataset
            except ImportError:
                from ...threedgrut.datasets.dataset_colmap import ColmapDataset
            
            # 创建测试数据集
            dataset = ColmapDataset(
                path=self.dataset_path,
                device='cuda',
                split='test',
                downsample_factor=self.downsample_factor,
                test_split_interval=8
            )
            
            logger.info(f"✅ Dataset loaded: {len(dataset)} test views")
            
            # 选择指定的测试视角
            if test_view_id >= len(dataset):
                logger.warning(f"⚠️ Requested view_id {test_view_id} >= dataset size {len(dataset)}, using view 0")
                test_view_id = 0
            
            # 获取测试batch - 模仿DataLoader的行为
            test_item = dataset[test_view_id]
            
            # 将单个样本转换为批次格式（模仿DataLoader的collate_fn）
            # DataLoader会将单个样本放入列表中，并且会把标量转换为张量
            # 包含数据、相机位姿、内参
            batch = {
                "data": [test_item["data"]],  # DataLoader创建的格式: [[1, H, W, 3]]
                "pose": [test_item["pose"]],  # DataLoader创建的格式: [[1, 4, 4]]
                "intr": [torch.tensor(test_item["intr"])]   # DataLoader创建的格式: [tensor(int)]
            }
            
            # 如果有mask，也要包含
            if "mask" in test_item:
                batch["mask"] = [test_item["mask"]]
            
            test_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            logger.info(f"🔍 Test batch type: {type(test_batch)}")
            logger.info(f"🔍 Test batch attributes: {[attr for attr in dir(test_batch) if not attr.startswith('_')]}")
            
            # 获取图像分辨率
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
            
            # 根据tensor维度确定H, W  
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
        """设置确定性运行环境"""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        
        # 设置CUDA确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info("🔧 Deterministic environment configured")
    
    def capture_baseline_outputs(self, output_path: str, test_view_id: int = 5) -> Dict[str, Any]:
        """
        按照原仓库逻辑进行完整推理，在推理前后捕获数据，保存为基准
        
        Args:
            output_path: 基准文件保存路径
            test_view_id: 测试视角ID
            
        Returns:
            捕获的基准数据
        """
        logger.info("📸 正在捕获K-Buffer基准数据...")
        
        # 1. 设置确定性环境
        self._setup_deterministic_environment()
        
        # 2. 加载数据集和checkpoint
        scene_data = self._load_treehill_scene(test_view_id)
        checkpoint = scene_data['checkpoint']
        gpu_batch = scene_data['test_batch']  # 这是通过get_gpu_batch_with_intrinsics得到的Batch对象
        
        # 3. 初始化tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # 使用Hydra正确加载配置 (模仿原始代码的做法)
        # config_path必须是相对路径，从当前文件到configs目录
        # config_name需要包含子目录路径
        config_name = "apps/colmap_3dgut.yaml"  # 配置文件在configs/apps/子目录中
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. 初始化MixtureOfGaussians模型（按照原仓库逻辑）
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint)
        model.build_acc()  # 构建加速结构
        logger.info("✅ MixtureOfGaussians model initialized")
        
        # 5. 进行完整推理，捕获trace的输入输出
        logger.info("🔄 Running inference and capturing trace inputs/outputs...")
        with torch.no_grad():
            # 这里的 model(gpu_batch) 内部会调用 self.renderer.render(self, gpu_batch, train, frame_id)
            # 即 tracer.render(gaussians=model, gpu_batch=gpu_batch, train=False, frame_id=0)
            trace_outputs = model(gpu_batch, train=False, frame_id=0)
        
        # 6. 保存基准数据
        baseline = {
            'trace_inputs': {
                # 保存trace函数的输入参数（按照原仓库逻辑：model + gpu_batch + train + frame_id）
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
                # 保存trace函数的输出结果
                'pred_rgb': trace_outputs['pred_rgb'].clone(),
                'pred_opacity': trace_outputs['pred_opacity'].clone(),
                'pred_dist': trace_outputs['pred_dist'].clone(),
                'hits_count': trace_outputs['hits_count'].clone(),
                'mog_visibility': trace_outputs['mog_visibility'].clone()
            },
            'metadata': {
                **scene_data['metadata'],
                'baseline_version': 'original_trace_function',
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': time.time(),
                'config_path': self.config_path
            }
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存基准
        torch.save(baseline, output_path)
        
        # 计算数据大小
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        logger.info(f"✅ Baseline saved: {output_path}")
        logger.info(f"   📊 Resolution: {baseline['metadata']['resolution']}")
        logger.info(f"   🔬 Gaussians: {baseline['trace_inputs']['model_state']['num_gaussians']}")
        logger.info(f"   💾 File size: {file_size_mb:.1f} MB")
        
        return baseline
    
    def capture_backward_baseline(self, output_path: str, test_view_id: int = 5) -> Dict[str, Any]:
        """
        捕获反向传播的基准数据，包括所有参数的梯度
        
        Args:
            output_path: 基准文件保存路径
            test_view_id: 测试视角ID
            
        Returns:
            捕获的反向传播基准数据
        """
        logger.info("📸 正在捕获K-Buffer反向传播基准数据...")
        
        # 1. 设置确定性环境
        self._setup_deterministic_environment()
        
        # 2. 加载数据集和checkpoint
        scene_data = self._load_treehill_scene(test_view_id)
        checkpoint = scene_data['checkpoint']
        gpu_batch = scene_data['test_batch']
        
        # 3. 初始化tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # 使用Hydra正确加载配置
        config_name = "apps/colmap_3dgut.yaml"
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. 初始化MixtureOfGaussians模型，并设置requires_grad=True
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint)
        model.build_acc()
        
        # 确保所有参数都需要梯度
        model.positions.requires_grad_(True)
        model.rotation.requires_grad_(True) 
        model.scale.requires_grad_(True)
        model.density.requires_grad_(True)
        model.features_albedo.requires_grad_(True)
        model.features_specular.requires_grad_(True)
        
        logger.info("✅ MixtureOfGaussians model initialized with gradients enabled")
        
        # 5. 前向传播
        logger.info("🔄 Running forward pass...")
        trace_outputs = model(gpu_batch, train=True, frame_id=0)  # train=True以启用梯度
        
        # 6. 构造损失函数并进行反向传播
        logger.info("🔄 Running backward pass...")
        
        # 使用ground truth构造RGB损失
        if gpu_batch.rgb_gt is not None:
            rgb_loss = torch.nn.functional.mse_loss(trace_outputs['pred_rgb'], gpu_batch.rgb_gt)
            logger.info(f"🔍 RGB MSE Loss: {rgb_loss.item():.6f}")
        else:
            # 如果没有ground truth，使用简单的sum损失来触发backward
            rgb_loss = trace_outputs['pred_rgb'].sum()
            logger.info(f"🔍 Simple sum loss: {rgb_loss.item():.6f}")
        
        # 清零之前的梯度
        model.zero_grad()
        
        # 反向传播
        rgb_loss.backward()
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        
        logger.info("✅ Backward pass completed")
        
        # 7. 保存基准数据（包括前向输出和梯度）
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
                # 保存所有参数的梯度
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
                'baseline_version': 'original_backward_function',
                'test_type': 'backward',
                'loss_type': 'mse_rgb' if gpu_batch.rgb_gt is not None else 'sum_rgb',
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': time.time(),
                'config_path': self.config_path
            }
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存基准
        torch.save(baseline, output_path)
        
        # 计算数据大小和梯度统计
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        # 统计非零梯度的参数
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
        验证修改后的K-Buffer实现
        
        Args:
            baseline_path: 基准文件路径
            tolerance_rgb: RGB差异容忍度
            tolerance_depth: 深度差异容忍度
            
        Returns:
            (是否通过, 详细测试结果)
        """
        logger.info("🔍 正在验证修改后的K-Buffer...")
        
        # 1. 加载基准数据
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
            
        baseline = torch.load(baseline_path, map_location='cuda', weights_only=False)
        trace_inputs = baseline['trace_inputs']
        expected_outputs = baseline['trace_outputs']
        
        logger.info(f"📂 Loaded baseline: {baseline_path}")
        logger.info(f"   🏷️  Version: {baseline['metadata']['baseline_version']}")
        logger.info(f"   📅 Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(baseline['metadata']['timestamp']))}")
        
        # 2. 设置确定性环境
        self._setup_deterministic_environment()
        
        # 3. 初始化tracer
        try:
            from omegaconf import DictConfig
            from hydra.compose import compose
            from hydra.initialize import initialize
        except ImportError:
            logger.error("❌ Cannot import Hydra/OmegaConf. Please install hydra-core and omegaconf")
            raise
        
        # 使用Hydra正确加载配置 (模仿原始代码的做法)
        # config_path必须是相对路径，从当前文件到configs目录
        # config_name需要包含子目录路径
        config_name = "apps/colmap_3dgut.yaml"  # 配置文件在configs/apps/子目录中
        
        with initialize(version_base=None, config_path="../../configs"):
            conf = compose(config_name=config_name)
        
        logger.info(f"✅ Config loaded with Hydra from {config_name}")
        
        # 4. 重建MixtureOfGaussians模型（按照原仓库逻辑）
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        self._restore_model_state(model, trace_inputs['model_state'])
        model.build_acc()  # 构建加速结构
        logger.info("✅ MixtureOfGaussians model restored")
        
        # 重建gpu_batch
        gpu_batch = self._rebuild_gpu_batch(trace_inputs['gpu_batch'])
        
        # 5. 运行修改后的版本
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
        
        # 6. 对比结果
        results = self._compare_trace_outputs(
            expected_outputs, 
            actual_outputs,
            tolerance_rgb, 
            tolerance_depth,
            baseline['metadata']
        )
        
        # 7. 添加高斯点处理统计
        gaussian_stats = self._analyze_gaussian_processing_stats(actual_outputs, baseline['metadata'])
        results['gaussian_processing_stats'] = gaussian_stats
        
        # 保存原始outputs供其他分析使用
        results['raw_outputs'] = actual_outputs
        
        # 8. 生成报告
        self._print_verification_report(results)
        
        return results['passed'], results
    
    def verify_backward_modification(self, baseline_path: str,
                                   tolerance_grad: float = 1e-5) -> Tuple[bool, Dict[str, Any]]:
        """
        验证修改后的反向传播实现
        
        Args:
            baseline_path: 反向传播基准文件路径
            tolerance_grad: 梯度差异容忍度
            
        Returns:
            (是否通过, 详细测试结果)
        """
        logger.info("🔍 正在验证修改后的反向传播...")
        
        # 1. 加载基准数据
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Backward baseline file not found: {baseline_path}")
            
        baseline = torch.load(baseline_path, map_location='cuda', weights_only=False)
        
        # 检查是否为backward基准文件
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
        
        # 2. 设置确定性环境
        self._setup_deterministic_environment()
        
        # 3. 初始化tracer
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
        
        # 4. 重建MixtureOfGaussians模型
        from threedgrut.model.model import MixtureOfGaussians
        model = MixtureOfGaussians(conf)
        self._restore_model_state(model, trace_inputs['model_state'])
        model.build_acc()
        
        # 设置requires_grad=True
        model.positions.requires_grad_(True)
        model.rotation.requires_grad_(True)
        model.scale.requires_grad_(True)
        model.density.requires_grad_(True)
        model.features_albedo.requires_grad_(True)
        model.features_specular.requires_grad_(True)
        
        logger.info("✅ MixtureOfGaussians model restored with gradients enabled")
        
        # 重建gpu_batch
        gpu_batch = self._rebuild_gpu_batch(trace_inputs['gpu_batch'])
        
        # 5. 运行修改后的版本
        # 确保启用梯度（默认就是启用的，无需context manager）
        logger.info("🔄 Running modified forward pass...")
        
        actual_outputs = model(
            gpu_batch, 
            train=trace_inputs['train'], 
            frame_id=trace_inputs['frame_id']
        )
        
        # 构造相同的损失函数
        logger.info("🔄 Running modified backward pass...")
        loss_type = baseline['metadata']['loss_type']
        
        if loss_type == 'mse_rgb' and gpu_batch.rgb_gt is not None:
            actual_loss = torch.nn.functional.mse_loss(actual_outputs['pred_rgb'], gpu_batch.rgb_gt)
        else:
            actual_loss = actual_outputs['pred_rgb'].sum()
            
        logger.info(f"🔍 Actual loss: {actual_loss.item():.6f}")
        
        # 清零梯度并进行反向传播
        model.zero_grad()
        actual_loss.backward()
        torch.cuda.synchronize()
        
        logger.info("✅ Modified backward pass completed")
        
        # 6. 收集实际梯度
        actual_gradients = {
            'positions_grad': model.positions.grad.clone() if model.positions.grad is not None else None,
            'rotation_grad': model.rotation.grad.clone() if model.rotation.grad is not None else None,
            'scale_grad': model.scale.grad.clone() if model.scale.grad is not None else None,
            'density_grad': model.density.grad.clone() if model.density.grad is not None else None,
            'features_albedo_grad': model.features_albedo.grad.clone() if model.features_albedo.grad is not None else None,
            'features_specular_grad': model.features_specular.grad.clone() if model.features_specular.grad is not None else None,
        }
        
        # 7. 对比结果
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
        
        # 8. 生成报告
        self._print_backward_verification_report(results)
        
        return results['passed'], results
    
    def _restore_model_state(self, model, model_state):
        """从保存的数据恢复模型状态"""
        import torch.nn as nn
        
        # 模型的参数需要是 torch.nn.Parameter 类型
        model.positions = nn.Parameter(model_state['positions'].clone())
        model.rotation = nn.Parameter(model_state['rotation'].clone())
        model.scale = nn.Parameter(model_state['scale'].clone())
        model.density = nn.Parameter(model_state['density'].clone())
        model.features_albedo = nn.Parameter(model_state['features_albedo'].clone())
        model.features_specular = nn.Parameter(model_state['features_specular'].clone())
        # 确保n_active_features正确设置
        model.n_active_features = model_state['n_active_features']
        
    def _rebuild_gpu_batch(self, batch_data):
        """从保存的数据重建gpu_batch"""
        from threedgrut.datasets.protocols import Batch
        
        # 重建完整的Batch对象，包含所有保存的属性
        kwargs = {
            'rays_ori': batch_data['rays_ori'].clone(),
            'rays_dir': batch_data['rays_dir'].clone(),
            'T_to_world': batch_data['T_to_world'].clone(),
        }
        
        # 添加可选属性（如果存在的话）
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
        """分析每个像素处理的高斯点数量统计"""
        hits_count_tensor = outputs['hits_count']
        logger.info(f"🔍 hits_count tensor shape: {hits_count_tensor.shape}")
        hits_count = hits_count_tensor.cpu().numpy()
        
        # 更健壮的维度处理
        original_shape = hits_count.shape
        logger.info(f"🔍 Original hits_count shape: {original_shape}")
        
        # 递归压缩所有size为1的维度，直到得到2D数组
        while len(hits_count.shape) > 2:
            # 找到第一个size为1的维度
            size_1_dims = [i for i, size in enumerate(hits_count.shape) if size == 1]
            if size_1_dims:
                hits_count = hits_count.squeeze(axis=size_1_dims[0])
            else:
                # 如果没有size为1的维度但还是>2D，可能是[B, C, H, W]格式
                # 尝试取第一个batch和第一个channel
                if len(hits_count.shape) == 4:  # [B, C, H, W]
                    hits_count = hits_count[0, 0]
                elif len(hits_count.shape) == 3:  # [B, H, W] 或 [C, H, W]
                    hits_count = hits_count[0]
                else:
                    break
        
        logger.info(f"🔍 Final hits_count shape: {hits_count.shape}")
        
        # 计算统计量
        valid_pixels = hits_count[hits_count > 0]  # 只统计有效像素
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
            'hits_count_raw': hits_count.copy()  # 保存原始数据用于可视化
        }
        
        return stats
    
    def _compute_hits_histogram(self, hits_count, bins=20):
        """计算hits_count的直方图"""
        max_hits = int(hits_count.max())
        if max_hits == 0:
            return {'bins': [], 'counts': []}
        
        # 创建直方图区间
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
        """对比trace函数的输出结果"""
        
        # 对比RGB
        rgb_expected = expected['pred_rgb']
        rgb_actual = actual['pred_rgb']
        rgb_diff = torch.abs(rgb_expected - rgb_actual)
        
        # 对比深度
        depth_expected = expected['pred_dist']
        depth_actual = actual['pred_dist']
        depth_diff = torch.abs(depth_expected - depth_actual)
        
        # 对比命中计数
        count_expected = expected['hits_count']
        count_actual = actual['hits_count']
        count_diff = torch.abs(count_expected - count_actual)
        
        # 计算统计量
        max_rgb_diff = rgb_diff.max().item()
        mean_rgb_diff = rgb_diff.mean().item()
        max_depth_diff = depth_diff.max().item()
        mean_depth_diff = depth_diff.mean().item()
        max_count_diff = count_diff.max().item()
        mean_count_diff = count_diff.mean().item()
        
        # 判断是否通过
        rgb_passed = max_rgb_diff < tolerance_rgb
        depth_passed = max_depth_diff < tolerance_depth
        count_passed = max_count_diff < 1e-6  # 命中计数应该完全一致
        
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
        """打印验证报告"""
        print("\n" + "🌳 " + "="*60)
        print("🌳 K-Buffer修改验证报告")
        print("🌳 " + "="*60)
        print(f"🏷️  基准版本: {results['metadata']['baseline_version']}")
        print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 分辨率: {results['metadata'].get('resolution', 'unknown')}")
        downsample_factor = results['metadata'].get('downsample_factor', 1)
        if downsample_factor > 1:
            print(f"📐 下采样因子: {downsample_factor}x")
        print("-" * 60)
        
        # RGB对比
        rgb_status = "✅ 通过" if results['max_rgb_diff'] < results['tolerance_rgb'] else "❌ 失败"
        print(f"🎨 RGB输出       | {rgb_status}")
        print(f"   最大误差: {results['max_rgb_diff']:.2e} (阈值: {results['tolerance_rgb']:.2e})")
        print(f"   平均误差: {results['mean_rgb_diff']:.2e}")
        print()
        
        # 深度对比
        depth_status = "✅ 通过" if results['max_depth_diff'] < results['tolerance_depth'] else "❌ 失败"
        print(f"📏 深度输出       | {depth_status}")
        print(f"   最大误差: {results['max_depth_diff']:.2e} (阈值: {results['tolerance_depth']:.2e})")
        print(f"   平均误差: {results['mean_depth_diff']:.2e}")
        print()
        
        # 命中计数对比
        count_status = "✅ 通过" if results['max_count_diff'] < 1e-6 else "❌ 失败"
        print(f"🎯 命中计数       | {count_status}")
        print(f"   最大误差: {results['max_count_diff']:.2e} (阈值: 1.00e-06)")
        print(f"   平均误差: {results['mean_count_diff']:.2e}")
        print()
        
        # 高斯点处理统计
        if 'gaussian_processing_stats' in results:
            self._print_gaussian_stats(results['gaussian_processing_stats'])
        
        # 总体评估
        overall_status = "🎉 整体通过" if results['passed'] else "⚠️  存在问题"
        print(f"🏆 {overall_status}")
        print("🌳 " + "="*60)
    
    def _print_gaussian_stats(self, stats: Dict[str, Any]):
        """打印高斯点处理统计信息"""
        print("🎯 高斯点处理统计")
        print("-" * 40)
        
        # 基本信息
        print(f"📊 图像分辨率: {stats['resolution'][1]}×{stats['resolution'][0]}")
        print(f"📊 总像素数: {stats['total_pixels']:,}")
        print(f"📊 有效像素数: {stats['valid_pixels']:,} ({stats['valid_pixel_ratio']:.1%})")
        print()
        
        # 全局统计
        print("🌐 全局统计:")
        print(f"   总高斯点击中: {stats['total_gaussian_hits']:,.0f}")
        print(f"   平均每像素: {stats['hits_per_pixel']['mean']:.1f} 个高斯点")
        print(f"   最大每像素: {stats['hits_per_pixel']['max']:.0f} 个高斯点")
        print(f"   中位数: {stats['hits_per_pixel']['median']:.1f} 个高斯点")
        print()
        
        # 有效像素统计（排除背景）
        if stats['valid_pixels'] > 0:
            print("🎯 有效像素统计 (排除背景):")
            print(f"   平均每像素: {stats['valid_hits_per_pixel']['mean']:.1f} 个高斯点")
            print(f"   最大每像素: {stats['valid_hits_per_pixel']['max']:.0f} 个高斯点")
            print(f"   最小每像素: {stats['valid_hits_per_pixel']['min']:.0f} 个高斯点")
            print(f"   标准差: {stats['valid_hits_per_pixel']['std']:.1f}")
            print()
        
        # 分布直方图
        if stats['histogram']['counts']:
            print("📈 击中分布:")
            self._print_histogram(stats['histogram'])
        print()
    
    def _print_histogram(self, histogram):
        """打印简单的ASCII直方图"""
        bins = histogram['bin_edges']
        counts = histogram['counts']
        
        if not counts or max(counts) == 0:
            print("   (无有效数据)")
            return
        
        # 归一化显示
        max_count = max(counts)
        max_width = 30  # 最大显示宽度
        
        for i in range(len(counts)):
            if i >= len(bins) - 1:
                break
            
            start_bin = bins[i]
            end_bin = bins[i + 1]
            count = counts[i]
            
            if count > 0:
                # 计算显示宽度
                width = int((count / max_count) * max_width)
                bar = "█" * width
                
                # 格式化输出
                print(f"   {start_bin:4.0f}-{end_bin:4.0f}: {bar} ({count:,})")
    
    def _compute_gradient_stats(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """计算梯度统计信息"""
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
        """对比反向传播的输出结果"""
        
        # 对比前向传播输出 (与forward test相同)
        rgb_expected = expected_outputs['pred_rgb']
        rgb_actual = actual_outputs['pred_rgb']
        rgb_diff = torch.abs(rgb_expected - rgb_actual)
        
        depth_expected = expected_outputs['pred_dist']
        depth_actual = actual_outputs['pred_dist']
        depth_diff = torch.abs(depth_expected - depth_actual)
        
        count_expected = expected_outputs['hits_count']
        count_actual = actual_outputs['hits_count']
        count_diff = torch.abs(count_expected - count_actual)
        
        # 对比损失值
        loss_diff = abs(expected_loss - actual_loss)
        
        # 对比梯度
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
                
                # 更新全局统计
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
                # 一个为None，一个不为None
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
        
        # 计算通过状态
        rgb_passed = rgb_diff.max().item() < 1e-6  # 前向输出应该完全一致
        depth_passed = depth_diff.max().item() < 1e-6
        count_passed = count_diff.max().item() < 1e-6
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
        """打印反向传播验证报告"""
        print("\n" + "🌳 " + "="*70)
        print("🌳 K-Buffer反向传播修改验证报告")
        print("🌳 " + "="*70)
        print(f"🏷️  基准版本: {results['metadata']['baseline_version']}")
        print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 分辨率: {results['metadata'].get('resolution', 'unknown')}")
        downsample_factor = results['metadata'].get('downsample_factor', 1)
        if downsample_factor > 1:
            print(f"📐 下采样因子: {downsample_factor}x")
        print("-" * 70)
        
        # 前向输出对比 (应该完全一致)
        forward_results = results['forward_outputs']
        print("🎨 前向输出验证 (应与原版本完全一致):")
        
        rgb_status = "✅ 通过" if forward_results['rgb_passed'] else "❌ 失败"
        print(f"   RGB输出:      {rgb_status} (最大差异: {forward_results['max_rgb_diff']:.2e})")
        
        depth_status = "✅ 通过" if forward_results['depth_passed'] else "❌ 失败"
        print(f"   深度输出:     {depth_status} (最大差异: {forward_results['max_depth_diff']:.2e})")
        
        count_status = "✅ 通过" if forward_results['count_passed'] else "❌ 失败"
        print(f"   命中计数:     {count_status} (最大差异: {forward_results['max_count_diff']:.2e})")
        print()
        
        # 损失值对比
        loss_results = results['loss']
        loss_status = "✅ 通过" if loss_results['passed'] else "❌ 失败"
        print(f"📉 损失值验证:   {loss_status}")
        print(f"   期望损失: {loss_results['expected']:.8f}")
        print(f"   实际损失: {loss_results['actual']:.8f}")
        print(f"   损失差异: {loss_results['diff']:.2e}")
        print()
        
        # 梯度对比
        grad_results = results['gradients']
        grad_status = "✅ 通过" if grad_results['passed'] else "❌ 失败"
        print(f"🎯 梯度验证:     {grad_status}")
        print(f"   最大梯度差异: {grad_results['max_diff']:.2e} (阈值: {grad_results['tolerance']:.2e})")
        print(f"   平均梯度差异: {grad_results['mean_diff']:.2e}")
        print()
        
        # 各参数梯度详情
        print("📊 各参数梯度对比:")
        param_comparisons = grad_results['param_comparisons']
        for param_name, comparison in param_comparisons.items():
            status = "✅" if comparison['passed'] else "❌"
            param_display_name = param_name.replace('_grad', '')
            print(f"   {param_display_name:20s} {status} "
                  f"(最大差异: {comparison['max_diff']:.2e}, "
                  f"期望范数: {comparison['expected_norm']:.2e}, "
                  f"实际范数: {comparison['actual_norm']:.2e})")
        print()
        
        # 总体评估
        overall_status = "🎉 整体通过" if results['passed'] else "⚠️  存在问题"
        print(f"🏆 {overall_status}")
        
        if not results['passed']:
            print("\n💡 调试建议:")
            if not forward_results['rgb_passed'] or not forward_results['depth_passed'] or not forward_results['count_passed']:
                print("   - 前向输出不一致，检查renderForward kernel的修改")
            if not loss_results['passed']:
                print("   - 损失值不一致，可能存在数值精度问题")
            if not grad_results['passed']:
                print("   - 梯度不一致，检查renderBackward kernel的修改")
                print("   - 可以尝试调整梯度容忍度参数")
        
        print("🌳 " + "="*70)


