// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <threedgutSlang.cuh>

// 内部实现参数（内存管理、系统架构相关）
struct model_InternalParams {
    static constexpr int GlobalParametersValueBufferIndex = 0; // 全局参数缓冲区（渲染设置，相机参数）
    static constexpr int DensityRawParametersBufferIndex  = 1; // 密度参数缓冲区（高斯粒子的密度和透明度）
    static constexpr int FeaturesRawParametersBufferIndex = 2; // 特征参数缓冲区（颜色，球谐系数等）

    static constexpr int DensityRawParametersGradientBufferIndex  = 0; // 密度参数梯度（训练时的密度梯度）
    static constexpr int FeaturesRawParametersGradientBufferIndex = 1; // 特征参数梯度（训练时的颜色梯度）

    static constexpr int FeatureShDegreeValueOffset = 4; // in bytes, offset in the global parameters buffer
};

// 外部算法参数
struct model_ExternalParams {
    static constexpr int FeaturesDim                 = 3;
    static constexpr float AlphaThreshold            = GAUSSIAN_PARTICLE_MIN_ALPHA;          // = 1.0/255.0
    static constexpr float MinTransmittanceThreshold = GAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD; // = 0.0001
    static constexpr int KernelDegree                = GAUSSIAN_PARTICLE_KERNEL_DEGREE;
    static constexpr float MinParticleKernelDensity  = GAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY;
    static const int RadianceMaxNumSphCoefficients   = PARTICLE_RADIANCE_NUM_COEFFS;
};

// ShGaussian<Particles>                    // 顶层：完整高斯模型
//       ↓
// ShRadiativeGaussianVolumetricFeaturesParticles<Internal,External,1>  // 中层：具体粒子类型
//       ↓
// model_InternalParams + model_ExternalParams  // 底层：参数配置

#include <3dgut/kernels/cuda/models/shRadiativeGaussianParticles.cuh>

using model_Particles = ShRadiativeGaussianVolumetricFeaturesParticles<model_InternalParams, model_ExternalParams, 1>;

#include <3dgut/kernels/cuda/models/shGaussianModel.cuh>

using model_ = ShGaussian<model_Particles>;

struct TGUTProjectorParams {
    static constexpr float ParticleMinSensorZ    = 0.2; // 相机近平面距离，避免过近粒子
    static constexpr float CovarianceDilation    = 0.3; // 协方差膨胀因子，用于控制粒子形状的模糊程度
    static constexpr float AlphaThreshold        = model_::Particles::AlphaThreshold; // 透明度阈值，用于过滤掉不透明度低的粒子
    static constexpr bool TightOpacityBounding   = GAUSSIAN_TIGHT_OPACITY_BOUNDING;
    static constexpr bool RectBounding           = GAUSSIAN_RECT_BOUNDING;
    static constexpr bool TileCulling            = GAUSSIAN_TILE_BASED_CULLING;
    static constexpr bool PerRayParticleFeatures = false;
    static constexpr float MaxDepthValue         = 3.4028235e+38;
    static constexpr bool GlobalZOrder           = GAUSSIAN_GLOBAL_Z_ORDER; // 全局深度排序
    static constexpr bool BackwardProjection     = false; // m_settings.renderMode == Settings::Splat
    static constexpr bool MipSplattingScaling    = true;
};

struct TGUTProjectionParams {
    static constexpr int NRollingShutterIterations = GAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS;
    static constexpr int D                         = 3;
    static constexpr float Alpha                   = GAUSSIAN_UT_ALPHA; // 控制sigma点的分布范围 
    static constexpr float Beta                    = GAUSSIAN_UT_BETA; // 融合高阶矩信息
    static constexpr float Kappa                   = GAUSSIAN_UT_KAPPA; // 融合高阶矩信息
    static constexpr float Delta                   = GAUSSIAN_UT_DELTA; ///< sqrt(Alpha*Alpha*(D+Kappa))
    static constexpr float ImageMarginFactor       = GAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR;
    static constexpr bool RequireAllSigmaPoints    = GAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID;
};

static_assert(TGUTProjectionParams::RequireAllSigmaPoints == false, "RequireAllSigmaPoints must be false");

#include <3dgut/kernels/cuda/renderers/gutProjector.cuh>

using TGUTProjector = GUTProjector<model_::Particles, TGUTProjectorParams, TGUTProjectionParams>;

struct TGUTRendererParams {
    static constexpr bool PerRayParticleFeatures = TGUTProjectorParams::PerRayParticleFeatures;
    static constexpr int KHitBufferSize          = GAUSSIAN_K_BUFFER_SIZE;
    static constexpr bool CustomBackward         = false;
};

#include <3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh>

using TGUTRenderer         = GUTKBufferRenderer<model_::Particles, TGUTRendererParams>;
using TGUTBackwardRenderer = GUTKBufferRenderer<model_::Particles, TGUTRendererParams, true>;

using TGUTModel = model_;

#include <3dgut/kernels/cuda/renderers/gutRenderer.cuh>
