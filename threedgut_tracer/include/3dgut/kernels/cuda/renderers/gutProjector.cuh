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

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/renderer/renderParameters.h>

template <typename Particles, typename Params, typename UTParams> // 粒子类型，projector参数类型，无迹变换参数类型
struct GUTProjector : Params, UTParams {  
    // 继承了两个参数结构体的所有成员
    // 可以直接访问这些参数作为成员变量
    using TFeaturesVec = typename Particles::TFeaturesVec;

    struct BoundingBox2D {
        tcnn::uvec2 min;
        tcnn::uvec2 max;
    };

    // 计算粒子在瓦片空间中的边界框
    // 粒子整体影响范围在瓦片网格里的最小和最大瓦片索引（二维坐标），形成一个矩形框。
    static inline __device__ BoundingBox2D computeTileSpaceBBox(const tcnn::uvec2& tileGrid, const tcnn::vec2& position, const tcnn::vec2& extent) {
        return BoundingBox2D{
            {
                min(tileGrid.x, max(0, static_cast<int>(floorf((position.x - 0.5f - extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(floorf((position.y - 0.5f - extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
            {
                min(tileGrid.x, max(0, static_cast<int>(ceilf((position.x - 0.5f + extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(ceilf((position.y - 0.5f + extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
        };
    }

    // 📄 【论文技术实现】tile/depth联合排序键构造
    // 论文描述："3DGS原本就采用了tile/depth联合排序键"
    // 🔧 将瓦片索引和深度索引合并为64位键值，用于后续的radix排序
    static inline __device__ uint64_t concatTileDepthKeys(uint32_t tileKey, uint32_t depthKey) {
        return (static_cast<uint64_t>(tileKey) << 32) | depthKey;
    }

    // 📄 【论文技术实现】基于Tile的剔除（Tile-based Culling）
    // 论文描述："我们计算每个tile X内能使2D高斯贡献G_2(x)最大的点ẋ：
    //          ẋ = arg max_{x∈X} G_2(x) = arg min_{x∈X} (x-μ₂)ᵀΣ₂⁻¹(x-μ₂)
    //          如果μ₂在X内，则ẋ=μ₂；否则，ẋ必定位于距离μ₂最近的两条tile边之一"
    //
    // 计算瓦片中粒子的最小强度响应（实现了论文的精确剔除算法）
    // 🎯 目的：避免对贡献低于εO=1/255的高斯进行无效计算
    // 
    // tcnn 函数说明：
    // tcnn::vec2: 二维向量（x, y），支持基本运算。
    // tcnn::vec4: 四维向量（x, y, z, w）。
    // tcnn::mix(a, b, c): 向量混合插值，返回 c 元素为 1 时取 a，否则取 b，等价于 select/lerp。
    // tcnn::copysign(a, b): 结果是 a 的绝对值但符号和 b 一致。
    // __frcp_rn(x): CUDA 的快速取倒数函数，等价于 1/x。
    // __saturatef(x): 截断/压缩，将 x 限定在 范围。
    static inline __device__ float tileMinParticlePowerResponse(const tcnn::vec2& tileCoords,
                                                                const tcnn::vec4& conicOpacity,
                                                                const tcnn::vec2& meanPosition) {

        const tcnn::vec2 tileSize = tcnn::vec2(threedgut::GUTParameters::Tiling::BlockX, threedgut::GUTParameters::Tiling::BlockY);
        const tcnn::vec2 tileMin  = tileSize * tileCoords;
        const tcnn::vec2 tileMax  = tileSize + tileMin;

        const tcnn::vec2 minOffset  = tileMin - meanPosition;
        const tcnn::vec2 leftAbove  = tcnn::vec2(minOffset.x > 0.0f, minOffset.y > 0.0f); // tileMin距离粒子中心点的偏移量是否大于0，大于0则表示在tileMin的左上方。
        const tcnn::vec2 notInRange = tcnn::vec2(leftAbove.x + (meanPosition.x > tileMax.x),
                                                 leftAbove.y + (meanPosition.y > tileMax.y));

        if ((notInRange.x + notInRange.y) > 0.0f) {
            // 🎯 粒子中心在tile外部，需要找到tile内使G_2(x)最大的点
            // 📐 实现论文公式：ẋ = arg max_{x∈X} G_2(x)
            // 找到最近的瓦片边界点（左上，左下，右上，右下）
            const tcnn::vec2 p    = tcnn::mix(tileMax, tileMin, leftAbove);
            // 按与最近点的方向设置tileSize的符号，如果minOffset为正正（在tileMin的左上方）
            // 注意为vec2，所以是x和y方向都考虑。
            const tcnn::vec2 dxy  = tcnn::copysign(tileSize, minOffset);
            // 粒子中心与最近点的矢量
            const tcnn::vec2 diff = meanPosition - p;
            // 归一化因子，分母是 tileSize² * conicOpacity（调控响应范围）
            const tcnn::vec2 rcp  = tcnn::vec2(__frcp_rn(tileSize.x * tileSize.x * conicOpacity.x),
                                               __frcp_rn(tileSize.y * tileSize.y * conicOpacity.z));
            // 分别计算响应在 x 和 y 维度上的强度，并保证落在[0,1]
            const float tx = notInRange.y * __saturatef((dxy.x * conicOpacity.x * diff.x + dxy.x * conicOpacity.y * diff.y) * rcp.x);
            const float ty = notInRange.x * __saturatef((dxy.y * conicOpacity.y * diff.x + dxy.y * conicOpacity.z * diff.y) * rcp.y);
            // 最近响应点与粒子质心的差值
            const tcnn::vec2 minPosDiff = meanPosition - tcnn::vec2(p.x + tx * dxy.x, p.y + ty * dxy.y);
            // 🧮 返回二次型混合响应：实现了论文的2D高斯响应计算G_2(ẋ)
            return 0.5f * (conicOpacity.x * minPosDiff.x * minPosDiff.x + conicOpacity.z * minPosDiff.y * minPosDiff.y) + conicOpacity.y * minPosDiff.x * minPosDiff.y;
        }
        // 🎯 粒子中心在tile内部，返回0（最大响应，必定通过剔除测试）
        // 对应论文："如果μ₂在X内，则ẋ=μ₂"
        return 0.f;
    }

    /// Convert a projected particle to its conic/opacity representation
    // 计算投影粒子的圆锥和不透明度表示
    // 将投影粒子的协方差描述转为圆锥系和不透明度的参数形式，便于后续高斯点云渲染或滤波
    // 这段代码为投影后的高斯粒子计算一个包围盒，用于：
    // 确定粒子影响的像素范围
    // 优化渲染性能（只处理相关像素）
    // 进行瓦片裁剪
    static inline __device__ bool computeProjectedExtentConicOpacity(tcnn::vec3 covariance, // 投影粒子的2D协方差
                                                                     float opacity, // 粒子的原始不透明度
                                                                     tcnn::vec2& extent, // 输出，粒子的包围半径/范围
                                                                     tcnn::vec4& conicOpacity, // 输出，圆锥/二次型系数和不透明度，形式[A,B,C,opacity]，用于表达高斯或椭圆滤波。
                                                                     float& maxConicOpacityPower) { // 输出，不透明度对阈值的对数规模，用于包围范围估算。
        // 通常为了处理反走样或软边界，多加一个正数，膨胀分布范围。通常只对对角线做加法，维持矩阵对称
        const tcnn::vec3 dilatedCovariance = tcnn::vec3{covariance.x + Params::CovarianceDilation, covariance.y, covariance.z + Params::CovarianceDilation};
        const float dilatedCovDet          = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
        if (dilatedCovDet == 0.0f) {
            return false;
        }
        // 其实就是协方差矩阵的逆矩阵，用于后续计算椭圆的形状和大小。
        conicOpacity.slice<0, 3>() = tcnn::vec3{dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x} / dilatedCovDet;

        // see Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
        // 参考论文 Mip-Splatting，缩放处理避免多重分辨率映射时的能量丢失。
        // 保证比例大于某阈值（0.000025f，防止精度溢出）。
        // opacity 根据尺度变化做自适应缩放。
        if constexpr (TGUTProjectorParams::MipSplattingScaling) {
            const float covDet            = covariance.x * covariance.z - covariance.y * covariance.y;
            const float convolutionFactor = sqrtf(fmaxf(0.000025f, covDet / dilatedCovDet));
            conicOpacity.w                = opacity * convolutionFactor;
        } else {
            conicOpacity.w = opacity;
        }

        if (conicOpacity.w < Params::AlphaThreshold) {
            return false; // 过滤掉太透明、几乎不可见的粒子，加速后续处理。
        }

        // maxConicOpacityPower = log(opacity / AlphaThreshold)：表示“从最大强度衰减到阈值需要的距离平方”。
        // 这是用来求解等强度边界的位置，也是包围盒的尺寸依据。
        maxConicOpacityPower     = logf(conicOpacity.w / Params::AlphaThreshold); // 取对数是常用的包围/阈值距离预估手法。

        // 根据不透明度衰减速度，计算包围盒的尺寸因子。
        // 如果使用tight opacity bounding，则限制在3.33倍以内，避免过度膨胀。
        // 否则使用默认值3.33，表示最大衰减距离。
        const float extentFactor = Params::TightOpacityBounding ? fminf(3.33f, sqrtf(2.0f * maxConicOpacityPower)) : 3.33f; // 比3\sigma安全
        const float minLambda    = 0.01f;
        const float mid          = 0.5f * (dilatedCovariance.x + dilatedCovariance.z); // 协方差矩阵的迹
        const float lambda       = mid + sqrtf(fmaxf(minLambda, mid * mid - dilatedCovDet)); // 最大特征值
        const float radius       = extentFactor * sqrtf(lambda);
        // 矩形边界 (RectBounding)：
        // X方向：min(extentFactor * √a, radius)
        // Y方向：min(extentFactor * √c, radius)
        // 沿椭圆主轴的紧密矩形，但不超过圆形边界
        extent                   = Params::RectBounding ? min(extentFactor * sqrt(tcnn::vec2{dilatedCovariance.x, dilatedCovariance.z}), tcnn::vec2{radius}) : tcnn::vec2{radius};

        return radius > 0.f;
    }

    // 无迹粒子投影（处理运动模糊和rolling shutter）
    static inline __device__ bool unscentedParticleProjection(
        const tcnn::ivec2& resolution, // 图像分辨率 [width, height]
        const threedgut::TSensorModel& sensorModel, // 相机内参模型（焦距、主点、畸变）
        const tcnn::vec3& sensorWorldPosition, // 相机在世界坐标系中的位置
        const tcnn::mat4x3& sensorMatrix, // 4x3 世界到相机的变换矩阵
        const threedgut::TSensorState& sensorShutterState, // Rolling Shutter状态（时间信息）
        const Particles& particles, // 粒子系统
        const typename Particles::DensityParameters& particleParameters, // 粒子参数
        tcnn::vec3& particleSensorRay, // 输出，从相机到粒子的射线向量
        float& particleProjOpacity, // 输出，投影后的不透明度
        tcnn::vec2& particleProjCenter, // 输出，粒子投影后的2D中心位置
        tcnn::vec3& particleProjCovariance) { // 输出，粒子投影的2D协方差矩阵

        // 获取粒子不透明度，过滤掉透明度低的粒子
        particleProjOpacity = particles.opacity(particleParameters);
        if (particleProjOpacity < Params::AlphaThreshold) {
            return false;
        }

        const tcnn::vec3& particleMean = particles.position(particleParameters); // 调的是函数
        // 判断粒子是否在相机近平面之外
        if ((particleMean.x * sensorMatrix[0][2] + particleMean.y * sensorMatrix[1][2] +
             particleMean.z * sensorMatrix[2][2] + sensorMatrix[3][2]) < Params::ParticleMinSensorZ) {
            return false;
        }

        const tcnn::vec3& particleScale = particles.scale(particleParameters);
        const tcnn::mat3 particleRotation = particles.rotation(particleParameters);

        particleSensorRay = particleMean - sensorWorldPosition; // 从相机位置指向粒子中心的射线向量（用于光照计算）

        int numValidPoints = 0; 
        tcnn::vec2 projectedSigmaPoints[2 * UTParams::D + 1];  // UTParams::D=3，总共7个点

        // 无迹变换的缩放参数
        // Alpha：通常0.001-1，控制Sigma点的分散程度
        // Kappa：通常0或3-D，次要调节参数
        // Lambda：主缩放因子，影响Sigma点的分布范围
        constexpr float Lambda = UTParams::Alpha * UTParams::Alpha * (UTParams::D + UTParams::Kappa) - UTParams::D;

        if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>( // NRollingShutterIterations：Rolling Shutter迭代次数
                particleMean,
                resolution,
                sensorModel,
                sensorShutterState,
                UTParams::ImageMarginFactor, // ImageMarginFactor：图像边缘扩展因子
                projectedSigmaPoints[0])) {
            numValidPoints++;
        }
        particleProjCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));

        constexpr float weightI = 1.f / (2.f * (UTParams::D + Lambda));
#pragma unroll
        for (int i = 0; i < UTParams::D; ++i) {
            // 计算沿第i个主轴的扰动向量
            // UTParams::Delta：扰动幅度，通常是 √(D+λ)
            // particleScale[i]：椭球沿第i轴的半径
            // particleRotation[i]：椭球的第i个主轴方向向量
            const tcnn::vec3 delta = UTParams::Delta * particleScale[i] * particleRotation[i]; ///< CHECK : column or row ?

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean + delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1])) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1];

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean - delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1 + UTParams::D])) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1 + UTParams::D];
        }

        // 编译时条件，是否要求所有Sigma点都有效
        // 如果开启严格模式，需要全部7个点都投影成功
        // 宽松模式下，至少要有1个点投影成功
        if constexpr (UTParams::RequireAllSigmaPoints) {
            if (numValidPoints < (2 * UTParams::D + 1)) {
                return false;
            }
        } else if (numValidPoints == 0) {
            return false;
        }

        {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[0] - particleProjCenter;
            constexpr float weight0        = Lambda / (UTParams::D + Lambda) + (1.f - UTParams::Alpha * UTParams::Alpha + UTParams::Beta);
            particleProjCovariance         = weight0 * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                                  centeredPoint.x * centeredPoint.y,
                                                                  centeredPoint.y * centeredPoint.y);
        }
#pragma unroll
        for (int i = 0; i < 2 * UTParams::D; ++i) {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[i + 1] - particleProjCenter;
            particleProjCovariance += weightI * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                           centeredPoint.x * centeredPoint.y,
                                                           centeredPoint.y * centeredPoint.y);
        }

        return true;
    }

    // 主投影函数：计算粒子投影和瓦片交集
    static inline __device__ void eval(tcnn::uvec2 tileGrid, // tile网格大小 [width, height]
                                       uint32_t numParticles, // 粒子数量
                                       tcnn::ivec2 resolution, // 图像分辨率 [width, height]
                                       threedgut::TSensorModel sensorModel, // 相机内参模型（焦距、主点、畸变）
                                       tcnn::vec3 sensorWorldPosition, // 相机在世界坐标系中的位置
                                       tcnn::mat4x3 sensorViewMatrix, // 4x3 世界到相机的变换矩阵
                                       threedgut::TSensorState sensorShutterState, // Rolling Shutter状态（时间信息）
                                       uint32_t* __restrict__ particlesTilesCountPtr, // 输出，每个粒子影响的tile数量
                                       tcnn::vec2* __restrict__ particlesProjectedPositionPtr, // 输出，每个粒子投影后的2D中心位置
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr, // 输出，每个粒子投影后的协方差矩阵逆矩阵和不透明度
                                       tcnn::vec2* __restrict__ particlesProjectedExtentPtr, // 输出，每个粒子投影后的包围半径
                                       float* __restrict__ particlesGlobalDepthPtr, // 输出，每个粒子投影后的深度（z轴）
                                       float* __restrict__ particlesPrecomputedFeaturesPtr, // 输出，预计算的特征向量
                                       int* __restrict__ particlesVisibilityCudaPtr, // 输出，粒子可见性标记
                                       threedgut::MemoryHandles parameters) { // GPU内存句柄

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters); // 初始化密度相关的GPU内存访问
        // 设置缓冲区指针，准备读取粒子位置、旋转、缩放、密度等参数
        const auto particleParameters = particles.fetchDensityParameters(particleIdx);

        tcnn::vec2 particleProjCenter; // 粒子投影后的2D中心位置
        float particleProjOpacity; // 粒子投影后的不透明度
        tcnn::vec3 particleSensorRay; // 从相机指向粒子的向量
        tcnn::vec3 particleProjCovariance; // 粒子投影后的协方差矩阵
        bool validProjection = false;
        { // 作用域块，局部化临时变量
            validProjection = unscentedParticleProjection(
                resolution,
                sensorModel,
                sensorWorldPosition,
                sensorViewMatrix,
                // FIXME : work directly in sensor space to avoid all intermediate transforms
                sensorShutterState,
                particles,
                particleParameters,
                particleSensorRay,
                particleProjOpacity,
                particleProjCenter,
                particleProjCovariance);
        }

        tcnn::vec2 particleProjExtent;
        tcnn::vec4 particleProjConicOpacity;
        float particleMaxConicOpacityPower;
        bool validConicEstimation = false;
        {
            validConicEstimation = computeProjectedExtentConicOpacity(particleProjCovariance,
                                                                 particleProjOpacity,
                                                                 particleProjExtent,
                                                                 particleProjConicOpacity,
                                                                 particleMaxConicOpacityPower);
        }

        particlesVisibilityCudaPtr[particleIdx] = validConicEstimation ? 1 : 0; // 1表示可见，0表示不可见

        validProjection = validProjection && validConicEstimation;

        uint32_t numValidTiles = 0;
        if (validProjection) {
            const BoundingBox2D tileBBox = computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent);
            // 📄 【论文技术实现】精确Tile剔除 vs 保守估计
            // 论文描述："与3DGS类似，我们首先用2D协方差矩阵最大特征值确定轴对齐包围矩形，
            //          估算高斯可能影响的tile范围。这种保守估计对于高度各向异性的高斯会给出很大的范围。
            //          为了精确剔除，我们计算每个tile X内能使2D高斯贡献G_2(x)最大的点"
            //
            // 精确模式（TileCulling=true）- 实现论文的精确剔除算法
            // 逐个检查包围盒内的每个瓦片
            // 使用高斯函数精确计算瓦片中心的响应值
            // 只有响应值超过阈值(εO=1/255)的瓦片才计数
            if constexpr (Params::TileCulling) {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                    for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x) {
                        // 🎯 调用论文实现的精确剔除函数：计算G_2(ẋ)并与阈值比较
                        if (tileMinParticlePowerResponse(tcnn::vec2(x, y), particleProjConicOpacity, particleProjCenter) < particleMaxConicOpacityPower) {
                            numValidTiles++;  // 通过精确剔除测试
                        }
                    }
                }
            } else {
                // 🔄 保守估计模式：使用包围盒估计（对应论文提到的"保守估计"问题）
                numValidTiles = (tileBBox.max.x - tileBBox.min.x) * (tileBBox.max.y - tileBBox.min.y);
            }
        }

        particlesTilesCountPtr[particleIdx] = numValidTiles;
        if (numValidTiles == 0) {
            particlesProjectedPositionPtr[particleIdx]     = tcnn::vec2::zero();
            particlesProjectedConicOpacityPtr[particleIdx] = tcnn::vec4::zero();
            particlesProjectedExtentPtr[particleIdx]       = tcnn::vec2::zero();
            particlesGlobalDepthPtr[particleIdx]           = 0.f;
            return;
        }

        const float particleSensorDistance = length(particleSensorRay); // 从相机指向粒子的向量长度

        if constexpr (!Params::PerRayParticleFeatures) {
            particles.initializeFeatures(parameters);
            reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesPtr)[particleIdx] =
                particles.template featuresCustomFromBuffer<false>(particleIdx, particleSensorRay / particleSensorDistance);
        }

        particlesProjectedPositionPtr[particleIdx]     = particleProjCenter;
        particlesProjectedConicOpacityPtr[particleIdx] = particleProjConicOpacity;
        particlesProjectedExtentPtr[particleIdx]       = particleProjExtent;
        // 📄 【论文技术实现】Tile深度调整（Tile-depth Adjustment）
        // 论文描述："预排序时需要为每个tile选取一个代表性的topt。直观上，tile中心射线应是tile内所有射线的合理折中"
        if constexpr (Params::GlobalZOrder) {
            // 🎯 使用Z深度（对应论文的topt概念）
            // 计算粒子在相机坐标系中的Z轴深度，用于全局排序
            const tcnn::vec3& particleMean       = particles.position(particleParameters);
            particlesGlobalDepthPtr[particleIdx] = (particleMean.x * sensorViewMatrix[0][2] + particleMean.y * sensorViewMatrix[1][2] +
                                                    particleMean.z * sensorViewMatrix[2][2] + sensorViewMatrix[3][2]);
        } else {
            // 📅 使用欧几里得距离（传统方法）
            // 简单但不如Z深度精确，可能导致排序误差
            particlesGlobalDepthPtr[particleIdx] = particleSensorDistance;
        }
    }

    // 展开函数：生成粒子-瓦片交集列表
    static inline __device__ void expand(tcnn::uvec2 tileGrid,
                                         int numParticles,
                                         tcnn::ivec2 /*resolution*/,
                                         threedgut::TSensorModel /*sensorModel*/,
                                         threedgut::TSensorState /*sensorState*/,
                                         const uint32_t* __restrict__ particlesTilesOffsetPtr, // 每个粒子在输出数组中的起始偏移量
                                         const tcnn::vec2* __restrict__ particlesProjectedPositionPtr, // 粒子投影中心位置
                                         const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr, // Conic系数+不透明度
                                         const tcnn::vec2* __restrict__ particlesProjectedExtentPtr, // 粒子包围范围
                                         const float* __restrict__ particlesGlobalDepthPtr, // 粒子深度值
                                         threedgut::MemoryHandles parameters, // GPU内存句柄
                                         uint64_t* __restrict__ unsortedTileDepthKeysPtr, // 输出，未排序的深度键
                                         uint32_t* __restrict__ unsortedTileParticleIdxPtr) { // 输出，未排序的粒子索引

        const int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (particleIdx >= numParticles) {
            return;
        }

        const tcnn::vec2 particleProjExtent = particlesProjectedExtentPtr[particleIdx];
        // check the particle projected extent x
        constexpr float eps = 1e-06f;
        if (particleProjExtent.x <= eps) {
            return;
        }

        // 📄 【论文技术实现】tile/depth联合排序键生成
        // 论文描述："3DGS原本就采用了tile/depth联合排序键，可以用每个tile中心射线的topt值替代全局深度进行排序"
        //
        // 保持IEEE 754浮点数的位模式，用于排序
        // 将float深度值重新解释为uint32_t整数，用于排序（对应论文的深度排序键）
        const uint32_t depthKey             = *reinterpret_cast<const uint32_t*>(&particlesGlobalDepthPtr[particleIdx]);
        uint32_t tileOffset                 = (particleIdx == 0) ? 0 : particlesTilesOffsetPtr[particleIdx - 1]; // 每个粒子在输出数组中的起始偏移量
        const tcnn::vec2 particleProjCenter = particlesProjectedPositionPtr[particleIdx];
        const BoundingBox2D tileBBox        = computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent);

        if constexpr (Params::TileCulling) {

            const uint32_t maxTileOffset = particlesTilesOffsetPtr[particleIdx];

            const tcnn::vec4 conicOpacity    = particlesProjectedConicOpacityPtr[particleIdx];
            const float maxConicOpacityPower = logf(conicOpacity.w / Params::AlphaThreshold);

            for (int y = tileBBox.min.y; (y < tileBBox.max.y) && (tileOffset < maxTileOffset); ++y) {
                for (int x = tileBBox.min.x; (x < tileBBox.max.x) && (tileOffset < maxTileOffset); ++x) {
                    if (tileMinParticlePowerResponse(tcnn::vec2(x, y), conicOpacity, particleProjCenter) < maxConicOpacityPower) {
                        unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + x, depthKey);
                        unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                        tileOffset++;
                    }
                }
            }
            for (; tileOffset < maxTileOffset; ++tileOffset) {
                unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(threedgut::GUTParameters::Tiling::InvalidTileIdx,
                                                                             __float_as_uint(Params::MaxDepthValue));
                unsortedTileParticleIdxPtr[tileOffset] = threedgut::GUTParameters::InvalidParticleIdx;
            }

        } else {

            for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x) {
                    unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + x, depthKey);
                    unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                    tileOffset++;
                }
            }
        }
    }

    // 反向投影函数：计算投影参数的梯度
    static inline __device__ void
    evalBackward(tcnn::uvec2 tileGrid,
                 uint32_t numParticles,
                 tcnn::ivec2 resolution,
                 threedgut::TSensorModel sensorModel,
                 tcnn::vec3 sensorWorldPosition,
                 tcnn::mat4x3 sensorViewMatrix,
                 const uint32_t* __restrict__ particlesTilesCountPtr,
                 threedgut::MemoryHandles parameters,
                 const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                 const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                 const float* __restrict__ particlesGlobalDepthGradPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                 threedgut::MemoryHandles parametersGradient) {
        if constexpr (Params::PerRayParticleFeatures) {
            return;
        }

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }
        if (particlesTilesCountPtr[particleIdx] == 0) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters);
        const tcnn::vec3 incidentDirection = tcnn::normalize(particles.fetchPosition(particleIdx) - sensorWorldPosition);

        particles.initializeFeatures(parameters);
        particles.initializeFeaturesGradient(parametersGradient);
        particles.featuresBwdCustomToBuffer<false>(
            particleIdx,
            reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr)[particleIdx],
            reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr)[particleIdx],
            incidentDirection);
        particles.initializeDensityGradient(parametersGradient);
        particles.template densityIncidentDirectionBwdToBuffer<true>(particleIdx, sensorWorldPosition);
    }
};
