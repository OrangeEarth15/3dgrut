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

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

// 光线击中粒子的数据结构
struct HitParticle {
    static constexpr float InvalidHitT = -1.0f; // 无效击中的标记值
    int idx                            = -1; // 粒子索引（-1表示无效）
    float hitT                         = InvalidHitT; // 击中距离（沿射线的参数T），射线点 = 起点 + t * 方向
    float alpha                        = 0.0f; // 粒子的不透明度
};

// ========== K-Buffer核心数据结构 ==========
//
// 📄 【论文技术实现】局部重排序（Local Reordering）
// 论文描述："逐射线的重排序扩展...在寄存器中保留一个小的重排序窗口。
//          每加载一个高斯，计算其t_opt并用插入排序放入窗口；
//          若窗口溢出，则混合最小深度的样本。"
//
// 【功能】：维护每条光线的K个最近击中粒子，实现多层透明度混合
// 【设计原理】：
//   - 固定大小缓冲区，避免动态内存分配（对应论文的"寄存器窗口"）
//   - 升序排列：m_kbuffer[0]最近，m_kbuffer[K-1]最远（按hitT排序，即论文的t_opt）
//   - 满时策略：处理最近击中，为新击中腾出空间（对应论文的"窗口溢出处理"）
// 【窗口大小】：论文建议16-24，3DGUT通过GAUSSIAN_K_BUFFER_SIZE配置
//
template <int K>
struct HitParticleKBuffer {
    
    // ========== 构造函数：初始化空缓冲区 ==========
    __device__ HitParticleKBuffer() {
        m_numHits = 0; // 当前缓冲区中存储的有效击中数量
        
        // 将所有槽位初始化为无效状态
        // #pragma unroll: 编译器指令，展开循环提高性能
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle(); // 默认构造：idx=-1, hitT=-1.0f, alpha=0.0f
        }
    }

    // ========== 核心插入算法：维持升序排列 ==========
    //
    // 📄 【论文技术实现】插入排序的局部重排序窗口
    // 论文描述："计算其t_opt并用插入排序放入窗口"
    //
    // 【算法思想】：插入排序变体，新击中会被插入到正确位置
    // 【排序规则】：按hitT升序排列（hitT对应论文的t_opt，最近的在前面）
    // 【满时策略】：标记最近击中为无效，准备被处理（对应论文的"混合最小深度样本"）
    // 【性能优化】：#pragma unroll展开循环，寄存器级操作
    //
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        
        if (isFull) {
            // ⚠️ 关键策略：缓冲区满时，标记最近击中为"待处理"
            // 设置hitT为Invalid，这样在后续的排序中，这个击中会被移到末尾
            // 实际上这是在为processHitParticle做准备
            m_kbuffer[0].hitT = HitParticle::InvalidHitT; 
        } else {
            m_numHits++; // 还有空间，增加计数
        }
        
        // ========== 插入排序核心算法 ==========
        // 从后往前遍历，寻找新击中的正确插入位置
        // 如果新的hitT更大，就与当前位置交换，继续向前寻找
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            // 关键条件：新击中距离 > 当前位置距离
            // 说明新击中应该放在更靠后的位置（更远）
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                // 交换：将更远的击中向后移动
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
            // 否则已找到正确位置，停止交换
        }
        
        // 算法结果：
        // - m_kbuffer[0]: 最近击中（hitT最小）
        // - m_kbuffer[K-1]: 最远击中（hitT最大）
        // - 所有击中按hitT升序排列
    }

    // ========== 访问器方法 ==========
    
    // 获取指定索引位置的击中粒子（只读访问）
    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    // 获取当前缓冲区中有效击中的数量
    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    // 检查缓冲区是否已满
    inline __device__ bool full() const {
        return m_numHits == K;
    }

    // ========== 核心访问方法 ==========
    //
    // 【功能】：获取最近的击中粒子（用于立即处理）
    // 【设计】：总是返回m_kbuffer[0]，因为它是hitT最小的击中
    // 【用途】：当缓冲区满时，需要立即处理最近击中为新击中腾出空间
    //
    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];  // 返回距离最近的击中（hitT最小）
    }

private:
    HitParticle m_kbuffer[K];  // 击中粒子数组，按hitT升序排列
    uint32_t m_numHits;        // 当前有效击中数量 [0, K]
};

// ========== K=0特化：完全禁用K-Buffer ==========
//
// 📄 【论文技术对比】退化到传统3DGS渲染
// 论文描述："不进行片段合并，要求射线上的高斯近似有序"
//
// 【用途】：当KHitBufferSize=0时，完全跳过K-Buffer机制
// 【优势】：
//   - 零运行时开销：所有方法都是constexpr no-op
//   - 编译时优化：相关代码会被完全移除
//   - 适用场景：不需要多层混合的简单渲染（退化到传统3DGS）
// 【性能】：避免了局部重排序开销，但可能产生popping伪影
//
template <>
struct HitParticleKBuffer<0> {
    // 所有方法都是constexpr，编译时求值，运行时零开销
    constexpr inline __device__ void insert(HitParticle& hitParticle) const { 
        /* no-op: 不执行任何操作 */ 
    }
    
    constexpr inline __device__ HitParticle operator[](int) const { 
        return HitParticle();  // 返回默认的无效击中
    }
    
    constexpr inline __device__ uint32_t numHits() const { 
        return 0;  // 永远没有击中
    }
    
    constexpr inline __device__ bool full() const { 
        return true;  // 永远"满"，确保不会尝试插入
    }
    
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { 
        return hitParticle;  // 直接返回输入的击中（用于立即处理）
    }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec; // vector3

    using TRayPayload         = RayPayload<Particles::FeaturesDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::FeaturesDim>;

    // 用于优化内存访问的缓存结构，预先加载粒子数据
    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedRawParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityRawParameters densityParameters;
    };

    template <typename TRayPayload> // 处理单个击中粒子，计算颜色和透明度混合
    static inline __device__ void processHitParticle(
        TRayPayload& ray,                                     // 输入输出：光线数据载荷，包含累积特征、透射率等（会被修改）
        const HitParticle& hitParticle,                      // 输入：击中粒子信息，包含索引、不透明度、击中距离等（只读）
        const Particles& particles,                          // 输入：粒子系统接口，提供特征和密度计算方法（只读）
        const TFeaturesVec* __restrict__ particleFeatures,   // 输入：预计算特征数组指针（静态模式下的粒子颜色/RGB，只读）
        TFeaturesVec* __restrict__ particleFeaturesGradient) { // 输出：特征梯度数组指针（训练模式下累积梯度，可写）
        // 处理K-Buffer中单个击中粒子对光线的影响，支持前向渲染和反向梯度计算
        // 反向传播模式：计算梯度，用于神经网络训练
        // 前向渲染模式：计算最终颜色，用于图像生成
        if constexpr (Backward) {
            float hitAlphaGrad = 0.f; // alpha参数的梯度
            if constexpr (Params::PerRayParticleFeatures) {
                // 【动态特征模式】：每条光线动态计算粒子特征（如球谐光照） 
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction, // 光线方向（影响球谐计算）
                                                              hitParticle.alpha, // 当前alpha值
                                                              hitAlphaGrad, // 输出：alpha梯度
                                                              hitParticle.idx, // 粒子索引
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction), // 动态特征
                                                              ray.featuresBackward, // 输出：光线特征梯度
                                                              ray.featuresGradient); // 输入输出：光线特征梯度梯度
            } else {
                // 【静态特征模式】：使用预计算的粒子特征
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticle.alpha,
                                               hitAlphaGrad,
                                               particleFeatures[hitParticle.idx],
                                               particleFeaturesGradientVec,
                                               ray.featuresBackward,
                                               ray.featuresGradient);
            // 原子累加到全局梯度缓冲区
#pragma unroll
                for (int i = 0; i < Particles::FeaturesDim; ++i) {
                    atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                }
            }
            //  将alpha梯度反向传播到粒子的几何参数（位置、旋转、缩放、密度），用于优化
            particles.densityProcessHitBwdToBuffer<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient);

            ray.transmittance *= (1.0 - hitParticle.alpha);

        } else {
            // ========== densityIntegrateHit调用链 - 第1层：K-Buffer渲染器 ==========
            //
            // 📍 【调用链结构】Alpha混合的权重计算核心
            // 1. 【当前层】K-Buffer (gutKBufferRenderer.cuh:244) → particles.densityIntegrateHit()
            // 2. C++包装层 (shRadiativeGaussianParticles.cuh:344) → particleDensityIntegrateHit()
            // 3. Slang导出层 (gaussianParticles.slang:873) → gaussianParticle.integrateHit<false>()
            // 4. 核心实现层 (gaussianParticles.slang:557) → 实际Alpha混合计算
            //
            // 【本层作用】：渲染管线中的权重计算请求
            // - 为当前击中粒子计算对最终像素的贡献权重
            // - 执行标准的Alpha混合公式：weight = alpha * transmittance  
            // - 同时更新深度和透射率，维持渲染状态的一致性
            // - 返回权重值供后续颜色混合使用
            //
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,    // 输入：粒子不透明度[0,1]，控制遮挡强度
                                              ray.transmittance,    // 输入输出：当前光线透射率，会被递减
                                              hitParticle.hitT,     // 输入：光线击中距离（沿光线的参数t）
                                              ray.hitT);            // 输入输出：光线累积深度，按权重更新
            
            // ========== featureIntegrateFwd调用链 - 第1层：K-Buffer渲染器 ==========
            //
            // 🎨 【调用链结构】颜色特征的加权混合累积  
            // 1. 【当前层】K-Buffer (gutKBufferRenderer.cuh:251) → particles.featureIntegrateFwd()
            // 2. C++包装层 (shRadiativeGaussianParticles.cuh:638) → particleFeaturesIntegrateFwd()
            // 3. Slang导出层 (shRadiativeParticles.slang:298) → shRadiativeParticle.integrateRadiance<false>()
            // 4. 核心实现层 (shRadiativeParticles.slang:底层) → 实际特征加权累积
            //
            // 【本层作用】：颜色特征的Alpha混合计算
            // - 获取粒子的颜色/辐射特征（RGB或球谐系数）
            // - 按权重累积到光线的总颜色中：ray.features += weight * particleFeatures
            // - 支持静态特征（预计算RGB）和动态特征（球谐光照）两种模式
            // - 累积结果将成为最终像素的RGB颜色值
            //
            particles.featureIntegrateFwd(
                hitWeight,                                          // 输入：混合权重，由densityIntegrateHit计算得出
                Params::PerRayParticleFeatures ?                   // 条件分支：特征模式选择
                    particles.featuresFromBuffer(hitParticle.idx, ray.direction) :  // 动态模式：球谐光照，视角相关
                    tcnn::max(particleFeatures[hitParticle.idx], 0.f),            // 静态模式：预计算RGB，视角无关
                ray.features);                                     // 输入输出：光线累积特征，会被更新

            if (hitWeight > 0.0f) ray.countHit(); // 统计有效击中次数，用于渲染质量分析
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            // 后续粒子对最终颜色贡献可忽略
            ray.kill(); // 提前终止优化 - 当透射率过低意味着光线被完全阻挡
        }
    }

    template <typename TRay>
    // K-Buffer主渲染函数：处理单条光线与tile内粒子的相互作用
    static inline __device__ void eval(
        const threedgut::RenderParameters& params,           // 渲染参数配置
        TRay& ray,                                           // 光线数据(会被修改)
        const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,  // 每个tile的粒子范围[start,end]
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,      // 排序后的粒子索引数组
        // 这么做的好处是：
        // 不给参数起名或把名字注释掉，可以避免“未使用参数”的编译警告；
        // 注释里写上原来的名字，则又保留了文档信息，方便阅读和维护。
        const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,     // 未使用：粒子投影位置
        const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/, // 未使用：粒子投影椭圆+不透明度
        const float* __restrict__ /*particlesGlobalDepthPtr*/,                // 未使用：粒子全局深度
        const float* __restrict__ particlesPrecomputedFeaturesPtr,            // 预计算粒子特征(RGB等)
        threedgut::MemoryHandles parameters,                                  // GPU内存句柄集合
        tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,     // 梯度：位置
        tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,     // 梯度：椭圆+不透明度  
        float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,     // 梯度：深度
        float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,     // 梯度：特征
        threedgut::MemoryHandles parametersGradient                        = {}) {        // 梯度内存句柄

        using namespace threedgut;

        // === 计算当前线程的tile和线程索引 ===
        const uint32_t tileIdx = blockIdx.y * gridDim.x + blockIdx.x;      // 当前处理的tile索引(2D->1D)
        const uint32_t tileThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;  // 当前线程在tile内的索引
        
        // === 获取当前tile内的粒子信息 ===
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];  // 粒子范围[start,end]
        uint32_t tileNumParticlesToProcess = tileParticleRangeIndices.y - tileParticleRangeIndices.x;  // 要处理的粒子数量
        const uint32_t tileNumBlocksToProcess = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);  // 需要的数据块数
        
        // === 设置特征缓冲区指针 ===
        // 根据是否使用per-ray特征(球谐函数等)来决定使用预计算特征还是动态特征
        const TFeaturesVec* particleFeaturesBuffer = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        // === 初始化粒子系统 ===
        Particles particles;  // 粒子接口对象
        particles.initializeDensity(parameters);  // 初始化密度计算相关参数
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);  // 反向模式：初始化密度梯度
        }
        particles.initializeFeatures(parameters);  // 初始化特征计算相关参数
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);  // 反向模式：初始化特征梯度
        }

        // === 根据模式选择处理路径 ===
        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            // 路径1: 反向传播 + 无K缓冲 = 直接处理模式
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            // 路径2: 前向传播 或 使用K缓冲 = K-Buffer模式  
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    // 主渲染函数
    static inline __device__ void evalBalanced(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                       const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                       const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       const tcnn::uvec2& tile,
                                       const tcnn::uvec2& tileGrid,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {}) {
        using namespace threedgut;

        const int balancedTileIdx = tile.y * tileGrid.x + tile.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[balancedTileIdx];
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        // 其实这个接口代表是使用SH（与dir有关），还是单纯的rgb
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            // 反向传播且不使用K缓冲
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);

        } else if constexpr (Params::KHitBufferSize == 0) {
            // 前向传播且无K缓冲：使用Gaussian-wise并行优化
            evalForwardNoKBuffer_optimized(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            // 前向传播使用K缓冲：内存优化版本
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    // 前向无K缓冲的Gaussian-wise优化版本：简洁高效的实现
    static inline __device__ void evalForwardNoKBuffer_optimized(
        TRay& ray,                                    
        Particles& particles,                         
        const tcnn::uvec2& tileParticleRangeIndices, 
        uint32_t tileNumBlocksToProcess,             
        uint32_t tileNumParticlesToProcess,          
        const uint32_t tileThreadIdx,                
        const uint32_t* __restrict__ sortedTileParticleIdxPtr, 
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,     
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) { 
        
        static_assert(!Backward && (Params::KHitBufferSize == 0), "Optimized path for forward pass with no KBuffer");
        using namespace threedgut;
        
        // 适度增大共享内存批处理（避免超限）
        constexpr uint32_t SHMEM_SIZE_MULTIPLIER = 2;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize * SHMEM_SIZE_MULTIPLIER];
        
        const uint32_t laneId = tileThreadIdx % 32;
        const uint32_t expandedBlockSize = GUTParameters::Tiling::BlockSize * SHMEM_SIZE_MULTIPLIER;
        const uint32_t expandedNumBlocksToProcess = tcnn::div_round_up(tileNumParticlesToProcess, expandedBlockSize);
        
        // 主循环：分批处理
        for (uint32_t i = 0; i < expandedNumBlocksToProcess; i++, tileNumParticlesToProcess -= expandedBlockSize) {
            
            if (__syncthreads_and(!ray.isAlive())) break;
            
            // === 🏗️ 协作式数据预取 ===
            uint32_t baseProgress = tileParticleRangeIndices.x + i * expandedBlockSize + tileThreadIdx;
            
            for (uint32_t j = 0; j < SHMEM_SIZE_MULTIPLIER; j++) {
                uint32_t currentProgress = baseProgress + j * GUTParameters::Tiling::BlockSize;
                uint32_t sharedMemIdx = tileThreadIdx + j * GUTParameters::Tiling::BlockSize;
                
                if (currentProgress < tileParticleRangeIndices.y) {
                    const uint32_t particleIdx = sortedTileParticleIdxPtr[currentProgress];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                        prefetchedParticlesData[sharedMemIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                    } else {
                        prefetchedParticlesData[sharedMemIdx].idx = GUTParameters::InvalidParticleIdx;
                    }
                } else {
                    prefetchedParticlesData[sharedMemIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            }
            __syncthreads();

            // === Gaussian-wise并行：完全按照render_warp的正确实现
            //
            // render_warp证明了gaussian-wise并行是可行的
            // 关键是正确实现：
            // 1. 外层循环：遍历32条光线
            // 2. 内层循环：32线程并行处理高斯点  
            // 3. 并行前缀积：正确处理透射率依赖
            // 4. Warp归约：正确累积特征到目标光线
            
            uint32_t alignedParticleCount = ((min(expandedBlockSize, tileNumParticlesToProcess) + 31) / 32) * 32;
            
            // 外层循环：遍历warp内32条光线（完全参考render_warp）
            for (uint32_t rayLane = 0; rayLane < 32; rayLane++) {
                
                // 检查当前光线状态
                bool rayDone = __shfl_sync(0xffffffff, !ray.isAlive(), rayLane);
                if (rayDone) continue;
                
                // 获取当前光线的数据（通过shuffle）
                tcnn::vec3 currentRayOrigin, currentRayDirection;
                tcnn::vec2 currentRayTMinMax;
                float currentRayTransmittance, currentRayHitT;
                TFeaturesVec currentRayFeatures;
                
                currentRayOrigin.x = __shfl_sync(0xffffffff, ray.origin.x, rayLane);
                currentRayOrigin.y = __shfl_sync(0xffffffff, ray.origin.y, rayLane);
                currentRayOrigin.z = __shfl_sync(0xffffffff, ray.origin.z, rayLane);
                currentRayDirection.x = __shfl_sync(0xffffffff, ray.direction.x, rayLane);
                currentRayDirection.y = __shfl_sync(0xffffffff, ray.direction.y, rayLane);
                currentRayDirection.z = __shfl_sync(0xffffffff, ray.direction.z, rayLane);
                currentRayTMinMax.x = __shfl_sync(0xffffffff, ray.tMinMax.x, rayLane);
                currentRayTMinMax.y = __shfl_sync(0xffffffff, ray.tMinMax.y, rayLane);
                currentRayTransmittance = __shfl_sync(0xffffffff, ray.transmittance, rayLane);
                currentRayHitT = __shfl_sync(0xffffffff, ray.hitT, rayLane);
                
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    currentRayFeatures[featIdx] = __shfl_sync(0xffffffff, ray.features[featIdx], rayLane);
                }
                
                // 临时累积变量（每个线程维护）
                TFeaturesVec tempFeatures = TFeaturesVec::zero();
                float tempWeight = 0.0f;
                float tempDepth = 0.0f;
                uint32_t tempHitCount = 0;  // 关键修正：统计实际击中次数
                
                // 内层循环：32线程并行处理高斯点（核心算法）
                for (uint32_t j = laneId; j < alignedParticleCount; j += 32) {
                    
                    if (rayDone) break;
                    
                    float hitAlpha = 0.0f;
                    float hitT = 0.0f;
                    TFeaturesVec hitFeatures = TFeaturesVec::zero();
                    bool validHit = false;
                    
                    // 步骤1：每个线程检测一个高斯点
                    if (j < min(expandedBlockSize, tileNumParticlesToProcess)) {
                        const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                        
                        if (particleData.idx != GUTParameters::InvalidParticleIdx) {
                            if (particles.densityHit(currentRayOrigin,
                                                   currentRayDirection,
                                                   particleData.densityParameters,
                                                   hitAlpha,
                                                   hitT) &&
                                (hitT > currentRayTMinMax.x) &&
                                (hitT < currentRayTMinMax.y)) {
                                
                                validHit = true;
                                
                                // 获取高斯点特征
                                if constexpr (Params::PerRayParticleFeatures) {
                                    hitFeatures = particles.featuresFromBuffer(particleData.idx, currentRayDirection);
                                } else {
                                    hitFeatures = tcnn::max(particleFeaturesBuffer[particleData.idx], 0.f);
                                }
                            }
                        }
                    }
                    
                    // 如果warp内无击中，跳过
                    if (__all_sync(0xffffffff, !validHit)) continue;
                    
                    // 步骤2：并行前缀积计算透射率（完全参考render_warp）
                    float oneMinusAlpha = validHit ? (1.0f - hitAlpha) : 1.0f;
                    
                    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                        float n = __shfl_up_sync(0xffffffff, oneMinusAlpha, offset);
                        if (laneId >= offset) {
                            oneMinusAlpha *= n;
                        }
                    }
                    
                    // 关键修正：按照render_warp第299行的精确逻辑
                    float testT = currentRayTransmittance * oneMinusAlpha;  // test_T = w_T * one_alpha
                    
                    // 步骤3：早停检测（render_warp第300-308行逻辑）
                    uint32_t anyDone = __ballot_sync(0xffffffff, testT < Particles::MinTransmittanceThreshold);
                    if (anyDone) {
                        if (laneId == rayLane) {
                            ray.kill();
                        }
                        rayDone = true;
                        break;
                    }
                    
                    // 步骤4：特征积分（render_warp第310-316行的精确逻辑）
                    float wT = testT;  // 每个线程的局部透射率状态（对应render_warp的w_T）
                    
                    if (validHit && testT >= Particles::MinTransmittanceThreshold) {
                        // 关键修正：按照render_warp第311行恢复透射率
                        wT /= (1.0f - hitAlpha);  // 恢复处理当前高斯点前的透射率
                        
                        // render_warp第313-315行：alpha * test_T 计算权重
                        float hitWeight = hitAlpha * wT;
                        
                        // 累积贡献（完全按照render_warp）
                        for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                            tempFeatures[featIdx] += hitFeatures[featIdx] * hitWeight;
                        }
                        tempWeight += hitWeight;
                        tempDepth += hitT * hitWeight;
                        
                        if (hitWeight > 0.0f) {
                            tempHitCount++;
                        }
                    }
                    
                    // 关键修正：按照render_warp第317行同步透射率
                    // 注意：这里更新的是处理完当前高斯点后的透射率状态
                    currentRayTransmittance = __shfl_sync(0xffffffff, testT, 31);  // 使用testT而不是wT
                }
                
                // 步骤5：按照render_warp第319-324行的精确归约逻辑
                
                // 关键修正：直接按照warp_prefixsum_to_lane实现归约
                // 特征归约（对应render_warp第322行）
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    float src = tempFeatures[featIdx];
                    src += __shfl_up_sync(0xffffffff, src, 1);
                    src += __shfl_up_sync(0xffffffff, src, 2);
                    src += __shfl_up_sync(0xffffffff, src, 4);
                    src += __shfl_up_sync(0xffffffff, src, 8);
                    src += __shfl_up_sync(0xffffffff, src, 16);
                    src = __shfl_sync(0xffffffff, src, 31);
                    if (rayLane == laneId) {
                        currentRayFeatures[featIdx] += src;  // 对应render_warp: dst += src
                    }
                }
                
                // 深度归约（对应render_warp第324行w_D）
                {
                    float src = tempDepth;
                    src += __shfl_up_sync(0xffffffff, src, 1);
                    src += __shfl_up_sync(0xffffffff, src, 2);
                    src += __shfl_up_sync(0xffffffff, src, 4);
                    src += __shfl_up_sync(0xffffffff, src, 8);
                    src += __shfl_up_sync(0xffffffff, src, 16);
                    src = __shfl_sync(0xffffffff, src, 31);
                    if (rayLane == laneId) {
                        currentRayHitT += src;
                    }
                }
                
                // 命中计数修正：按实际击中的高斯点数量计数
                // 原始逻辑：每个高斯点如果有贡献(hitWeight > 0)就计数一次
                // Gaussian-wise版本：统计32个线程中有多少个高斯点有贡献
                {
                    uint32_t src = tempHitCount;  // 每个线程统计自己处理的高斯点击中次数
                    src += __shfl_up_sync(0xffffffff, src, 1);
                    src += __shfl_up_sync(0xffffffff, src, 2);
                    src += __shfl_up_sync(0xffffffff, src, 4);
                    src += __shfl_up_sync(0xffffffff, src, 8);
                    src += __shfl_up_sync(0xffffffff, src, 16);
                    src = __shfl_sync(0xffffffff, src, 31);
                    if (rayLane == laneId) {
                        // 关键修正：按实际有贡献的高斯点数量调用
                        // 如果32个线程中总共有N个高斯点有贡献，就调用N次ray.countHit()
                        for (uint32_t h = 0; h < src; h++) {
                            ray.countHit();
                        }
                    }
                }
                
                // 步骤6：更新目标光线状态（对应render_warp第319-320行）
                if (laneId == rayLane) {  // 对应render_warp: if (lane_id % 32 == l)
                    ray.transmittance = currentRayTransmittance;  // T = w_T
                    ray.hitT = currentRayHitT;
                    for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                        ray.features[featIdx] = currentRayFeatures[featIdx];
                    }
                }
            }
        }
    }

    template <typename TRay>
    // Fine-grained warp-level处理函数 - 基于gaussian-wise并行 (算法3优化版)
    static inline __device__ void evalFineGrainedWarp(
        const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
        const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
        const float* __restrict__ particlesGlobalDepthPtr,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       const tcnn::uvec2& tile,
                                       const tcnn::uvec2& tileGrid,
        const int laneId,
                                       threedgut::MemoryHandles parameters,
        tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr     = nullptr,
        tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr = nullptr,
        float* __restrict__ particlesGlobalDepthGradPtr                = nullptr,
        float* __restrict__ particlesPrecomputedFeaturesGradPtr        = nullptr,
        threedgut::MemoryHandles parametersGradient                    = {}) {

        using namespace threedgut;
        
        // 使用原始16x16 tile的粒子数据，每个warp处理1个pixel
        const uint32_t tileIdx = tile.y * tileGrid.x + tile.x;
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];
        
        uint32_t tileNumParticlesToProcess = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        
        const TFeaturesVec* particleFeaturesBuffer = 
            Params::PerRayParticleFeatures ? nullptr : 
            reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = 
            (Params::PerRayParticleFeatures || !Backward) ? nullptr : 
            reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (Params::KHitBufferSize == 0) {
            // K=0时使用Gaussian-wise并行处理（类似evalForwardNoKBuffer_optimized的单光线版本）
            
            uint32_t alignedParticleCount = ((tileNumParticlesToProcess + 31) / 32) * 32;
            
            // Gaussian-wise并行：32线程并行处理高斯点，单条光线
            for (uint32_t j = laneId; j < alignedParticleCount; j += 32) {
                
                if (!ray.isAlive()) break;
                
                float hitAlpha = 0.0f;
                float hitT = 0.0f;
                TFeaturesVec hitFeatures = TFeaturesVec::zero();
                bool validHit = false;
                
                // 🔍 **步骤1：每个线程检测一个高斯点**
                if (j < tileNumParticlesToProcess) {
                    const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + j;
                    const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                    
                    if (particleIdx != GUTParameters::InvalidParticleIdx) {
                        auto densityParams = particles.fetchDensityParameters(particleIdx);
                        
                        if (particles.densityHit(ray.origin,
                                               ray.direction,
                                               densityParams,
                                               hitAlpha,
                                               hitT) &&
                            (hitT > ray.tMinMax.x) &&
                            (hitT < ray.tMinMax.y)) {
                            
                            validHit = true;
                            
                            // 获取高斯点特征
                            if constexpr (Params::PerRayParticleFeatures) {
                                hitFeatures = particles.featuresFromBuffer(particleIdx, ray.direction);
        } else {
                                hitFeatures = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                            }
                        }
                    }
                }
                
                // 如果warp内无击中，跳过
                if (__all_sync(0xffffffff, !validHit)) continue;
                
                // 步骤2：计算每个线程的透射率贡献
                float localTransmittance = validHit ? (1.0f - hitAlpha) : 1.0f;
                
                // 步骤3：Warp内前缀积计算累积透射率
                for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                    float n = __shfl_up_sync(0xffffffff, localTransmittance, offset);
                    if (laneId >= offset) {
                        localTransmittance *= n;
                    }
                }
                
                // 当前warp处理的粒子批次对ray透射率的影响
                float batchTransmittance = __shfl_sync(0xffffffff, localTransmittance, 31);
                float newTransmittance = ray.transmittance * batchTransmittance;
                
                // 🚨 **步骤4：早停检测**
                if (newTransmittance < Particles::MinTransmittanceThreshold) {
                    ray.kill();
                    break;
                }
                
                // 💫 **步骤5：Warp内归约计算特征贡献**
                TFeaturesVec accumulatedFeatures = TFeaturesVec::zero();
                float accumulatedHitT = 0.0f;
                uint32_t accumulatedHitCount = 0;
                
                if (validHit) {
                    // 使用已计算的前缀透射率（localTransmittance在当前线程包含了前面所有线程的累积）
                    // 我们需要的是不包括当前粒子的前缀透射率
                    float prefixTransmittance = (laneId > 0) ? 
                        (localTransmittance / (1.0f - hitAlpha)) : 1.0f;
                    float particleTransmittance = ray.transmittance * prefixTransmittance;
                    float hitWeight = hitAlpha * particleTransmittance;
                    
                    // 计算特征贡献
                    for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                        accumulatedFeatures[featIdx] = hitFeatures[featIdx] * hitWeight;
                    }
                    accumulatedHitT = hitT * hitWeight;
                    accumulatedHitCount = (hitWeight > 0.0f) ? 1 : 0;
                }
                
                // 步骤6：Warp内归约求和
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    for (uint32_t offset = 16; offset > 0; offset /= 2) {
                        accumulatedFeatures[featIdx] += __shfl_down_sync(0xffffffff, accumulatedFeatures[featIdx], offset);
                    }
                }
                
                for (uint32_t offset = 16; offset > 0; offset /= 2) {
                    accumulatedHitT += __shfl_down_sync(0xffffffff, accumulatedHitT, offset);
                    accumulatedHitCount += __shfl_down_sync(0xffffffff, accumulatedHitCount, offset);
                }
                
                // 步骤7：只有lane 0更新ray（避免数据竞争）
                if (laneId == 0) {
                    for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                        ray.features[featIdx] += accumulatedFeatures[featIdx];
                    }
                    ray.hitT += accumulatedHitT;
                    ray.countHit(accumulatedHitCount);
                }
                
                // 步骤8：更新透射率
                ray.transmittance = newTransmittance;
            }
            
        } else {
            // K>0时使用传统K-Buffer处理
            
            HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;
            const uint32_t tileNumWarpIterations = tcnn::div_round_up(tileNumParticlesToProcess, 32u);
            
            for (uint32_t i = 0; i < tileNumWarpIterations; i++, tileNumParticlesToProcess -= 32) {
                
                if (__all_sync(0xFFFFFFFF, !ray.isAlive())) {
                    break;
                }
                
                // 每个lane处理一个粒子
                uint32_t particleIdx = GUTParameters::InvalidParticleIdx;
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * 32 + laneId;
                
                if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                    particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                }
                
                if (particleIdx != GUTParameters::InvalidParticleIdx && ray.isAlive()) {
                    
                    HitParticle hitParticle;
                    hitParticle.idx = particleIdx;
                    
                    auto densityParams = particles.fetchDensityParameters(particleIdx);
                    
                    if (particles.densityHit(ray.origin,
                                           ray.direction,
                                           densityParams,
                                           hitParticle.alpha,
                                           hitParticle.hitT) &&
                        (hitParticle.hitT > ray.tMinMax.x) &&
                        (hitParticle.hitT < ray.tMinMax.y)) {
                        
                        // K-Buffer插入逻辑
                        if (hitParticleKBuffer.full()) {
                            processHitParticle(ray,
                                             hitParticleKBuffer.closestHit(hitParticle),
                                             particles,
                                             particleFeaturesBuffer,
                                             particleFeaturesGradientBuffer);
                        }
                        
                        hitParticleKBuffer.insert(hitParticle);
                    }
                }
            }
            
            // 处理K-Buffer中剩余的击中
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                 hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                 particles,
                                 particleFeaturesBuffer,
                                 particleFeaturesGradientBuffer);
            }
        }
    }

    template <typename TRay>
    // 使用K缓冲的渲染函数 - 实现多层透明度混合的核心算法
    static inline __device__ void evalKBuffer(
        TRay& ray,                                    // 光线数据，包含起点、方向、透射率等，会被修改
        Particles& particles,                         // 粒子系统，提供密度和特征计算接口
        const tcnn::uvec2& tileParticleRangeIndices, // 当前瓦片中粒子的索引范围 [start, end)
        uint32_t tileNumBlocksToProcess,             // 需要处理的数据块数量（用于分批处理）
        uint32_t tileNumParticlesToProcess,          // 该瓦片中需要处理的粒子总数
        const uint32_t tileThreadIdx,                // 当前线程在瓦片内的索引（用于共享内存寻址）
        const uint32_t* __restrict__ sortedTileParticleIdxPtr, // 全局排序后的粒子索引数组
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,     // 预计算的粒子特征缓冲区（如果不使用per-ray特征）
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) { // 特征梯度缓冲区（反向传播时使用）
        using namespace threedgut;
        // 声明共享内存数组，用于协作式数据预取
        // BlockSize个线程协作加载BlockSize个粒子的数据，减少全局内存访问延迟
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        // 为每条光线创建私有的K-Buffer，大小为KHitBufferSize
        // 用于存储最近的K个粒子击中，实现多层透明度混合
        HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;

        // 主循环：分批处理粒子，每批处理BlockSize个粒子
        // 这样设计是因为共享内存有限，无法一次性加载所有粒子数据
        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            // 早停优化：如果warp中所有线程的光线都已死亡，则提前退出
            // __syncthreads_and() 确保所有线程都满足条件时才返回true
            // 这避免了无效的计算，提高GPU利用率
            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // === 集体数据预取阶段 ===
            // 计算当前线程要预取的粒子在全局排序数组中的索引
            // 每个线程负责预取一个粒子的数据到共享内存
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            
            // 边界检查：确保不超出当前瓦片的粒子范围
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                // 从全局排序数组中获取实际的粒子索引
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                
                // 检查粒子索引是否有效（-1U表示无效粒子，用于填充）
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    // 预取粒子的密度参数到共享内存
                    // fetchDensityParameters() 从全局内存加载粒子的几何和密度信息
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    // 标记为无效粒子
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                // 超出范围，标记为无效
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            
            // 同步屏障：等待所有线程完成数据预取
            // 确保共享内存中的数据对所有线程都可见
            __syncthreads();

            // === 粒子处理阶段 ===
            // 处理当前批次中的每个预取的粒子
            // min() 确保不处理超过剩余粒子数量的数据
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                // 从共享内存获取预取的粒子数据
                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                
                // 检查粒子是否有效
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    // 遇到无效粒子时强制退出外层循环
                    // 因为粒子是排序的，后续粒子也都是无效的
                    i = tileNumBlocksToProcess;
                    break;
                }

                // 初始化击中粒子结构
                HitParticle hitParticle;
                hitParticle.idx = particleData.idx; // 设置粒子索引

                // ========== 粒子击中检测与验证 ==========
                
                // 【第一步：几何相交测试】densityHit()
                // 功能：计算光线与3D高斯粒子的相交情况
                // 原理：将光线变换到粒子的标准化空间（椭球变为单位球），然后进行射线-球相交测试
                // 算法流程：
                //   1. 光线变换：rayOrigin/rayDirection → canonicalRayOrigin/canonicalRayDirection
                //   2. 核函数计算：maxResponse = exp(-0.5 * minSquaredDistance)
                //   3. 透明度计算：alpha = min(MaxAlpha, maxResponse * density)
                //   4. 击中距离：hitT = canonicalRayDistance() (光线参数化距离t，使得P = origin + t*direction)
                //
                // 注意：hitT ≠ globalDepth！
                // - globalDepth：粒子中心到相机的距离（用于全局排序）
                // - hitT：光线与粒子表面相交的参数化距离（用于精确排序）
                if (particles.densityHit(ray.origin,                    // 输入：光线起点世界坐标
                                       ray.direction,                   // 输入：光线方向向量（归一化）
                                       particleData.densityParameters,  // 输入：从共享内存预取的粒子参数
                                       hitParticle.alpha,              // 输出：计算得到的粒子不透明度[0,1]
                                       hitParticle.hitT) &&            // 输出：光线击中距离（t参数）
                
                // 【第二步：有效范围验证】确保击中点在光线的有效区间内
                // ray.tMinMax.x：光线起始距离（通常为相机近平面或AABB入口）
                // ray.tMinMax.y：光线终止距离（通常为相机远平面或AABB出口）
                    (hitParticle.hitT > ray.tMinMax.x) &&              // 击中点不在光线起点之前
                    (hitParticle.hitT < ray.tMinMax.y)) {              // 击中点不在光线终点之后

                    // === K-Buffer核心逻辑 ===
                    // 如果K缓冲区已满，需要为新击中让出空间
                    if (hitParticleKBuffer.full()) {
                        // 立即处理最近的击中（索引0）
                        // 这实现了"流式处理"：边发现边处理较近的击中
                        // closestHit() 返回距离最小的击中粒子
                        processHitParticle(ray,
                                         hitParticleKBuffer.closestHit(hitParticle), // 最近的击中
                                         particles,
                                         particleFeaturesBuffer,
                                         particleFeaturesGradientBuffer);
                    }
                    
                    // 将新击中插入K-Buffer
                    // insert() 会自动维护升序排列，新击中会被插入到合适位置
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        // === 最终处理阶段 ===
        // 编译时检查：只有当K > 0时才执行最终处理
        // 这是编译器优化，K=0时这段代码会被完全移除
        if constexpr (Params::KHitBufferSize > 0) {
            // 处理K-Buffer中剩余的所有击中
            // 按从近到远的顺序处理（升序排列）
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                // 计算正确的索引：从最近的开始处理
                // KHitBufferSize - numHits() + i 确保从有效击中的起始位置开始
                processHitParticle(ray,
                                 hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                 particles,
                                 particleFeaturesBuffer,
                                 particleFeaturesGradientBuffer);
            }
        }
    }

    template <typename TRay>
    // 不使用K缓冲的反向渲染函数
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedRawParticlesData[tileThreadIdx].features = TFeaturesVec::zero();
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].features = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                    }
                    prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = make_float3(0.0f);
                densityRawParametersGrad.quaternion = make_float4(0.0f);
                densityRawParametersGrad.scale      = make_float3(0.0f);

                TFeaturesVec featuresGrad = TFeaturesVec::zero();

                if (ray.isAlive()) {
                    particles.processHitBwd<Params::PerRayParticleFeatures>(
                        ray.origin,
                        ray.direction,
                        particleData.idx,
                        particleData.densityParameters,
                        &densityRawParametersGrad,
                        particleData.features,
                        &featuresGrad,
                        ray.transmittance,
                        ray.transmittanceBackward,
                        ray.transmittanceGradient,
                        ray.features,
                        ray.featuresBackward,
                        ray.featuresGradient,
                        ray.hitT,
                        ray.hitTBackward,
                        ray.hitTGradient);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                }
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};
