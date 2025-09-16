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

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];  // 返回距离最近的击中（hitT最小）
    }

private:
    HitParticle m_kbuffer[K];  // 击中粒子数组，按hitT升序排列
    uint32_t m_numHits;        // 当前有效击中数量 [0, K]
};

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
        if constexpr (Backward) {
            float hitAlphaGrad = 0.f; // alpha参数的梯度
            if constexpr (Params::PerRayParticleFeatures) {
                // 【动态特征模式】：每条光线动态计算粒子特征（如球谐光照）
                // 📋 FEATURES反向传播调用链（第1层/5层）：
                // 1. 【当前层】gutKBufferRenderer.cuh:161 → particles.featuresIntegrateBwdToBuffer<false>()
                // 2. 第2层：shRadiativeGaussianParticles.cuh:753 → particleFeaturesIntegrateBwdToBuffer()  
                // 3. 第3层：shRadiativeParticles.slang:485 → bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)()
                // 4. 第4层：shRadiativeParticles.slang:294 → integrateRadianceFromBuffer<backToFront>()
                // 5. 第5层：shRadiativeParticles.slang:301 → integrateRadianceFromParameters<backToFront>() + sphericalHarmonics.decode<>() + fetchParametersFromBuffer()
                //
                // 🎯 调用链功能：
                // - 从RGB梯度反向传播到球谐系数梯度
                // - 处理视角相关的光照计算
                // - 自动更新全局球谐参数缓冲区
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
            // 📋 DENSITY反向传播调用链（第1层/4层）：
            // 1. 【当前层】gutKBufferRenderer.cuh:195 → particles.densityProcessHitBwdToBuffer<false>()
            // 2. 第2层：shRadiativeGaussianParticles.cuh:417 → particleDensityProcessHitBwdToBuffer() 
            // 3. 第3层：gaussianParticles.slang:1063 → bwd_diff(gaussianParticle.processHitFromBuffer<false>)()
            // 4. 第4层：gaussianParticles.slang:内部实现 → 几何变换梯度计算 + 原子更新到全局梯度缓冲区
            //
            // 🎯 调用链功能：
            // - 将alpha梯度反向传播到粒子几何参数（位置、旋转、缩放、密度）
            // - 处理坐标变换链的梯度传播（世界坐标→粒子局部坐标）
            // - 计算高斯核函数对几何参数的敏感度
            // - 自动更新全局参数梯度缓冲区（用于神经网络优化）
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
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,    // 输入：粒子不透明度[0,1]，控制遮挡强度
                                              ray.transmittance,    // 输入输出：当前光线透射率，会被递减
                                              hitParticle.hitT,     // 输入：光线击中距离（沿光线的参数t）
                                              ray.hitT);            // 输入输出：光线累积深度，按权重更新
            
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
    // Fine-grained warp-level处理函数 - 基于gaussian-wise并行 (算法3原始版本 - 备份保留)
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

        static_assert(Params::KHitBufferSize == 0, "evalFineGrainedWarp only supports K=0 (no hit buffer). Use evalKBuffer for K>0 cases.");
        
        // K=0时使用Gaussian-wise并行处理
        // 将粒子数量向上对齐到32的倍数（warp大小），确保无warp divergence
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
                    auto densityParams = particles.fetchDensityParameters(particleIdx); // 不连续访问
                    
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
                
            // 🚨 **步骤4：早停检测 - 使用__ffs精确确定终止位置**
            unsigned int early_termination_mask = __ballot_sync(0xffffffff, 
                validHit && (ray.transmittance * localTransmittance) < Particles::MinTransmittanceThreshold);
            
            bool should_terminate = false;
            int termination_lane = -1;
            
            if (early_termination_mask) {
                termination_lane = __ffs(early_termination_mask) - 1; // 找到第一个满足条件的lane
                should_terminate = true;
                ray.kill();
            }
            
            // 💫 **步骤5：Warp内归约计算特征贡献**
            TFeaturesVec accumulatedFeatures = TFeaturesVec::zero();
            float accumulatedHitT = 0.0f;
            uint32_t accumulatedHitCount = 0;
                
            // 只计算终止位置之前（包括终止位置）的有效粒子贡献
            bool should_contribute = validHit && (!should_terminate || laneId <= termination_lane);
            
            if (should_contribute) {
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
                for (uint32_t offset = 16; offset > 0; offset >>= 1) {
                    accumulatedFeatures[featIdx] += __shfl_down_sync(0xffffffff, accumulatedFeatures[featIdx], offset);
                }
            }
            
            for (uint32_t offset = 16; offset > 0; offset >>= 1) {
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
            
            // 如果发生了early termination，跳出循环
            if (should_terminate) {
                break;
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

    /**
     * 🔄 无K缓冲反向渲染函数 - 调用链第1层：反向传播主调度器
     * 
     * 📋 调用链总览:
     *   1. 【当前层】evalBackwardNoKBuffer (gutKBufferRenderer.cuh:600) - 主调度器
     *      ├── particles.processHitBwd<>() → 第2层 → 第3层(最底层)
     *      ├── particles.processHitBwdUpdateFeaturesGradient() → 第2层(最终层)
     *      └── particles.processHitBwdUpdateDensityGradient() → 第2层(最终层)
     * 
     * 🎯 主调度器职责:
     *   - 按block批量预取粒子数据到shared memory
     *   - 逐粒子协调三个子系统：梯度计算 + 特征更新 + 几何更新
     *   - 管理光线生命周期（ray.isAlive()）和早停优化
     *   - 协调warp级同步和内存访问模式
     * 
     * 🚀 性能优化:
     *   - Shared memory协同加载减少全局内存访问
     *   - Warp级同步优化 (__all_sync) 提前退出无效光线
     *   - 批量梯度更新减少原子操作开销
     */
    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(
        TRay& ray,                                                  // 反向光线 (包含Forward结果和损失梯度)
        Particles& particles,                                       // 粒子系统接口
        const tcnn::uvec2& tileParticleRangeIndices,               // 当前tile的粒子索引范围 [start, end)
        uint32_t tileNumBlocksToProcess,                           // 需要处理的block数量
        uint32_t tileNumParticlesToProcess,                        // 需要处理的粒子总数
        const uint32_t tileThreadIdx,                              // tile内的thread索引
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,     // 排序后的粒子索引数组
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,   // 粒子特征数据 (Forward计算)
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer  // 粒子特征梯度输出缓冲区
    ) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        // 📦 Shared memory缓存: 一次预取整个block的粒子数据 (通常256个)
        __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];

        // 🔄 按block循环处理粒子
        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            // ========== 步骤1: 协同预取粒子数据到Shared Memory ==========
            __syncthreads();  // 确保所有thread开始新的批次
            if (!ray.isAlive()) {
                break;  // 光线已终止，提前退出
            }

            // 📍 每个thread负责加载一个粒子的数据
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    // 🔧 加载粒子的几何参数 (位置、旋转、缩放、密度)
                    prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                    
                    // 🎨 加载粒子特征 (颜色等)
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedRawParticlesData[tileThreadIdx].features = TFeaturesVec::zero();  // 动态计算特征
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].features = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);  // 预计算特征 + ReLU
                    }
                    prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    // 无效粒子标记
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                // 超出范围标记
                prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();  // 确保所有粒子数据已加载完成

            // ========== 步骤2: 逐粒子处理反向传播 ==========
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                // 🚀 Warp级优化: 如果整个warp的光线都死亡，提前退出
                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();  // 遇到无效粒子，终止光线
                    break;
                }

                // 📊 初始化梯度累积器
                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;         // ∂L/∂密度
                densityRawParametersGrad.position   = make_float3(0.0f);  // ∂L/∂位置
                densityRawParametersGrad.quaternion = make_float4(0.0f);  // ∂L/∂旋转
                densityRawParametersGrad.scale      = make_float3(0.0f);  // ∂L/∂缩放

                TFeaturesVec featuresGrad = TFeaturesVec::zero();   // ∂L/∂特征，对像素的梯度，所以只有3维，之后会逐步传播到球谐函数的系数（每个系数都是RGB三维）

                // ========== 步骤3: 计算Ray-Particle交互梯度 ==========
                if (ray.isAlive()) {
                    // 🔥 核心计算: 使用自动微分重新计算ray-particle交互
                    particles.processHitBwd<Params::PerRayParticleFeatures>(
                        // 光线几何
                        ray.origin, ray.direction,
                        
                        // 粒子参数
                        particleData.idx, particleData.densityParameters, &densityRawParametersGrad,
                        particleData.features, &featuresGrad,
                        
                        // 光线状态 (Forward值 + 梯度)
                        ray.transmittance, ray.transmittanceBackward, ray.transmittanceGradient,
                        ray.features, ray.featuresBackward, ray.featuresGradient,
                        ray.hitT, ray.hitTBackward, ray.hitTGradient);
                    
                    // 🛑 透射率阈值检查: 光线能量耗尽时终止
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                // ========== 步骤4: 梯度更新到全局内存 ==========
                // 🎨 特征梯度更新 (原子操作)
                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                }
                
                // 🔧 几何参数梯度更新 (原子操作)
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};
