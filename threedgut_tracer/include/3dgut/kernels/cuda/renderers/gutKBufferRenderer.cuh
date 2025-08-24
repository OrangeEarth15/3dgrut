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

    template <typename TRayPayload> // 处理单个击中粒子，计算颜色和透明度
    static inline __device__ void processHitParticle(
        TRayPayload& ray, // 光线载荷（前向或后向）
        const HitParticle& hitParticle, // 击中粒子信息
        const Particles& particles, // 粒子系统接口
        const TFeaturesVec* __restrict__ particleFeatures, // 预计算特征数组
        TFeaturesVec* __restrict__ particleFeaturesGradient) { // 特征梯度数组
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
            // 这计算当前粒子对最终像素的贡献权重
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha, // 粒子不透明度
                                              ray.transmittance, // 当前光线透射率
                                              hitParticle.hitT, // 击中距离
                                              ray.hitT); // 光线总行进距离
            
            // 将粒子的特征（颜色等）按权重累积到光线上
            particles.featureIntegrateFwd(hitWeight, // 击中权重
                                          Params::PerRayParticleFeatures ? particles.featuresFromBuffer(hitParticle.idx, ray.direction) : tcnn::max(particleFeatures[hitParticle.idx], 0.f), // 粒子特征
                                          ray.features); // 光线累积特征

            if (hitWeight > 0.0f) ray.countHit(); // 统计有效击中次数，用于渲染质量分析
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            // 后续粒子对最终颜色贡献可忽略
            ray.kill(); // 提前终止优化 - 当透射率过低意味着光线被完全阻挡
        }
    }

    template <typename TRay>
    // 主渲染函数
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                       const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                       const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {}) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[tileIdx];
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
        } else {
            // 使用K缓冲
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
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
