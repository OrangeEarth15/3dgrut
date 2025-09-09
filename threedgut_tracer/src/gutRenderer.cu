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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <3dgut/threedgut.cuh>

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/sensors/sensors.h>

#include <tiny-cuda-nn/common_host.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <limits>

using namespace tcnn;

namespace {

using namespace threedgut;

constexpr int featuresDim() {
    return model_ExternalParams::FeaturesDim; // 外部算法参数，3
}

// identify tiles start/end indices in the sorted tile/depth keys buffer
// 计算排序后tile的范围索引（guassian粒子范围的开始和结束）
__global__ void computeSortedTileRangeIndices(
    int numKeys,
    const uint64_t* __restrict__ sortedTileDepthKeys,
    uvec2* __restrict__ tileRangeIndices) {

    const int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (keyIdx >= numKeys) {
        return;
    }

    const uint32_t tileIdx = sortedTileDepthKeys[keyIdx] >> 32;
    const bool validTile   = tileIdx != GUTParameters::Tiling::InvalidTileIdx; // 初始值为-1U
    if (keyIdx == 0) {
        if (validTile) {
            tileRangeIndices[tileIdx].x = keyIdx;
        }
    } else {
        const uint32_t prevKeyTileIdx = sortedTileDepthKeys[keyIdx - 1] >> 32;
        if (prevKeyTileIdx != tileIdx) {
            if (prevKeyTileIdx != GUTParameters::Tiling::InvalidTileIdx) {
                tileRangeIndices[prevKeyTileIdx].y = keyIdx;
            }
            if (validTile) {
                tileRangeIndices[tileIdx].x = keyIdx;
            }
        }
    }
    if (validTile && (keyIdx == numKeys - 1)) {
        tileRangeIndices[tileIdx].y = numKeys;
    }
}

// TODO : review this
// 计算n的最高有效位（MSB，Most Significant Bit），即n的二进制表示中最高位1的位置
inline uint32_t higherMsb(uint32_t n) {
    uint32_t msb  = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (n >> msb) {
        msb++;
    }
    return msb;
}

} // namespace

// TODO : per-stream n context cache
// 渲染前向上下文，用于存储渲染过程中需要用到的缓冲区
struct GUTRenderer::GutRenderForwardContext {
    cudaStream_t cudaStream; // 声明流变量，用于管理异步GPU操作的执行顺序

    GutRenderForwardContext(cudaStream_t iCudaStream)
        : cudaStream(iCudaStream) {
    }

    ~GutRenderForwardContext() {
        // 句柄是一个抽象的标识符，用来引用系统资源而不是直接使用资源的实际地址或指针。它就像是一张"提货单"或"钥匙"，让你能够访问某个资源，但不直接暴露资源的内部实现细节。
        const uint64_t processQueueHandle = reinterpret_cast<uint64_t>(cudaStream); // 将cudaStream转换为uint64_t类型
        Logger logger;
        // 清除缓冲区，释放GPU内存
        unsortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileRangeIndices.clear(processQueueHandle, logger);
        unsortedTileParticleIdx.clear(processQueueHandle, logger);
        sortedTileParticleIdx.clear(processQueueHandle, logger);
        sortingWorkingBuffer.clear(processQueueHandle, logger);
        particlesTilesCount.clear(processQueueHandle, logger);
        particlesTilesOffset.clear(processQueueHandle, logger);
        particlesProjectedPosition.clear(processQueueHandle, logger);
        particlesProjectedConicOpacity.clear(processQueueHandle, logger);
        particlesProjectedExtent.clear(processQueueHandle, logger);
        particlesGlobalDepth.clear(processQueueHandle, logger);
        particlesPrecomputedFeatures.clear(processQueueHandle, logger);
        particlesProjectedPositionGradient.clear(processQueueHandle, logger);
        particlesProjectedConicOpacityGradient.clear(processQueueHandle, logger);
        particlesGlobalDepthGradient.clear(processQueueHandle, logger);
        particlesPrecomputedFeaturesGradient.clear(processQueueHandle, logger);
        scanningWorkingBuffer.clear(processQueueHandle, logger);
#if DYNAMIC_LOAD_BALANCING || FINE_GRAINED_LOAD_BALANCING
        nextTileCounter.clear(processQueueHandle, logger);
#endif
    }

    CudaBuffer unsortedTileDepthKeys; // 未排序的tile深度键
    CudaBuffer sortedTileDepthKeys; // 排序后的tile深度键
    CudaBuffer sortedTileRangeIndices; // 排序后的tile范围索引
    CudaBuffer unsortedTileParticleIdx; // 未排序的粒子索引
    CudaBuffer sortedTileParticleIdx; // 排序后的粒子索引
    CudaBuffer sortingWorkingBuffer; // 排序工作缓冲区

    Status updateTileSortingBuffers(const uvec2& tileGrid, int numKeys, cudaStream_t stream, const Logger& logger) {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(stream);
        const bool uptodate        = unsortedTileDepthKeys.size() >= sizeof(uint64_t) * numKeys;
        if (!uptodate) {
            CHECK_STATUS_RETURN(unsortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(unsortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            size_t sortingWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceRadixSort::SortPairs(
                    nullptr,
                    sortingWorkingBufferSize,
                    static_cast<const uint64_t*>(unsortedTileDepthKeys.data()),
                    static_cast<uint64_t*>(sortedTileDepthKeys.data()),
                    static_cast<const uint32_t*>(unsortedTileParticleIdx.data()),
                    static_cast<uint32_t*>(sortedTileParticleIdx.data()),
                    numKeys,
                    0, 32 + higherMsb(tileGrid.x * tileGrid.y),
                    stream),
                logger);
            CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, queueHandle, logger));
        }
        if (numKeys) {
            CHECK_STATUS_RETURN(sortedTileRangeIndices.resize(sizeof(uvec2) * tileGrid.x * tileGrid.y, queueHandle, logger));
            CUDA_CHECK_RETURN(cudaMemsetAsync(sortedTileRangeIndices.data(), 0, tileGrid.x * tileGrid.y * sizeof(uvec2), stream), logger);
        }
        return Status();
    }

    CudaBuffer particlesTilesCount;                            ///< number of intersected tiles per particles uint32_t [Nx1]
    CudaBuffer particlesTilesOffset;                           ///< cumulative sum of particle tiles count uint32_t [Nx1]
    CudaBuffer particlesProjectedPosition;                     ///< projected particles center Nx2
    CudaBuffer particlesProjectedConicOpacity;                 ///< projected particles conic and opacity Nx4
    CudaBuffer particlesProjectedExtent;                       ///< projected particles extent Nx2
    CudaBuffer particlesGlobalDepth;                           ///< particles global depth
    CudaBuffer particlesPrecomputedFeatures;                   ///< precomputed particle features float [NxFeaturesDim]
    mutable CudaBuffer particlesProjectedPositionGradient;     ///< projected particles center Nx2
    mutable CudaBuffer particlesProjectedConicOpacityGradient; ///< projected particles conic and opacity Nx4
    mutable CudaBuffer particlesGlobalDepthGradient;           ///< particles global depth
    mutable CudaBuffer particlesPrecomputedFeaturesGradient;   ///< precomputed particle features float [NxFeaturesDim]
    CudaBuffer scanningWorkingBuffer;                          ///< working buffer to compute the cumulative sum of particles/tiles intersections number
    
#if DYNAMIC_LOAD_BALANCING || FINE_GRAINED_LOAD_BALANCING
    CudaBuffer nextTileCounter;                                    ///< atomic counter for dynamic load balancing tile assignment
#endif

    inline Status updateParticlesWorkingBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) {
        const bool uptodate = particlesTilesCount.size() >= numParticles * sizeof(uint32_t);
        if (!uptodate) {
            const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);
            CHECK_STATUS_RETURN(particlesTilesCount.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesTilesOffset.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            size_t scanningWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceScan::InclusiveSum(
                    nullptr,
                    scanningWorkingBufferSize,
                    static_cast<const uint32_t*>(particlesTilesCount.data()),
                    static_cast<uint32_t*>(particlesTilesOffset.data()),
                    numParticles,
                    cudaStream),
                logger);
            CHECK_STATUS_RETURN(scanningWorkingBuffer.resize(scanningWorkingBufferSize, queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedPosition.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedConicOpacity.resize(numParticles * sizeof(vec4), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedExtent.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesGlobalDepth.resize(numParticles * sizeof(float), queueHandle, logger));
        }

        return Status();
    }

    inline Status updateParticlesProjectionGradientBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) const {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);

        CHECK_STATUS_RETURN(particlesProjectedPositionGradient.enlarge(numParticles * sizeof(vec2), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedPositionGradient.data(), 0, numParticles * sizeof(vec2), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesProjectedConicOpacityGradient.resize(numParticles * sizeof(vec4), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedConicOpacityGradient.data(), 0, numParticles * sizeof(vec4), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesGlobalDepthGradient.resize(numParticles * sizeof(float), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesGlobalDepthGradient.data(), 0, numParticles * sizeof(float), cudaStream), logger);

        return Status();
    }

    inline Status
    updateParticlesFeaturesBuffer(int featuresSize, cudaStream_t cudaStream, const Logger& logger) {
        CHECK_STATUS_RETURN(particlesPrecomputedFeatures.enlarge(featuresSize * sizeof(float), reinterpret_cast<uint64_t>(cudaStream), logger));
        return Status();
    }

    inline Status updateParticlesFeaturesGradientBuffer(uint32_t featuresSize, cudaStream_t cudaStream, const Logger& logger) const {
        const size_t newSize = featuresSize * sizeof(float);
        CHECK_STATUS_RETURN(particlesPrecomputedFeaturesGradient.enlarge(newSize, reinterpret_cast<uint64_t>(cudaStream), logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesPrecomputedFeaturesGradient.data(), 0, newSize, cudaStream), logger);
        return Status();
    }
};

threedgut::GUTRenderer::GUTRenderer(const nlohmann::json& config, const Logger& logger)
    : m_logger(logger) {
}

threedgut::GUTRenderer::~GUTRenderer() {
}

threedgut::Status threedgut::GUTRenderer::renderForward(const RenderParameters& params,
                                                        const vec3* sensorRayOriginCudaPtr,
                                                        const vec3* sensorRayDirectionCudaPtr,
                                                        float* worldHitCountCudaPtr,
                                                        float* worldHitDistanceCudaPtr,
                                                        vec4* radianceDensityCudaPtr,
                                                        int* particlesVisibilityCudaPtr,
                                                        Parameters& parameters,
                                                        int cudaDeviceIndex,
                                                        cudaStream_t cudaStream) {

    if (!m_forwardContext) {
        m_forwardContext = std::make_unique<GutRenderForwardContext>(cudaStream);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    const uint32_t numParticles = parameters.values.numParticles;

    // NOTE: to properly support rolling shutter camera, we need to interpolate sensor pose in the kernel
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    CHECK_STATUS_RETURN(m_forwardContext->updateParticlesWorkingBuffers(numParticles, cudaStream, m_logger));
    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(m_forwardContext->updateParticlesFeaturesBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

    {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::project"};
        ::projectOnTiles<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose) /** tcnn::mat4(params.objectToWorldTransform)*/,
            params.sensorState,
            (uint32_t*)m_forwardContext->particlesTilesCount.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (float*)m_forwardContext->particlesGlobalDepth.data(),
            (float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            particlesVisibilityCudaPtr,
            parameters.m_dptrParametersBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    deviceLaunchesLogger.push("render::prepare-expand");

    // inplace cumulative sum over list of number of intersected tiles per particles
    size_t scanningWorkingBufferSize = m_forwardContext->scanningWorkingBuffer.size();
    // TODO : check if using not inplace version has perf benefits
    CUDA_CHECK_RETURN(
        cub::DeviceScan::InclusiveSum(
            m_forwardContext->scanningWorkingBuffer.data(),
            scanningWorkingBufferSize,
            static_cast<const uint32_t*>(m_forwardContext->particlesTilesCount.data()),
            static_cast<uint32_t*>(m_forwardContext->particlesTilesOffset.data()),
            numParticles,
            cudaStream),
        m_logger);

    // fetch total number of particle/tile intersections to launch and resize the sorting buffers
    uint32_t numParticleTileIntersections;
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(&numParticleTileIntersections,
                        static_cast<uint32_t*>(m_forwardContext->particlesTilesOffset.data()) + numParticles - 1,
                        sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        cudaStream),
        m_logger);
    cudaStreamSynchronize(cudaStream);

    if (numParticleTileIntersections == 0) {
        return Status();
    }

    // sorting buffers allocation
    CHECK_STATUS_RETURN(
        m_forwardContext->updateTileSortingBuffers(tileGrid, numParticleTileIntersections, cudaStream, m_logger));

    deviceLaunchesLogger.pop("render::prepare-expand");

    {
        const auto expandProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::expand"};
        // ::表示全局作用域操作符，::expandTileProjections 指的是全局命名空间中的函数
        // 不是某个类的成员函数，而是一个自由函数
        // 在CUDA中，这通常是一个global kernel函数
        ::expandTileProjections<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            params.sensorState,
            (const uint32_t*)m_forwardContext->particlesTilesOffset.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            parameters.m_dptrParametersBuffer,
            (uint64_t*)m_forwardContext->unsortedTileDepthKeys.data(),
            (uint32_t*)m_forwardContext->unsortedTileParticleIdx.data());
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    deviceLaunchesLogger.push("render::sort");

    // Sort complete list of (duplicated) Gaussian indices by keys
    size_t sortingWorkingBufferSize = m_forwardContext->sortingWorkingBuffer.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
                          m_forwardContext->sortingWorkingBuffer.data(),
                          sortingWorkingBufferSize,
                          static_cast<const uint64_t*>(m_forwardContext->unsortedTileDepthKeys.data()),
                          static_cast<uint64_t*>(m_forwardContext->sortedTileDepthKeys.data()),
                          static_cast<const uint32_t*>(m_forwardContext->unsortedTileParticleIdx.data()),
                          static_cast<uint32_t*>(m_forwardContext->sortedTileParticleIdx.data()),
                          numParticleTileIntersections,
                          0, 32 + higherMsb(tileGrid.x * tileGrid.y), cudaStream),
                      m_logger);

    // Compute the tile range indices in the sorted keys
    // #if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
    // template <typename K, typename T, typename ... Types>
    // inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
    //     if (n_elements <= 0) {
    //         return;
    //     }
    //     kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
    // }
    linear_kernel(
        computeSortedTileRangeIndices, /*shmem=*/0, cudaStream,
        numParticleTileIntersections,
        static_cast<const uint64_t*>(m_forwardContext->sortedTileDepthKeys.data()),
        static_cast<uvec2*>(m_forwardContext->sortedTileRangeIndices.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);

    deviceLaunchesLogger.pop("render::sort");

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::render"};
        
#if DYNAMIC_LOAD_BALANCING || FINE_GRAINED_LOAD_BALANCING
        // 初始化动态负载均衡的计数器
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);
        CHECK_STATUS_RETURN(m_forwardContext->nextTileCounter.resize(sizeof(uint32_t), queueHandle, m_logger));
        
        // 添加调试信息
        // LOG_INFO(m_logger, "Dynamic load balancing: nextTileCounter allocated at %p, size %zu", 
        //          m_forwardContext->nextTileCounter.data(), m_forwardContext->nextTileCounter.size());
        
        uint32_t initialValue = 0;
        CUDA_CHECK_RETURN(cudaMemcpyAsync(m_forwardContext->nextTileCounter.data(), &initialValue, sizeof(uint32_t), cudaMemcpyHostToDevice, cudaStream), m_logger);
        
        // 同步确保初始化完成
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
        // LOG_INFO(m_logger, "nextTileCounter initialized successfully");
#endif
        
        // 简单直接的CUDA事件计时
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, cudaStream);
        
        // 根据配置选择不同的负载均衡策略
#if FINE_GRAINED_LOAD_BALANCING
        // Fine-grained load balancing enabled
        // Algorithm 3 line 21-27: Fine-grained launch configuration
        
        // 计算virtual tiles总数 (每个原始tile产生32个virtual tiles)
        const uint32_t virtual_tiles_per_original_tile = 32; // (16*16) / 8
        const uint32_t virtual_tiles_total = tileGrid.x * tileGrid.y * virtual_tiles_per_original_tile;
        
        // 计算最大硬件资源 (Algorithm 3 line 25)
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
        const uint32_t max_hw_resource = smCount * 16; // 16 blocks per SM
        
        // Algorithm 3 line 26: Launch with maximum hardware resource
        const uint32_t numBlocks = min(max_hw_resource, virtual_tiles_total);
        
        LOG_INFO(m_logger, "Fine-grained load balancing: virtualTiles=%u, numBlocks=%u, threadsPerBlock=256", 
                    virtual_tiles_total, numBlocks);
        
        ::renderFineGrainBalanced<<<numBlocks, 256, 0, cudaStream>>>( // 256 threads = 8 warps * 32 threads
            params,
            (const tcnn::uvec2*)m_forwardContext->sortedTileRangeIndices.data(),
            (const uint32_t*)m_forwardContext->sortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            worldHitCountCudaPtr,
            worldHitDistanceCudaPtr,
            radianceDensityCudaPtr,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer,
            (uint32_t*)m_forwardContext->nextTileCounter.data(),
            tcnn::uvec2{tileGrid.x, tileGrid.y}
        );
            
#else
        // Fine-grained load balancing disabled, use dynamic or static
#if DYNAMIC_LOAD_BALANCING
            // 动态负载均衡：优化为完整waves避免tail effect (H200: 660 blocks/wave)
            const uint32_t numBlocksX = min(tileGrid.x, 66u);  // 66×60=3960 blocks = 6×660 (6完整waves)
            const uint32_t numBlocksY = min(tileGrid.y, 60u);
            
            // 批量动态负载均衡: tileGrid(195, 130), blocks(66, 60)
            // LOG_INFO(m_logger, "Dynamic load balancing launch: tileGrid(%u, %u), blocks(%u, %u), threads(%u, %u)", 
            //          tileGrid.x, tileGrid.y, numBlocksX, numBlocksY, 
            //          GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY);
            // LOG_INFO(m_logger, "nextTileCounter pointer: %p", (uint32_t*)m_forwardContext->nextTileCounter.data());
            
            ::renderDynamic<<<dim3{numBlocksX, numBlocksY, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
#else
            ::render<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
#endif
                params, // threedgut::RenderParameters params
                (const tcnn::uvec2*)m_forwardContext->sortedTileRangeIndices.data(),
                (const uint32_t*)m_forwardContext->sortedTileParticleIdx.data(),
                (const tcnn::vec3*)sensorRayOriginCudaPtr,
                (const tcnn::vec3*)sensorRayDirectionCudaPtr,
                sensorPoseToMat(sensorPoseInv),
                worldHitCountCudaPtr,
                worldHitDistanceCudaPtr,
                radianceDensityCudaPtr,
                (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
                (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
                (const float*)m_forwardContext->particlesGlobalDepth.data(),
                (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
                parameters.m_dptrParametersBuffer
#if DYNAMIC_LOAD_BALANCING
                ,
                (uint32_t*)m_forwardContext->nextTileCounter.data(),
                tcnn::uvec2{tileGrid.x, tileGrid.y}
#endif
            );
#endif
        
        cudaEventRecord(stopEvent, cudaStream);
        cudaEventSynchronize(stopEvent);
        
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        LOG_INFO(m_logger, "render kernel took %.3f ms", elapsedTime);
        
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }
    return Status();
}

threedgut::Status threedgut::GUTRenderer::renderBackward(const RenderParameters& params,
                                                         const vec3* sensorRayOriginCudaPtr,
                                                         const vec3* sensorRayDirectionCudaPtr,
                                                         const float* worldHitDistanceCudaPtr,         //
                                                         const float* worldHitDistanceGradientCudaPtr, // TODO: not implemented yet
                                                         const vec4* radianceDensityCudaPtr,           //
                                                         const vec4* radianceDensityGradientCudaPtr,   // TODO: not implemented yet
                                                         vec3* worldRayOriginGradientCudaPtr,          // TODO: not implemented yet
                                                         vec3* worldRayDirectionGradientCudaPtr,       // TODO: not implemented yet
                                                         Parameters& parameters,
                                                         int cudaDeviceIndex,
                                                         cudaStream_t cudaStream) {

    if (!m_forwardContext || (m_forwardContext->cudaStream != cudaStream)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] cannot render backward, invalid forward context on device %d, cudaStream = %p.",
                     cudaDeviceIndex, cudaStream);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render-backward");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    const uint32_t numParticles = parameters.values.numParticles;

    // NOTE: to properly support rolling shutter camera, we need to interpolate sensor pose in the kernel
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    if (numParticles == 0) {
        LOG_ERROR(m_logger, "[GUTRenderer] number of particles is 0, cannot render backward.");
    }

    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesFeaturesGradientBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

    if (/*m_settings.renderMode == Settings::Splat*/ TGUTProjectorParams::BackwardProjection) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesProjectionGradientBuffers(numParticles, cudaStream, m_logger));
    }

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::render"};
        ::renderBackward<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params,
            (const tcnn::uvec2*)m_forwardContext->sortedTileRangeIndices.data(),
            (const uint32_t*)m_forwardContext->sortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const float*)worldHitDistanceCudaPtr,             //
            (const float*)worldHitDistanceGradientCudaPtr,     // TODO: not implemented yet
            (const tcnn::vec4*)radianceDensityCudaPtr,         //
            (const tcnn::vec4*)radianceDensityGradientCudaPtr, // TODO: not implemented yet
            (tcnn::vec3*)worldRayOriginGradientCudaPtr,        // TODO: not implemented yet
            (tcnn::vec3*)worldRayDirectionGradientCudaPtr,     // TODO: not implemented yet
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer,
            (tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::project"};
        ::projectBackward<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose),
            (const uint32_t*)m_forwardContext->particlesTilesCount.data(),
            parameters.m_dptrParametersBuffer,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (const float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }
    return Status();
}
