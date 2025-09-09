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

#include <3dgut/kernels/cuda/common/rayPayload.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

// 粒子投影到瓦片的CUDA内核
__global__ void projectOnTiles(tcnn::uvec2 tileGrid,
                               uint32_t numParticles,
                               tcnn::ivec2 resolution,
                               threedgut::TSensorModel sensorModel,
                               tcnn::vec3 sensorWorldPosition,
                               tcnn::mat4x3 sensorViewMatrix,
                               threedgut::TSensorState sensorShutterState,
                               uint32_t* __restrict__ particlesTilesOffsetPtr,
                               tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                               float* __restrict__ particlesGlobalDepthPtr,
                               float* __restrict__ particlesPrecomputedFeaturesPtr,
                               int* __restrict__ particlesVisibilityCudaPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles) {

    TGUTProjector::eval(tileGrid,
                        numParticles,
                        resolution,
                        sensorModel,
                        sensorWorldPosition,
                        sensorViewMatrix,
                        sensorShutterState,
                        particlesTilesOffsetPtr,
                        particlesProjectedPositionPtr,
                        particlesProjectedConicOpacityPtr,
                        particlesProjectedExtentPtr,
                        particlesGlobalDepthPtr,
                        particlesPrecomputedFeaturesPtr,
                        particlesVisibilityCudaPtr,
                        {parameterMemoryHandles});
}

// 展开瓦片投影的CUDA内核
__global__ void expandTileProjections(tcnn::uvec2 tileGrid,
                                      uint32_t numParticles,
                                      tcnn::ivec2 resolution,
                                      threedgut::TSensorModel sensorModel,
                                      threedgut::TSensorState sensorState,
                                      const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                      const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                      const float* __restrict__ particlesGlobalDepthPtr,
                                      const uint64_t* __restrict__ parameterMemoryHandles,
                                      uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                      uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

    TGUTProjector::expand(tileGrid,
                          numParticles,
                          resolution,
                          sensorModel,
                          sensorState,
                          particlesTilesOffsetPtr,
                          particlesProjectedPositionPtr,
                          particlesProjectedConicOpacityPtr,
                          particlesProjectedExtentPtr,
                          particlesGlobalDepthPtr,
                          {parameterMemoryHandles},
                          unsortedTileDepthKeysPtr,
                          unsortedTileParticleIdxPtr);
}

// 主渲染CUDA内核
// rm -rf /home/sarawang/.cache/torch_extensions/py311_cu128/lib3dgut_cc/
__global__ void renderDynamic(threedgut::RenderParameters params,
                        const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                        const uint32_t* __restrict__ sortedTileDataPtr,
                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                        tcnn::mat4x3 sensorToWorldTransform,
                        float* __restrict__ worldHitCountPtr,
                        float* __restrict__ worldHitDistancePtr,
                        tcnn::vec4* __restrict__ radianceDensityPtr,
                        const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                        const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                        const float* __restrict__ particlesGlobalDepthPtr,
                        const float* __restrict__ particlesPrecomputedFeaturesPtr,
                        const uint64_t* __restrict__ parameterMemoryHandles,
                        uint32_t* __restrict__ nextTileCounterPtr,
                        const tcnn::uvec2 tileGrid
                        ) {
        __shared__ uint32_t shared_tile_id;

        // 第一次获取tile ID
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            shared_tile_id = atomicAdd(nextTileCounterPtr, 1);
        }
        __syncthreads();

        // 循环处理tiles直到没有更多工作
        while (shared_tile_id < tileGrid.x * tileGrid.y) {
            const tcnn::uvec2 tile = {shared_tile_id % tileGrid.x, shared_tile_id / tileGrid.x};
            const tcnn::uvec2 pixel = {
                tile.x * threedgut::GUTParameters::Tiling::BlockX + threadIdx.x,
                tile.y * threedgut::GUTParameters::Tiling::BlockY + threadIdx.y
            };

        // // 检查pixel是否在有效范围内，但不要return，而是continue到下一个tile
        // if (pixel.x >= params.resolution.x || pixel.y >= params.resolution.y) {
        //     continue;
        // }

        auto ray = initializeRayPerPixel<TGUTRenderer::TRayPayload>(
        params, pixel, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);

        TGUTRenderer::evalBalanced(params,
                ray,
                sortedTileRangeIndicesPtr,
                sortedTileDataPtr,
                particlesProjectedPositionPtr,
                particlesProjectedConicOpacityPtr,
                particlesGlobalDepthPtr,
                particlesPrecomputedFeaturesPtr,
                tile,
                tileGrid,
                {parameterMemoryHandles});

        // TGUTModel::eval(params, ray, {parameterMemoryHandles});

        // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
        finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr, worldHitDistancePtr, radianceDensityPtr, sensorToWorldTransform);

        // 获取下一个tile ID
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            shared_tile_id = atomicAdd(nextTileCounterPtr, 1);
        }
        __syncthreads();
        }  // 结束while循环

}

// Fine-grained负载均衡渲染内核 (基于论文Algorithm 3)
__global__ void renderFineGrainBalanced(threedgut::RenderParameters params,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileDataPtr,
                                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                       const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                       tcnn::mat4x3 sensorToWorldTransform,
                                       float* __restrict__ worldHitCountPtr,
                                       float* __restrict__ worldHitDistancePtr,
                                       tcnn::vec4* __restrict__ radianceDensityPtr,
                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       const float* __restrict__ particlesGlobalDepthPtr,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       const uint64_t* __restrict__ parameterMemoryHandles,
                                       uint32_t* __restrict__ nextVirtualTileCounterPtr,
                                       const tcnn::uvec2 tileGrid) {
    
    // 实现论文Algorithm 3的核心逻辑
    __shared__ uint32_t shared_virtual_tile_id;
    
    // 计算总的virtual tiles数量 (Algorithm 3 line 23)  
    // 每个原始16x16 tile产生32个virtual tiles（每个8pixels，1个warp处理1个pixel）
    const uint32_t virtual_tiles_per_original_tile = 32; // (16*16) / 8
    const uint32_t total_virtual_tiles = tileGrid.x * tileGrid.y * virtual_tiles_per_original_tile;
    
    // Algorithm 3 line 4-6: 原子获取virtual tile ID
    if (threadIdx.x == 0) {
        shared_virtual_tile_id = atomicAdd(nextVirtualTileCounterPtr, 1);
    }
    __syncthreads(); // Algorithm 3 line 3, 7
    
    while (shared_virtual_tile_id < total_virtual_tiles) { // Algorithm 3 line 2, 8-10
        
        // 将virtual tile映射回原始16x16 tile和8个pixels位置
        const uint32_t original_tile_id = shared_virtual_tile_id / virtual_tiles_per_original_tile;
        const uint32_t virtual_tile_in_original = shared_virtual_tile_id % virtual_tiles_per_original_tile;
        
        const uint32_t original_tile_x = original_tile_id % tileGrid.x;
        const uint32_t original_tile_y = original_tile_id / tileGrid.x;
        
        // 将virtual tile映射到pixels区域
        // 32个virtual tiles按8x4方式排列在16x16 tile内，每个virtual tile = 2x4 pixels
        const uint32_t virtual_tile_x = virtual_tile_in_original % 8;  // 0-7
        const uint32_t virtual_tile_y = virtual_tile_in_original / 8;  // 0-3
        
        // 每个virtual tile对应2x4的pixels区域 (width=2, height=4)
        const uint32_t base_pixel_x = virtual_tile_x * 2;  // 0,2,4,6,8,10,12,14
        const uint32_t base_pixel_y = virtual_tile_y * 4;  // 0,4,8,12
        
        // Algorithm 3 line 11: 8个warps分别处理2x4区域内的8个pixels
        const uint32_t warpId = threadIdx.x / 32;
        const uint32_t laneId = threadIdx.x % 32;
        
        // 每个block处理1个virtual tile = 8个pixels，每个warp处理1个pixel
        if (warpId < 8) { // 8 warps per block (保持与原始BlockSize一致)
            // 在2x4区域内按行优先排列8个pixels
            // warp 0-7 对应 pixels: (0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)
            const uint32_t pixel_offset_x = warpId % 2;      // 0,1,0,1,0,1,0,1
            const uint32_t pixel_offset_y = warpId / 2;      // 0,0,1,1,2,2,3,3
            
            const uint32_t pixel_local_x = base_pixel_x + pixel_offset_x;
            const uint32_t pixel_local_y = base_pixel_y + pixel_offset_y;
            
            const tcnn::uvec2 pixel = {
                original_tile_x * 16 + pixel_local_x,
                original_tile_y * 16 + pixel_local_y
            };
                
                if (pixel.x < params.resolution.x && pixel.y < params.resolution.y) {
                    
                    // Algorithm 3 line 12: Initialize local variables
                    auto ray = initializeRayPerPixel<TGUTRenderer::TRayPayload>(
                        params, pixel, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);
                    
                    // Algorithm 3 line 13-17: Warp-level parallel processing
                    // 使用原始tile的粒子数据（16x16 tile的数据）
                    const tcnn::uvec2 original_tile = {original_tile_x, original_tile_y};
                    
                    TGUTRenderer::evalFineGrainedWarp(params,
                                                     ray,
                                                     sortedTileRangeIndicesPtr,
                                                     sortedTileDataPtr,
                                                     particlesProjectedPositionPtr,
                                                     particlesProjectedConicOpacityPtr,
                                                     particlesGlobalDepthPtr,
                                                     particlesPrecomputedFeaturesPtr,
                                                     original_tile,
                                                     tileGrid,
                                                     laneId, // lane ID for warp-level processing
                                                     {parameterMemoryHandles});
                    
                    // Algorithm 3 line 18: Write outputs
                    finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr, 
                              worldHitDistancePtr, radianceDensityPtr, sensorToWorldTransform);
                }
        }
        
        // 获取下一个virtual tile
        if (threadIdx.x == 0) {
            shared_virtual_tile_id = atomicAdd(nextVirtualTileCounterPtr, 1);
        }
        __syncthreads();
    }
}

__global__ void render(threedgut::RenderParameters params,
                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                       const uint32_t* __restrict__ sortedTileDataPtr,
                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                       const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                       tcnn::mat4x3 sensorToWorldTransform,
                       float* __restrict__ worldHitCountPtr,
                       float* __restrict__ worldHitDistancePtr,
                       tcnn::vec4* __restrict__ radianceDensityPtr,
                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                       const float* __restrict__ particlesGlobalDepthPtr,
                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                       const uint64_t* __restrict__ parameterMemoryHandles) {

    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);

    TGUTRenderer::eval(params,
                       ray,
                       sortedTileRangeIndicesPtr,
                       sortedTileDataPtr,
                       particlesProjectedPositionPtr,
                       particlesProjectedConicOpacityPtr,
                       particlesGlobalDepthPtr,
                       particlesPrecomputedFeaturesPtr,
                       {parameterMemoryHandles});

    // TGUTModel::eval(params, ray, {parameterMemoryHandles});

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
    finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr, worldHitDistancePtr, radianceDensityPtr, sensorToWorldTransform);
}

// 反向渲染CUDA内核
__global__ void renderBackward(threedgut::RenderParameters params,
                               const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                               const uint32_t* __restrict__ sortedTileDataPtr,
                               const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                               const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                               tcnn::mat4x3 sensorToWorldTransform,
                               const float* __restrict__ worldHitDistancePtr,
                               const float* __restrict__ worldHitDistanceGradientPtr,
                               const tcnn::vec4* __restrict__ radianceDensityPtr,
                               const tcnn::vec4* __restrict__ radianceDensityGradientPtr,
                               tcnn::vec3* __restrict__ /*worldRayOriginGradientPtr*/,
                               tcnn::vec3* __restrict__ /*worldRayDirectionGradientPtr*/,
                               const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               const float* __restrict__ particlesGlobalDepthPtr,
                               const float* __restrict__ particlesPrecomputedFeaturesPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles,
                               tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                               float* __restrict__ particlesGlobalDepthGradPtr,
                               float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                               const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    auto ray = initializeBackwardRay<TGUTRenderer::TRayPayloadBackward>(params,
                                                                        sensorRayOriginPtr,
                                                                        sensorRayDirectionPtr,
                                                                        worldHitDistancePtr,
                                                                        worldHitDistanceGradientPtr,
                                                                        radianceDensityPtr,
                                                                        radianceDensityGradientPtr,
                                                                        sensorToWorldTransform);

    // TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles});

    TGUTBackwardRenderer::eval(params,
                               ray,
                               sortedTileRangeIndicesPtr,
                               sortedTileDataPtr,
                               particlesProjectedPositionPtr,
                               particlesProjectedConicOpacityPtr,
                               particlesGlobalDepthPtr,
                               particlesPrecomputedFeaturesPtr,
                               {parameterMemoryHandles},
                                particlesProjectedPositionGradPtr,
                                particlesProjectedConicOpacityGradPtr,
                                 particlesGlobalDepthGradPtr,
                                particlesPrecomputedFeaturesGradPtr,
                              {parameterGradientMemoryHandles});
}

// 反向投影CUDA内核
__global__ void projectBackward(tcnn::uvec2 tileGrid,
                                uint32_t numParticles,
                                tcnn::ivec2 resolution,
                                threedgut::TSensorModel sensorModel,
                                tcnn::vec3 sensorWorldPosition,
                                tcnn::mat4x3 sensorViewMatrix,
                                const uint32_t* __restrict__ particlesTilesCountPtr,
                                const uint64_t* __restrict__ parameterMemoryHandles,
                                const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                const float* __restrict__ particlesGlobalDepthGradPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    TGUTProjector::evalBackward(tileGrid,
                                numParticles,
                                resolution,
                                sensorModel,
                                sensorWorldPosition,
                                sensorViewMatrix,
                                particlesTilesCountPtr,
                                {parameterMemoryHandles},
                                particlesProjectedPositionGradPtr,
                                particlesProjectedConicOpacityGradPtr,
                                particlesGlobalDepthGradPtr,
                                particlesPrecomputedFeaturesPtr,
                                particlesPrecomputedFeaturesGradPtr,
                                {parameterGradientMemoryHandles});
}
