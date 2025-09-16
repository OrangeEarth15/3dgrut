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
                                       const tcnn::uvec2 tileGrid) {
    
    // 静态分配，每个block处理一个virtual tile
    // 每个block对应一个virtual tile ID
    const uint32_t virtual_tile_id = blockIdx.x;
    
    // 计算总的virtual tiles数量
    const uint32_t virtual_tiles_per_original_tile = 64; // (16*16) / 4
    const uint32_t total_virtual_tiles = tileGrid.x * tileGrid.y * virtual_tiles_per_original_tile;
    
    // 边界检查
    if (virtual_tile_id >= total_virtual_tiles) return;
        
    // 将virtual tile映射回原始16x16 tile和4个pixels位置
    const uint32_t original_tile_id = virtual_tile_id / virtual_tiles_per_original_tile; // 除64得到真正的tile id
    const uint32_t virtual_tile_in_original = virtual_tile_id % virtual_tiles_per_original_tile; // 取余64的到在大tile中的小tile编号
    
    const uint32_t original_tile_x = original_tile_id % tileGrid.x;
    const uint32_t original_tile_y = original_tile_id / tileGrid.x;
    
    // 将virtual tile映射到pixels区域
    // 64个virtual tiles按8x8方式排列在16x16 tile内，每个virtual tile = 2x2 pixels
    const uint32_t virtual_tile_x = virtual_tile_in_original % 8;  // 0-7
    const uint32_t virtual_tile_y = virtual_tile_in_original / 8;  // 0-7
    
    // 每个virtual tile对应2x2的pixels区域 (width=2, height=2)
    const uint32_t base_pixel_x = virtual_tile_x * 2;  // 0,2,4,6,8,10,12,14
    const uint32_t base_pixel_y = virtual_tile_y * 2;  // 0,2,4,6,8,10,12,14
    
    // Algorithm 3 line 11: 4个warps分别处理2x2区域内的4个pixels
    const uint32_t warpId = threadIdx.x / 32;
    const uint32_t laneId = threadIdx.x % 32;
    
    // 每个block处理1个virtual tile = 4个pixels，每个warp处理1个pixel
    if (warpId < 4) { // 4 warps per block (每个warp处理1个pixel)
        // 在2x2区域内按行优先排列4个pixels
        // warp 0-3 对应 pixels: (0,0),(1,0),(0,1),(1,1)
        const uint32_t pixel_offset_x = warpId % 2;      // 0,1,0,1
        const uint32_t pixel_offset_y = warpId / 2;      // 0,0,1,1
        
        const uint32_t pixel_local_x = base_pixel_x + pixel_offset_x;
        const uint32_t pixel_local_y = base_pixel_y + pixel_offset_y;
        
        const tcnn::uvec2 pixel = {
            original_tile_x * 16 + pixel_local_x,
            original_tile_y * 16 + pixel_local_y
        };
                
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

/**
 * 🔄 反向传播渲染内核 - 从Forward结果计算所有参数梯度
 * 
 * 💡 核心流程: 加载Forward结果 → 初始化反向光线 → 遍历粒子计算梯度
 * ⚠️  重要: 使用与Forward相同的粒子排序数据，但独立的执行配置
 */
__global__ void renderBackward(
    // 基础参数
    threedgut::RenderParameters params,
    const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,      // Forward计算的tile粒子范围
    const uint32_t* __restrict__ sortedTileDataPtr,                 // Forward计算的粒子排序索引
    
    // 光线几何
    const tcnn::vec3* __restrict__ sensorRayOriginPtr,              // 光线起点
    const tcnn::vec3* __restrict__ sensorRayDirectionPtr,           // 光线方向
    tcnn::mat4x3 sensorToWorldTransform,                            // 坐标变换
    
    // Forward结果 + 损失梯度 (输入)
    const float* __restrict__ worldHitDistancePtr,                  // Forward: 击中距离
    const float* __restrict__ worldHitDistanceGradientPtr,          // ∂L/∂distance
    const tcnn::vec4* __restrict__ radianceDensityPtr,              // Forward: 最终颜色+密度
    const tcnn::vec4* __restrict__ radianceDensityGradientPtr,      // ∂L/∂color (主要输入)
    
    // 光线梯度 (暂未使用)
    tcnn::vec3* __restrict__ /*worldRayOriginGradientPtr*/,
    tcnn::vec3* __restrict__ /*worldRayDirectionGradientPtr*/,
    
    // 粒子数据 (Forward计算)
    const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,   // 粒子屏幕位置
    const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr, // 椭圆参数+透明度
    const float* __restrict__ particlesGlobalDepthPtr,              // 粒子深度
    const float* __restrict__ particlesPrecomputedFeaturesPtr,      // 粒子特征
    const uint64_t* __restrict__ parameterMemoryHandles,            // 神经网络参数
    
    // 梯度输出
    tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,     // ∂L/∂粒子位置
    tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr, // ∂L/∂椭圆参数
    float* __restrict__ particlesGlobalDepthGradPtr,                // ∂L/∂粒子深度
    float* __restrict__ particlesPrecomputedFeaturesGradPtr,        // ∂L/∂粒子特征
    const uint64_t* __restrict__ parameterGradientMemoryHandles     // ∂L/∂网络参数
) {

    // 步骤1: 从Forward结果初始化反向光线
    auto ray = initializeBackwardRay<TGUTRenderer::TRayPayloadBackward>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr,
        worldHitDistancePtr, worldHitDistanceGradientPtr,
        radianceDensityPtr, radianceDensityGradientPtr, sensorToWorldTransform);

    // 步骤2: 使用链式法则计算所有参数梯度
    TGUTBackwardRenderer::eval(
        params, ray,
        sortedTileRangeIndicesPtr, sortedTileDataPtr,
        particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr,
        particlesGlobalDepthPtr, particlesPrecomputedFeaturesPtr,
        {parameterMemoryHandles},
        particlesProjectedPositionGradPtr, particlesProjectedConicOpacityGradPtr,
        particlesGlobalDepthGradPtr, particlesPrecomputedFeaturesGradPtr,
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