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

#include <3dgut/kernels/cuda/common/cudaMath.cuh>
#include <3dgut/kernels/cuda/common/random.cuh>
#include <3dgut/renderer/renderParameters.h>

// 光线数据载荷模板结构，存储光线状态和特征
template <int FeatN>
struct RayPayload {
    static constexpr uint32_t FeatDim = FeatN; // 特征维度

    threedgut::TTimestamp timestamp; // 时间戳，用于记录光线追踪的开始时间
    tcnn::vec3 origin; // 光线起点
    tcnn::vec3 direction; // 光线方向
    tcnn::vec2 tMinMax; // 光线的有效范围 [tMin, tMax]，用于记录光线与场景的交点
    float hitT; // 当前光线击中距离，记录光线最大响应的距离
    float transmittance; // 透射率（未被吸收的光线比例），记录光线在场景中的透射率

    enum {
        Default = 0, // 默认状态 
        Valid   = 1 << 0, // 光线有效
        Alive   = 1 << 2, // 光线仍在追踪中
        // BackHit             = 1 << 3,
        // BackHitProxySurface = 1 << 4,
        // FrontHit            = 1 << 5,
    };
    uint32_t flags; // 光线状态标志
    uint32_t idx; // 光线索引（通常对应像素）
    tcnn::vec<FeatN> features; // 累积的特征向量（颜色等）

#if GAUSSIAN_ENABLE_HIT_COUNT
    uint32_t hitN;
#endif

    // 检查光线是否仍然活跃（未被终止）
    __device__ __forceinline__ bool isAlive() const {
        return flags & Alive;
    }

    // 终止光线追踪
    __device__ __forceinline__ void kill() {
        flags &= ~Alive;
    }

    // 检查光线是否有效
    __device__ __forceinline__ bool isValid() const {
        return flags & Valid;
    }

    // __device__ __forceinline__ bool isFrontHit() const {
    //     return flags & FrontHit;
    // }

    // __device__ __forceinline__ void hitFront() {
    //     flags |= FrontHit;
    // }

    // 统计光线击中次数
    __device__ __forceinline__ void countHit(uint32_t count = 1) {
#if GAUSSIAN_ENABLE_HIT_COUNT
        hitN += count;
#endif
    }
};

// 初始化光线数据，设置起点、方向和参数
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRay(const threedgut::RenderParameters& params,
                                                const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                const tcnn::mat4x3& sensorToWorldTransform) {
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;
    if ((x >= params.resolution.x) || (y >= params.resolution.y)) {
        return ray;
    }
    ray.idx           = x + params.resolution.x * y;
    ray.hitT          = 0.0f; // 初始化光线击中距离为0
    ray.transmittance = 1.0f; // 初始化透射率为1
    ray.features      = tcnn::vec<RayPayloadT::FeatDim>::zero(); // 初始化特征向量为0向量

    ray.origin    = sensorToWorldTransform * tcnn::vec4(sensorRayOriginPtr[ray.idx], 1.0f); // 将传感器坐标系下的光线起点转换为世界坐标系
    ray.direction = tcnn::mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx]; // 将传感器坐标系下的光线方向转换为世界坐标系

    // 计算光线与场景包围盒（AABB）的交点
    // tMinMax.x：光线进入包围盒的参数t
    // tMinMax.y：光线离开包围盒的参数t
    // fmaxf(ray.tMinMax.x, 0.0f)：确保起点不在相机后面
    // 只有当光线确实穿过包围盒时（tMinMax.y > tMinMax.x），才标记为有效和活跃
    ray.tMinMax   = params.objectAABB.ray_intersect(ray.origin, ray.direction);
    ray.tMinMax.x = fmaxf(ray.tMinMax.x, 0.0f);

    if (ray.tMinMax.y > ray.tMinMax.x) {
        ray.flags |= RayPayloadT::Valid | RayPayloadT::Alive;
    }

#if GAUSSIAN_ENABLE_HIT_COUNT
    ray.hitN = 0;
#endif

    return ray;
}

// 完成光线追踪，输出最终结果
template <typename TRayPayload>
__device__ __inline__ void finalizeRay(const TRayPayload& ray,
                                       const threedgut::RenderParameters& params,
                                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                       float* __restrict__ worldCountPtr,
                                       float* __restrict__ worldHitDistancePtr,
                                       tcnn::vec4* __restrict__ radianceDensityPtr,
                                       const tcnn::mat4x3& sensorToWorldTransform) {
    if (!ray.isValid()) {
        return;
    }

    radianceDensityPtr[ray.idx] = {ray.features[0], ray.features[1], ray.features[2], (1.0f - ray.transmittance)};

    worldHitDistancePtr[ray.idx] = ray.hitT;

#if GAUSSIAN_ENABLE_HIT_COUNT
    worldCountPtr[ray.idx] = (float)ray.hitN;
#endif
}
