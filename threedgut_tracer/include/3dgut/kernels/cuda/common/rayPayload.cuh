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

// 通用光线初始化内部函数
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRayCommon(const threedgut::RenderParameters& params,
                                                      const tcnn::uvec2& pixel,
                                                      const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                      const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                      const tcnn::mat4x3& sensorToWorldTransform) {
    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;
    
    // 边界检查：确保像素在有效范围内
    if ((pixel.x >= params.resolution.x) || (pixel.y >= params.resolution.y)) {
        return ray;
    }
    
    // 计算线性索引并初始化光线基本属性
    ray.idx           = pixel.x + params.resolution.x * pixel.y;
    ray.hitT          = 0.0f; // 初始化光线击中距离为0
    ray.transmittance = 1.0f; // 初始化透射率为1
    ray.features      = tcnn::vec<RayPayloadT::FeatDim>::zero(); // 初始化特征向量为0向量

    // 坐标变换：传感器空间 -> 世界空间
    ray.origin    = sensorToWorldTransform * tcnn::vec4(sensorRayOriginPtr[ray.idx], 1.0f);
    ray.direction = tcnn::mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx];

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

// 基于线程索引的光线初始化（标准渲染模式）
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRay(const threedgut::RenderParameters& params,
                                                const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                const tcnn::mat4x3& sensorToWorldTransform) {
    // 从当前线程的2D索引计算像素坐标
    const tcnn::uvec2 pixel = {
        threadIdx.x + blockDim.x * blockIdx.x,
        threadIdx.y + blockDim.y * blockIdx.y
    };
    
    return initializeRayCommon<RayPayloadT>(params, pixel, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);
}

// 基于给定像素坐标的光线初始化（动态负载均衡模式）
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRayPerPixel(const threedgut::RenderParameters& params,
                                                        const tcnn::uvec2& pixel,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform) {
    return initializeRayCommon<RayPayloadT>(params, pixel, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);
}


// ========== 光线最终化函数：将光线追踪结果输出为渲染图像 ==========
//
// 📄 【功能概述】
// 这是3DGUT渲染管线的最后一步，负责将每条光线的计算结果写入输出缓冲区
// 生成最终的RGBA图像、深度图和调试信息
//
// 📊 【输出数据】
// 1. RGBA颜色：ray.features[RGB] + (1-transmittance)[Alpha]
// 2. 深度信息：ray.hitT (加权平均击中距离)  
// 3. 击中计数：ray.hitN (调试用，可选)
//
// 🚀 【性能特点】
// - GPU设备函数：在CUDA核心上并行执行
// - 内联优化：减少函数调用开销
// - 内存合并：所有输出使用相同索引ray.idx，提高内存带宽利用率
//
template <typename TRayPayload>  // 模板参数：支持不同类型的光线载荷(RayPayload<3>, RayPayloadBackward等)
__device__ __inline__ void finalizeRay(
    const TRayPayload& ray,                    // 输入：光线数据载荷，包含累积的特征、透射率、击中距离等
    const threedgut::RenderParameters& params, // 输入：渲染参数（本函数中未使用，但保持接口一致性）
    const tcnn::vec3* __restrict__ sensorRayOriginPtr,  // 输入：传感器光线起点数组（本函数中未使用）
    float* __restrict__ worldCountPtr,         // 输出：每像素击中粒子计数 [width×height] （调试信息）
    float* __restrict__ worldHitDistancePtr,   // 输出：每像素深度信息 [width×height] （Z-buffer）
    tcnn::vec4* __restrict__ radianceDensityPtr, // 输出：每像素RGBA颜色 [width×height×4] （最终图像）
    const tcnn::mat4x3& sensorToWorldTransform) { // 输入：传感器到世界空间变换矩阵（本函数中未使用）

    // ========== 第1步：光线有效性检查 ==========
    // 
    // 📋 【检查条件】只有满足以下条件的光线才被认为是有效的：
    // - 光线与场景包围盒相交：ray.tMinMax.y > ray.tMinMax.x
    // - 在initializeRay中被标记为Valid: ray.flags |= RayPayloadT::Valid
    //
    // 🎯 【优化原理】早期退出策略：
    // - 避免对无效像素进行无意义的写入操作
    // - 减少内存带宽消耗
    // - 防止向输出缓冲区写入未初始化的垃圾数据
    if (!ray.isValid()) {
        return; // 早期退出：光线未与场景相交，该像素保持背景色
    }

    // ========== 第2步：输出RGBA颜色到主渲染缓冲区 ==========
    //
    // 📊 【数据来源与物理意义】
    // RGB: ray.features[0,1,2] - 光线在追踪过程中累积的辐射特征
    //      - 通过Alpha混合公式计算：color += hitWeight * particleColor
    //      - 代表该像素接收到的红、绿、蓝光辐射量
    //
    // Alpha: (1.0f - ray.transmittance) - 不透明度
    //        - transmittance=1.0 → 完全透明 → Alpha=0.0
    //        - transmittance=0.0 → 完全不透明 → Alpha=1.0  
    //        - transmittance在光线追踪中递减：transmittance *= (1.0 - hitAlpha)
    //
    // 🎨 【颜色混合原理】
    // 每个粒子对最终颜色的贡献：hitWeight = hitAlpha * currentTransmittance
    // 累积过程：ray.features += hitWeight * particleFeatures
    // 透射率更新：ray.transmittance *= (1.0 - hitAlpha)
    //
    // 💾 【内存布局】tcnn::vec4结构 = {x, y, z, w} = {R, G, B, A}
    radianceDensityPtr[ray.idx] = {
        ray.features[0],              // R: 红色通道辐射量
        ray.features[1],              // G: 绿色通道辐射量  
        ray.features[2],              // B: 蓝色通道辐射量
        (1.0f - ray.transmittance)    // A: Alpha不透明度 = 1 - 透明度
    };

    // ========== 第3步：输出深度信息到Z-Buffer ==========
    //
    // 📏 【深度计算原理】
    // ray.hitT存储的是加权平均击中距离：
    // - 每次击中时累积：ray.hitT += hitT * hitWeight
    // - 其中hitT是光线参数化距离：P = origin + hitT * direction
    // - hitWeight是该粒子对最终颜色的贡献权重
    //
    // 🎯 【用途】
    // - 深度图可视化：显示场景的3D结构
    // - 深度测试：用于后期处理效果（DOF、AO等）
    // - 调试工具：检查光线追踪的正确性
    //
    // 📊 【数值范围】
    // - 近似范围：[ray.tMinMax.x, ray.tMinMax.y] 
    // - 实际值：加权平均，可能不等于任何单个粒子的真实距离
    worldHitDistancePtr[ray.idx] = ray.hitT;

    // ========== 第4步：可选的调试信息输出 ==========
    //
    // 🔧 【条件编译】只有在编译时定义GAUSSIAN_ENABLE_HIT_COUNT宏时才编译此代码
    // - 优化考虑：避免在生产环境中产生不必要的内存写入开销
    // - 调试开关：可通过编译选项控制是否启用击中计数统计
    //
    // 📈 【击中计数的用途】
    // - 性能调试：高击中数可能表示该区域粒子密度过高
    // - 质量分析：击中数太少可能导致采样不足，产生噪声
    // - 负载均衡：帮助识别计算密集的像素区域
    //
    // 💡 【典型数值含义】
    // - hitN = 0：该像素未击中任何粒子（背景区域）
    // - hitN = 1-10：正常密度区域
    // - hitN > 50：高密度区域，可能存在性能瓶颈
#if GAUSSIAN_ENABLE_HIT_COUNT
    worldCountPtr[ray.idx] = (float)ray.hitN; // 将整数击中计数转换为浮点数存储
#endif

    // ========== 函数执行完毕 ==========
    // 
    // 🎉 【执行结果】
    // 经过此函数处理后，该线程对应的像素已完成：
    // 1. ✅ RGBA颜色写入 → 可用于图像显示
    // 2. ✅ 深度信息写入 → 可用于后处理
    // 3. ✅ 调试信息写入 → 可用于性能分析
    //
    // 🔄 【后续流程】
    // GPU内核执行完毕后，这些缓冲区将被传回CPU/显示系统：
    // - radianceDensityPtr → 转换为最终图像（PNG/JPG等）
    // - worldHitDistancePtr → 可视化为深度图或用于后处理
    // - worldCountPtr → 生成热力图用于性能分析
}
