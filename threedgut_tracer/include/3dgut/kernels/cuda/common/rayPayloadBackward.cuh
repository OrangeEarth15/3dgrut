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

/**
 * 🔄 反向传播光线数据结构和初始化函数
 * 
 * 📋 核心功能:
 *   - 扩展基础光线结构，添加反向传播所需的梯度字段
 *   - 从前向传播结果初始化反向光线，加载损失梯度
 * 
 * 🎯 设计原理:
 *   - 继承Forward的光线几何信息 (位置、方向、索引等)
 *   - 添加"Backward"和"Gradient"成对字段存储前向结果和对应梯度
 *   - 初始化时将像素级损失梯度分解到各个光线属性上
 */

#pragma once

#include <3dgut/kernels/cuda/common/rayPayload.cuh>

/**
 * 🌟 反向传播光线载荷结构
 * 
 * 📊 数据组织:
 *   - 继承基础光线的所有几何信息 (origin, direction, transmittance, etc.)
 *   - 每个可微分属性都有 "Backward" + "Gradient" 成对字段
 * 
 * 🔑 字段含义:
 *   - *Backward: 前向传播计算的最终值 (从GPU内存加载)
 *   - *Gradient: 损失函数对该属性的偏导数 (反向传播起点)
 */
template <int FeatN>
struct RayPayloadBackward : public RayPayload<FeatN> {
    // 📈 透射率相关: 光线穿过场景后剩余的强度
    float transmittanceBackward;        // Forward计算的最终透射率 T_final
    float transmittanceGradient;        // ∂L/∂T_final (来自像素损失)
    
    // 📏 击中距离相关: 光线与场景的交点深度
    float hitTBackward;                 // Forward计算的击中距离 t_hit  
    float hitTGradient;                 // ∂L/∂t_hit (深度损失，通常为0)
    
    // 🎨 特征相关: 光线携带的颜色/特征信息
    tcnn::vec<FeatN> featuresBackward;  // Forward计算的最终特征 [R,G,B,...]
    tcnn::vec<FeatN> featuresGradient;  // ∂L/∂features (主要梯度来源!)
};

/**
 * 🚀 初始化反向传播光线
 * 
 * 📋 执行步骤:
 *   1. 复用Forward的光线几何信息 (无需重新计算)
 *   2. 从GPU内存加载Forward的计算结果
 *   3. 加载对应的像素级损失梯度
 *   4. 设置反向传播的初始状态
 * 
 * ⚡ 性能特点:
 *   - 轻量级初始化，主要是内存读取操作
 *   - 不重复Forward的几何计算，只加载结果
 */
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeBackwardRay(
    const threedgut::RenderParameters& params,                                 // 渲染参数
    const tcnn::vec3* __restrict__ sensorRayOriginPtr,                         // 传感器光线起点数组
    const tcnn::vec3* __restrict__ sensorRayDirectionPtr,                      // 传感器光线方向数组
    const float* __restrict__ worldHitDistancePtr,                             // [Height×Width] Forward输出: 击中距离
    const float* __restrict__ worldHitDistanceGradientPtr,                     // [Height×Width] ∂L/∂distance
    const tcnn::vec<RayPayloadT::FeatDim + 1>* __restrict__ featuresDensityPtr, // [Height×Width×(Feat+1)] Forward输出: 颜色+密度
    const tcnn::vec<RayPayloadT::FeatDim + 1>* __restrict__ featuresDensityGradientPtr, // [Height×Width×(Feat+1)] ∂L/∂(颜色+密度)
    const tcnn::mat4x3& sensorToWorldTransform                                 // 传感器到世界坐标变换
) {

    // ========== 步骤1: 复用Forward光线几何信息 ==========
    // 📍 注意: 光线初始化/终结化过程本身不参与反向传播
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 sensorToWorldTransform);

    // ========== 步骤2: 加载Forward结果和损失梯度 ==========
    if (ray.isAlive()) {
        // 📊 从GPU内存读取当前像素的Forward计算结果
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensity         = featuresDensityPtr[ray.idx];         // [R,G,B,density]
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensityGradient = featuresDensityGradientPtr[ray.idx]; // ∂L/∂[R,G,B,density]
        
        // 🔄 透射率: T = 1 - density (体渲染标准公式)
        ray.transmittanceBackward = 1.f - featuresDensity[RayPayloadT::FeatDim];           // T_final = 1 - α_final
        ray.transmittanceGradient = -1.f * featuresDensityGradient[RayPayloadT::FeatDim];  // ∂L/∂T = -∂L/∂α
        
        // 📏 击中距离: 直接从Forward结果加载
        ray.hitTBackward = worldHitDistancePtr[ray.idx];           // t_hit (Forward计算)
        ray.hitTGradient = worldHitDistanceGradientPtr[ray.idx];   // ∂L/∂t_hit (通常为0)
        
        // 🎨 特征向量: 分离颜色部分 (排除density维度)
        ray.featuresBackward = threedgut::sliceVec<0, RayPayloadT::FeatDim>(featuresDensity);         // [R,G,B] (前N维)
        ray.featuresGradient = threedgut::sliceVec<0, RayPayloadT::FeatDim>(featuresDensityGradient); // ∂L/∂[R,G,B]
    }

    return ray;
}
