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
 * 球谐辐射高斯粒子系统 (Spherical Harmonic Radiative Gaussian Particles)
 * 
 * 本文件实现了3D高斯溅射(3DGS)的C++包装层，提供了Slang核心算法与CUDA C++代码之间的桥梁。
 * 主要功能包括：
 * - 高效的内存缓冲区管理（支持可微分/非可微分模式）
 * - 类型安全的接口封装（TCNN ↔ CUDA类型转换）
 * - 高度优化的GPU计算内核（Warp级别负载均衡）
 * - 自动微分梯度管理（训练阶段的参数优化）
 * 
 * 设计理念：
 * 🔧 零开销抽象：模板特化确保编译时优化，运行时无性能损失
 * 🚀 GPU友好架构：针对CUDA warp执行模型深度优化
 * 🎯 类型安全：编译时检查确保接口正确性
 * ⚡ 内存高效：精确的缓冲区管理，最小化内存占用
 */

#pragma once

#include <3dgut/kernels/cuda/models/gaussianParticles.cuh>
#include <3dgut/renderer/renderParameters.h>

// ========== 智能缓冲区管理系统 - 支持可微分/非可微分双模式 ==========
//
// 【设计原理】：
// 使用模板特化实现条件编译，根据TDifferentiable参数决定缓冲区结构：
// - false: 只包含数据指针（推理模式，节省内存）
// - true:  包含数据+梯度指针（训练模式，支持反向传播）
//
// 【性能优势】：
// - 编译时决策：零运行时开销
// - 内存精确控制：推理时不分配梯度缓冲区
// - 类型安全：模板确保类型一致性

/**
 * 基础缓冲区模板 - 非可微分版本（推理模式）
 * 
 * 功能：为推理阶段提供轻量级的数据缓冲区
 * 特点：只包含数据指针，不包含梯度指针，节省内存
 * 用途：模型推理、实时渲染等不需要梯度计算的场景
 */
template <typename TBuffer, bool TDifferentiable>
struct ShRadiativeGaussianParticlesBuffer {
    TBuffer* ptr = nullptr; // 数据缓冲区指针（只读或读写）
};

/**
 * 可微分版本 - 模板特化（训练模式）
 * 
 * 功能：为训练阶段提供完整的数据+梯度缓冲区
 * 特点：包含数据指针和梯度指针，支持自动微分
 * 用途：神经网络训练、参数优化等需要梯度计算的场景
 */
template <typename TBuffer>
struct ShRadiativeGaussianParticlesBuffer<TBuffer, true> {
    TBuffer* ptr     = nullptr;  // 前向数据缓冲区指针（参数值）
    TBuffer* gradPtr = nullptr;  // 反向梯度缓冲区指针（参数梯度）
};

/**
 * 可选缓冲区模板 - 默认禁用版本
 * 
 * 功能：为可选功能提供条件编译支持
 * 设计：使用SFINAE技术，根据Enabled参数决定是否包含缓冲区
 * 用途：实现功能开关，如调试信息、统计数据等可选缓冲区
 */
template <typename TBuffer, bool TDifferentiable, bool Enabled>
struct ShRadiativeGaussianParticlesOptionalBuffer {
    // 空结构体：当Enabled=false时，不包含任何成员（零内存开销）
};

/**
 * 可选缓冲区模板 - 启用版本（模板特化）
 * 
 * 功能：当Enabled=true时，继承标准缓冲区功能
 * 设计：通过继承实现代码复用，避免重复定义
 * 优势：编译时条件包含，不影响性能
 */
template <typename TBuffer, bool TDifferentiable>
struct ShRadiativeGaussianParticlesOptionalBuffer<TBuffer, TDifferentiable, true> : ShRadiativeGaussianParticlesBuffer<TBuffer, TDifferentiable> {
    // 继承父类的所有成员和功能
};

// ========== 核心粒子系统类 - 多层模板架构的集大成者 ==========
//
// 【设计理念】：
// 这是整个3DGUT系统的核心C++封装类，采用多重继承+模板特化的先进设计：
// - Params: 提供缓冲区索引等编译时常量（配置层）
// - ExtParams: 提供算法参数如KernelDegree等（算法层）  
// - TDifferentiable: 控制梯度计算的开启/关闭（性能层）
//
// 【功能概览】：
// 1. 内存管理：高效的GPU缓冲区管理和类型转换
// 2. 接口适配：Slang ↔ CUDA C++的无缝桥接
// 3. 性能优化：Warp级别的并行计算优化
// 4. 自动微分：完整的前向+反向传播支持

/**
 * 球谐辐射体积特征粒子系统
 * 
 * 这是3DGUT渲染管线的核心C++封装类，整合了：
 * - 基础几何计算（来自gaussianParticles.slang）
 * - 球谐光照计算（来自shRadiativeParticles.slang）
 * - 高性能GPU内存管理
 * - 自动微分梯度计算
 * 
 * 模板参数：
 * @tparam Params 配置参数类（提供缓冲区索引等常量）
 * @tparam ExtParams 扩展参数类（提供算法参数如核函数阶数）
 * @tparam TDifferentiable 是否支持可微分计算（影响梯度缓冲区分配）
 */
template <typename Params,
          typename ExtParams,
          bool TDifferentiable = true>
struct ShRadiativeGaussianVolumetricFeaturesParticles : Params, public ExtParams {

    // ========== 类型定义：建立CUDA与Slang之间的类型映射 ==========
    using DensityParameters    = threedgut::ParticeFetchedDensity;    // 处理后的密度参数（包含预计算的旋转矩阵）
    using DensityRawParameters = threedgut::ParticleDensity;          // 原始密度参数（紧凑存储格式）

    // ========== 系统初始化函数：GPU内存管理的核心 ==========
    
    /**
     * 初始化密度参数缓冲区指针
     * 
     * 功能：建立GPU内存句柄与类型化指针之间的映射关系
     * 重要：使用static_assert确保C++类型与Slang类型的二进制兼容性
     * 
     * 设计原理：
     * - 内存句柄提供统一的64位指针存储
     * - 类型化指针提供类型安全的数据访问
     * - 编译时断言确保跨语言类型一致性
     * 
     * @param parameters 内存句柄管理器，包含所有GPU缓冲区的指针
     */
    __forceinline__ __device__ void initializeDensity(threedgut::MemoryHandles parameters) {
        // 🔒 编译时类型安全检查：确保C++与Slang数据结构完全匹配
        // 这些断言防止因类型大小不匹配导致的内存访问错误
        //
        // 📝 重要说明：gaussianParticle_RawParameters_0 类型来源
        // ============================================================
        // 这个类型并非在C++源码中直接定义，而是由Slang编译器自动生成：
        //
        // 1. 🎯 生成过程：
        //    - setup_3dgut.py 调用 slangc 编译器
        //    - 编译 include/3dgut/kernels/slang/models/gaussianParticles.slang
        //    - 自动生成 threedgutSlang.cuh 头文件
        //
        // 2. 🔄 类型映射：
        //    Slang源码:     struct RawParameters { ... }  
        //    生成的C++:    struct gaussianParticle_RawParameters_0 { ... }
        //    C++包装:      using DensityRawParameters = threedgut::ParticleDensity;
        //
        // 3. 🛡️ 作用：确保跨语言（C++ ↔ Slang）的内存布局兼容性
        //    - 防止结构体大小不匹配导致的内存访问错误
        //    - 保证GPU内存中的数据可以在两种语言间安全传递
        //
        // 4. 📁 生成文件位置：通常在构建目录中的 threedgutSlang.cuh
        //
        static_assert(sizeof(DensityRawParameters) == sizeof(gaussianParticle_RawParameters_0), 
                     "Sizes must match for binary compatibility");
        static_assert(sizeof(DensityParameters) == sizeof(gaussianParticle_Parameters_0), 
                     "Sizes must match for binary compatibility");
        
        // 这是一个典型的类型安全的内存映射过程，将抽象的内存句柄转换为具体的类型化指针，实现高效的GPU数据访问！
        m_densityRawParameters.ptr =
            parameters.bufferPtr<DensityRawParameters>(Params::DensityRawParametersBufferIndex);
    }

    /**
     * 初始化密度梯度缓冲区指针
     * 
     * 功能：为训练模式配置梯度缓冲区（仅在TDifferentiable=true时生效）
     * 优势：编译时条件编译，推理模式下完全无开销
     * 
     * 设计细节：
     * - constexpr if：C++17特性，编译时分支消除
     * - 推理模式：gradPtr保持nullptr，节省内存
     * - 训练模式：分配梯度缓冲区，支持反向传播
     * 
     * @param parametersGradient 梯度内存句柄管理器
     */
    __forceinline__ __device__ void initializeDensityGradient(threedgut::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            // 🎓 训练模式：分配梯度缓冲区，用于存储反向传播的梯度信息
            m_densityRawParameters.gradPtr =
                parametersGradient.bufferPtr<DensityRawParameters>(Params::DensityRawParametersGradientBufferIndex);
        }
        // 🚀 推理模式：if constexpr确保此分支在推理模式下完全被优化掉
    };

    // ========== 数据访问接口：高效的粒子参数获取系统 ==========
    
    /**
     * 获取原始密度参数（紧凑存储格式）
     * 
     * 功能：直接从GPU全局内存获取粒子的原始存储参数
     * 特点：最小内存占用，适合大规模粒子系统
     * 用途：存储和传输，需要进一步处理才能用于计算
     * 
     * @param particleIdx 粒子在缓冲区中的索引
     * @return 包含位置、旋转四元数、缩放、密度的原始参数
     */
    __forceinline__ __device__ DensityRawParameters fetchDensityRawParameters(uint32_t particleIdx) const {
        return m_densityRawParameters.ptr[particleIdx];  // 直接内存访问，最高效
    }

    /**
     * 获取处理后的密度参数（计算优化格式）
     * 
     * 功能：调用Slang函数将原始参数转换为计算友好的格式
     * 优化：四元数预转换为旋转矩阵转置，避免渲染循环中重复计算
     * 关键：这里调用了particleDensityParameters (Slang导出函数)
     * 
     * 转换内容：
     * - 四元数 → 3x3旋转矩阵转置
     * - 保持位置、缩放、密度不变
     * - 内存布局优化
     * 
     * @param particleIdx 粒子索引
     * @return 包含预计算旋转矩阵的优化参数结构
     */
    __forceinline__ __device__ DensityParameters fetchDensityParameters(uint32_t particleIdx) const {
        // 🔄 调用Slang导出函数进行参数优化转换
        // 这里调用了particleDensityParameters (Slang导出函数)
        const auto parameters = particleDensityParameters(
            particleIdx,
            {reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr});
        // 🎯 类型转换：Slang类型 → C++类型
        return *reinterpret_cast<const DensityParameters*>(&parameters);
    }

    /**
     * 获取粒子位置（快速访问接口）
     * 
     * 功能：直接从原始参数中提取位置信息，无需完整参数转换
     * 优势：避免不必要的四元数→矩阵转换，提高性能
     * 用途：粒子排序、剔除、AABB计算等只需位置的操作
     * 
     * @param particleIdx 粒子索引
     * @return 粒子在世界空间的3D位置
     */
    __forceinline__ __device__ tcnn::vec3 fetchPosition(uint32_t particleIdx) const {
        // 🚀 直接内存访问+类型转换，最高效的位置获取方式
        return *(reinterpret_cast<const tcnn::vec3*>(&m_densityRawParameters.ptr[particleIdx].position));
    }

    // ========== 参数提取器：从处理后参数中提取特定属性 ==========
    
    /**
     * 从密度参数中提取位置
     * 
     * 功能：类型安全的位置属性访问器
     * 设计：引用返回避免不必要的拷贝
     */
    __forceinline__ __device__ const tcnn::vec3& position(const DensityParameters& parameters) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.position));
    }

    /**
     * 从密度参数中提取缩放
     * 
     * 功能：获取粒子在三个轴向的缩放因子
     * 用途：椭球形状控制、体积计算
     */
    __forceinline__ __device__ const tcnn::vec3& scale(const DensityParameters& parameters) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.scale));
    }

    /**
     * 从密度参数中提取旋转矩阵
     * 
     * 功能：获取预计算的3x3旋转矩阵
     * 重要：处理行主序(Slang)与列主序(TCNN)之间的差异
     * 
     * 技术细节：
     * - Slang使用行主序存储
     * - TCNN使用列主序存储  
     * - 返回旋转矩阵(非转置)以匹配TCNN约定
     */
    __forceinline__ __device__ const tcnn::mat3& rotation(const DensityParameters& parameters) const {
        // 🔄 内存布局转换：Slang行主序 → TCNN列主序
        // slang uses row-major order (tcnn uses column-major order), so we return the rotation (not transposed)
        return *(reinterpret_cast<const tcnn::mat3*>(&parameters.rotationT));
    }

    /**
     * 从密度参数中提取不透明度
     * 
     * 功能：获取粒子的基础密度值
     * 用途：透明度计算、体积渲染权重
     */
    __forceinline__ __device__ const float& opacity(const DensityParameters& parameters) const {
        return parameters.density;
    }

    // ========== densityHit调用链 - 第2层：C++包装接口 ==========
    //
    // 调用链结构：
    // 1. K-Buffer (gutKBufferRenderer.cuh:326) → particles.densityHit()
    // 2. 【当前层】C++包装 (shRadiativeGaussianParticles.cuh:115) → particleDensityHit()
    // 3. Slang导出 (gaussianParticles.slang:568) → gaussianParticle.hit()
    // 4. 核心实现 (gaussianParticles.slang:357) → 实际计算hitT
    //
    // 【本层作用】：类型转换和接口适配
    // - 将TCNN向量类型转换为CUDA内置类型
    // - 适配C++对象成员函数调用到全局函数调用
    // - 处理可选参数（法线计算）
    __forceinline__ __device__ bool densityHit(const tcnn::vec3& rayOrigin,     // 输入：光线起点（TCNN向量格式）
                                               const tcnn::vec3& rayDirection,   // 输入：光线方向（TCNN向量格式）
                                               const DensityParameters& parameters, // 输入：粒子密度参数
                                               float& alpha,                     // 输出：不透明度
                                               float& depth,                     // 输出：击中距离（hitT）
                                               tcnn::vec3* normal = nullptr) const { // 输出：表面法线（可选）

        // ========== 类型转换和函数调用转发 ==========
        // 功能：将C++成员函数调用转换为Slang全局函数调用
        // 类型转换：tcnn::vec3 → float3, DensityParameters → gaussianParticle_Parameters_0
        return particleDensityHit(*reinterpret_cast<const float3*>(&rayOrigin),        // TCNN → CUDA类型转换
                                  *reinterpret_cast<const float3*>(&rayDirection),      // TCNN → CUDA类型转换
                                  reinterpret_cast<const gaussianParticle_Parameters_0&>(parameters), // 参数结构转换
                                  &alpha,                                               // 直接传递指针
                                  &depth,                                               // 直接传递指针（这将成为hitT）
                                  normal != nullptr,                                    // 布尔标志：是否计算法线
                                  reinterpret_cast<float3*>(normal));                   // 法线指针转换（可为空）
    }

    // ========== densityIntegrateHit调用链 - 第2层：C++包装接口 ==========
    //
    // 🔗 【调用链位置】：
    // 1. K-Buffer (gutKBufferRenderer.cuh:244) → particles.densityIntegrateHit()
    // 2. 【当前层】C++包装 (shRadiativeGaussianParticles.cuh:344) → particleDensityIntegrateHit()
    // 3. Slang导出 (gaussianParticles.slang:873) → gaussianParticle.integrateHit<false>()
    // 4. 核心实现 (gaussianParticles.slang:551) → 实际Alpha混合计算
    //
    // 【本层作用】：类型转换和参数适配层
    // - 将TCNN类型转换为CUDA原生类型：tcnn::vec3* → float3*
    // - 将C++引用参数转换为指针参数：float& → float*
    // - 处理可选参数的逻辑：normal指针的空值检查和默认值设置
    // - 提供类型安全的接口，隐藏底层Slang实现细节
    //
    // 📊 【参数转换映射】：
    // alpha (float) → alpha (float)                    // 直接传递
    // transmittance (float&) → &transmittance (float*) // 引用转指针
    // depth (float) → depth (float)                    // 直接传递
    // integratedDepth (float&) → &integratedDepth (float*) // 引用转指针
    // normal (tcnn::vec3*) → *reinterpret_cast<float3*>(normal) // 类型转换
    //
    __forceinline__ __device__ float densityIntegrateHit(float alpha,
                                                         float& transmittance,
                                                         float depth,
                                                         float& integratedDepth,
                                                         const tcnn::vec3* normal     = nullptr,
                                                         tcnn::vec3* integratedNormal = nullptr) const {
        // ========== 类型转换和参数转发 ==========
        // 功能：将C++风格的API转换为Slang兼容的C风格API
        // 关键：确保内存布局兼容性和参数传递的正确性
        return particleDensityIntegrateHit(alpha,                      // 直接传递不透明度值
                                           &transmittance,              // 引用转指针：透射率（输入输出）
                                           depth,                       // 直接传递深度值
                                           &integratedDepth,            // 引用转指针：积分深度（输入输出）
                                           normal != nullptr,           // 布尔标志：是否需要计算法线
                                           normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(&normal), // 法线转换或默认值
                                           reinterpret_cast<float3*>(integratedNormal)); // 积分法线指针转换
    }

    // 从缓冲区处理密度光线击中（前向）
    __forceinline__ __device__ float densityProcessHitFwdFromBuffer(const tcnn::vec3& rayOrigin,
                                                                    const tcnn::vec3& rayDirection,
                                                                    uint32_t particleIdx,
                                                                    float& transmittance,
                                                                    float& integratedDepth,
                                                                    tcnn::vec3* integratedNormal = nullptr) const {
        return particleDensityProcessHitFwdFromBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                      *reinterpret_cast<const float3*>(&rayDirection),
                                                      particleIdx,
                                                      {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr, true}},
                                                      &transmittance,
                                                      &integratedDepth,
                                                      integratedNormal != nullptr,
                                                      reinterpret_cast<float3*>(integratedNormal));
    }

    template <bool exclusiveGradient>
    /**
     * 📋 DENSITY反向传播调用链（第2层/4层）- C++到Slang桥接层
     * 
     * 🔄 调用路径：
     * 1. 上层：gutKBufferRenderer.cuh:205 → particles.densityProcessHitBwdToBuffer<false>()
     * 2. 【当前层】shRadiativeGaussianParticles.cuh:417 → particleDensityProcessHitBwdToBuffer()
     * 3. 下层：gaussianParticles.slang:1063 → [Slang自动微分系统]
     * 
     * 🎯 本层职责：类型转换和参数组织
     * - TCNN向量类型 → CUDA原生float3类型  
     * - 封装高斯粒子参数缓冲区信息（几何参数指针+梯度指针）
     * - 处理可选参数（法线计算的条件处理）
     * - 传递exclusiveGradient控制（性能优化标志）
     * 
     * 🧮 数学原理：
     * - 从∂L/∂alpha反向传播到∂L/∂(位置,旋转,缩放,密度)
     * - 链式法则：∂L/∂θ = ∂L/∂alpha × ∂alpha/∂θ 
     * - 涉及复杂的坐标变换链：世界坐标→粒子坐标→标准化坐标
     * - 高斯核函数的导数：∂G/∂θ = 梯度(exp(-0.5×|变换后距离|²))
     */
    __forceinline__ __device__ void densityProcessHitBwdToBuffer(const tcnn::vec3& rayOrigin,
                                                                 const tcnn::vec3& rayDirection,
                                                                 uint32_t particleIdx,
                                                                 float alpha,
                                                                 float alphaGrad,
                                                                 float& transmittance,
                                                                 float& transmittanceGrad,
                                                                 float depth,
                                                                 float& integratedDepth,
                                                                 float& integratedDepthGrad,
                                                                 const tcnn::vec3* normal         = nullptr,
                                                                 tcnn::vec3* integratedNormal     = nullptr,
                                                                 tcnn::vec3* integratedNormalGrad = nullptr

    ) const {
        if constexpr (TDifferentiable) {
            // 🌉 关键桥接：C++ → Slang，复杂的类型转换和参数重组
            particleDensityProcessHitBwdToBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                 *reinterpret_cast<const float3*>(&rayDirection),
                                                 particleIdx,
                                                 {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr),
                                                   reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.gradPtr),
                                                   exclusiveGradient}},
                                                 alpha,
                                                 alphaGrad,
                                                 &transmittance,
                                                 &transmittanceGrad,
                                                 depth,
                                                 &integratedDepth,
                                                 &integratedDepthGrad,
                                                 normal != nullptr,
                                                 normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                 reinterpret_cast<float3*>(integratedNormal),
                                                 reinterpret_cast<float3*>(integratedNormalGrad));
        }
    }

    // 自定义密度光线击中检测
    __forceinline__ __device__ bool densityHitCustom(const tcnn::vec3& rayOrigin,
                                                     const tcnn::vec3& rayDirection,
                                                     uint32_t particleIdx,
                                                     float minHitDistance,
                                                     float maxHitDistance,
                                                     float maxParticleSquaredDistance,
                                                     float& hitDistance) const {
        return particleDensityHitCustom(*reinterpret_cast<const float3*>(&rayOrigin),
                                        *reinterpret_cast<const float3*>(&rayDirection),
                                        particleIdx,
                                        {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr, true}},
                                        minHitDistance,
                                        maxHitDistance,
                                        maxParticleSquaredDistance,
                                        &hitDistance);
    }

    // 实例化密度击中检测
    __forceinline__ __device__ bool densityHitInstance(const tcnn::vec3& canonicalRayOrigin,
                                                       const tcnn::vec3& canonicalUnormalizedRayDirection,
                                                       float minHitDistance,
                                                       float maxHitDistance,
                                                       float maxParticleSquaredDistance,
                                                       float& hitDistance

    ) const {
        return particleDensityHitInstance(*reinterpret_cast<const float3*>(&canonicalRayOrigin),
                                          *reinterpret_cast<const float3*>(&canonicalUnormalizedRayDirection),
                                          minHitDistance,
                                          maxHitDistance,
                                          maxParticleSquaredDistance,
                                          &hitDistance);
    }

    // 计算密度入射方向
    __forceinline__ __device__ tcnn::vec3 densityIncidentDirection(const DensityParameters& parameters,
                                                                   const tcnn::vec3& sourcePosition)

    {
        const auto incidentDirection = particleDensityIncidentDirection(reinterpret_cast<const gaussianParticle_Parameters_0&>(parameters),
                                                                        *reinterpret_cast<const float3*>(&sourcePosition));
        return *reinterpret_cast<const tcnn::vec3*>(&incidentDirection);
    }

    template <bool exclusiveGradient>
    // 密度入射方向反向传播到缓冲区
    __forceinline__ __device__ void densityIncidentDirectionBwdToBuffer(uint32_t particlesIdx,
                                                                        const tcnn::vec3& sourcePosition)

    {
        particleDensityIncidentDirectionBwdToBuffer(particlesIdx,
                                                    {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr),
                                                      reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.gradPtr),
                                                      exclusiveGradient}},
                                                    *reinterpret_cast<const float3*>(&sourcePosition));
    }

    // ========== 球谐特征系统：高级光照计算的核心基础 ==========
    
    using FeaturesParameters = shRadiativeParticle_Parameters_0;         // Slang球谐参数类型
    using TFeaturesVec       = typename tcnn::vec<ExtParams::FeaturesDim>; // 特征向量类型（通常3D RGB）

    /**
     * 初始化球谐特征参数缓冲区
     * 
     * 功能：建立球谐光照系统的内存映射和配置加载
     * 关键：同时设置数据指针和球谐阶数（影响计算复杂度和质量）
     * 
     * 球谐系统架构：
     * - 每个粒子存储N个球谐系数（N由球谐阶数决定）
     * - 系数数组：[DC, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, ...]
     * - 动态阶数：运行时可调整质量vs性能平衡
     * 
     * 内存布局：
     * - 连续存储：particleIdx * RadianceMaxNumSphCoefficients + coeffIdx
     * - 类型安全：编译时检查确保3D辐射维度兼容性
     * - 全局配置：球谐阶数作为运行时参数
     * 
     * @param parameters 内存句柄管理器
     */
    inline __device__ void initializeFeatures(threedgut::MemoryHandles parameters) {
        // 🔒 编译时安全检查：确保特征维度为3（RGB兼容性）
        static_assert(ExtParams::FeaturesDim == 3, "Hardcoded 3-dimensional radiance because of Slang-Cuda interop");
        
        // 🎨 球谐系数缓冲区指针设置
        m_featureRawParameters.ptr = parameters.bufferPtr<float3>(Params::FeaturesRawParametersBufferIndex);
        
        // 📊 动态球谐阶数加载：从全局参数缓冲区读取当前使用的球谐阶数
        // 这允许运行时调整质量（更高阶数=更精确的光照，但计算开销更大）
        m_featureActiveShDegree = *reinterpret_cast<int*>(
            parameters.bufferPtr<uint8_t>(Params::GlobalParametersValueBufferIndex) + 
            Params::FeatureShDegreeValueOffset);
    };

    /**
     * 初始化球谐特征梯度缓冲区
     * 
     * 功能：为训练模式配置球谐系数的梯度缓冲区
     * 重要：只在可微分模式下分配，推理模式下完全无开销
     * 
     * 梯度管理策略：
     * - 推理模式：gradPtr = nullptr，零内存开销
     * - 训练模式：分配与数据缓冲区同样大小的梯度缓冲区
     * - 自动累积：多线程安全的梯度累积机制
     * 
     * @param parametersGradient 梯度内存句柄管理器
     */
    inline __device__ void initializeFeaturesGradient(threedgut::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            // 🎓 训练模式：分配球谐系数梯度缓冲区
            m_featureRawParameters.gradPtr = parametersGradient.bufferPtr<float3>(Params::FeaturesRawParametersGradientBufferIndex);
        }
        // 🚀 推理模式：编译时分支消除，零开销
    };

    // ========== 球谐特征计算接口：视角相关颜色生成的核心 ==========
    
    /**
     * 从缓冲区获取球谐特征向量（标准Slang接口）
     * 
     * 功能：根据入射方向解码球谐系数，生成该方向的辐射亮度
     * 原理：球谐基函数展开 - 从紧凑系数重建连续函数
     * 
     * 球谐解码过程：
     * 1. 获取粒子的所有球谐系数（16个float3，对应不同基函数）
     * 2. 根据入射方向计算各球谐基函数的值
     * 3. 加权求和：color = Σ(coefficient_i * basis_function_i(direction))
     * 4. 结果：该方向看到的RGB颜色
     * 
     * 技术细节：
     * - 调用Slang导出函数particleFeaturesFromBuffer
     * - 类型转换：Slang float3 ↔ TCNN vec3
     * - 方向相关：同一粒子在不同角度显示不同颜色
     * 
     * @param particleIdx 粒子索引
     * @param incidentDirection 入射光线方向（观察方向）
     * @return 该方向的辐射特征（通常为RGB颜色）
     */
    __forceinline__ __device__ TFeaturesVec featuresFromBuffer(uint32_t particleIdx,
                                                               const tcnn::vec3& incidentDirection) const {
        // 🌈 调用Slang球谐解码函数：coefficients + direction → color
        const auto features = particleFeaturesFromBuffer(
            particleIdx,
            {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree}, // 球谐参数配置
            *reinterpret_cast<const float3*>(&incidentDirection));                    // 观察方向
        
        // 🔄 类型转换：Slang结果 → TCNN向量格式
        return *reinterpret_cast<const TFeaturesVec*>(&features);
    }

    /**
     * 从缓冲区获取自定义球谐特征向量（高性能直接接口）
     * 
     * 功能：绕过Slang接口，直接调用优化的C++球谐解码函数
     * 优势：避免跨语言调用开销，更精确的数值控制
     * 
     * 技术差异：
     * - 标准接口：通过Slang导出函数，具有最好的兼容性
     * - 自定义接口：直接C++实现，具有更高的性能
     * - 都实现相同的数学算法：球谐基函数展开
     * 
     * 模板参数：
     * @tparam Clamped 是否对结果进行值域限制（避免HDR溢出）
     * 
     * @param particleIdx 粒子索引  
     * @param incidentDirection 入射光线方向
     * @return 解码后的辐射特征
     */
    template <bool Clamped = true>
    __forceinline__ __device__ TFeaturesVec featuresCustomFromBuffer(uint32_t particleIdx,
                                                                     const tcnn::vec3& incidentDirection) const {
        // 🚀 高性能路径：直接调用优化的C++球谐解码函数
        const float3 gradu = threedgut::radianceFromSpH(
            m_featureActiveShDegree,  // 球谐阶数
            reinterpret_cast<const float3*>(&m_featureRawParameters.ptr[particleIdx * ExtParams::RadianceMaxNumSphCoefficients]), // 系数指针
            *reinterpret_cast<const float3*>(&incidentDirection),  // 方向
            Clamped);  // 数值限制
        
        return *reinterpret_cast<const TFeaturesVec*>(&gradu);
    }

    template <bool exclusiveGradient>
    // 特征反向传播到缓冲区
    __forceinline__ __device__ void featuresBwdToBuffer(uint32_t particleIdx,
                                                        const TFeaturesVec& featuresGrad,
                                                        const tcnn::vec3& incidentDirection) const {

        particleFeaturesBwdToBuffer(particleIdx,
                                    {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                    *reinterpret_cast<const float3*>(&featuresGrad),
                                    *reinterpret_cast<const float3*>(&incidentDirection));
    }

    template <bool Atomic = false>
    // 自定义特征反向传播到缓冲区
    __forceinline__ __device__ void featuresBwdCustomToBuffer(uint32_t particleIdx,
                                                              const TFeaturesVec& features,
                                                              const TFeaturesVec& featuresGrad,
                                                              const tcnn::vec3& incidentDirection) const {
        threedgut::radianceFromSpHBwd<Atomic>(m_featureActiveShDegree,
                                      *reinterpret_cast<const float3*>(&incidentDirection),
                                      *reinterpret_cast<const float3*>(&featuresGrad),
                                      reinterpret_cast<float3*>(&m_featureRawParameters.gradPtr[particleIdx * ExtParams::RadianceMaxNumSphCoefficients]),
                                      *reinterpret_cast<const float3*>(&features));
    }

    // ========== 体积渲染积分接口：颜色的混合与累积 ==========
    
    // ========== featureIntegrateFwd调用链 - 第2层：C++包装接口 ==========
    //
    // 🎨 【调用链位置】：
    // 1. K-Buffer (gutKBufferRenderer.cuh:264) → particles.featureIntegrateFwd()
    // 2. 【当前层】C++包装 (shRadiativeGaussianParticles.cuh:661) → particleFeaturesIntegrateFwd()
    // 3. Slang导出 (shRadiativeParticles.slang:299) → shRadiativeParticle.integrateRadiance<false>()
    // 4. 核心实现 (shRadiativeParticles.slang:205) → 颜色混合算法
    //
    // 【本层作用】：辐射特征的类型转换和接口适配
    // - 功能：将单个粒子的辐射特征按权重积分到总颜色中
    // - 数学公式：integratedFeatures += weight * features
    // - 类型转换：TFeaturesVec (tcnn::vec3) → float3
    // - 提供类型安全的模板化API接口
    //
    // 【应用场景】：
    // - 传统的前向后渲染顺序（front-to-back rendering）
    // - 体积渲染中的视线穿越积分（ray marching integration）
    // - K-Buffer中的局部积分计算（local integration in K-Buffer）
    //
    // 📊 【参数转换映射】：
    // weight (float) → weight (float)                                               // 直接传递权重值
    // features (TFeaturesVec&) → *reinterpret_cast<float3*>(&features)               // TCNN向量转CUDA类型
    // integratedFeatures (TFeaturesVec&) → reinterpret_cast<float3*>(&integratedFeatures) // 累积结果转换
    //
    __forceinline__ __device__ void featureIntegrateFwd(float weight,
                                                        const TFeaturesVec& features,
                                                        TFeaturesVec& integratedFeatures) const {
        // ========== TCNN向量类型到CUDA原生类型的转换 ==========
        // 功能：将高层模板化的向量类型转换为底层Slang兼容的数据类型
        // 关键：reinterpret_cast确保内存布局兼容性，避免数据拷贝开销
        particleFeaturesIntegrateFwd(weight,                                          // 权重值直接传递
                                     *reinterpret_cast<const float3*>(&features),     // 粒子特征向量转换
                                     reinterpret_cast<float3*>(&integratedFeatures)); // 累积结果向量转换
    }

    /**
     * 从缓冲区进行特征前向积分（一体化接口）
     * 
     * 功能：组合了特征获取+积分的一体化操作，提高效率
     * 优势：减少中间变量和函数调用开销
     * 
     * 工作流程：
     * 1. 根据入射方向解码粒子的球谐系数
     * 2. 获取该方向的辐射亮度
     * 3. 按权重积分到累积结果中
     * 
     * @param incidentDirection 入射光线方向
     * @param weight 混合权重
     * @param particleIdx 粒子索引
     * @param integratedFeatures 累积特征结果（输入输出）
     */
    __forceinline__ __device__ void featuresIntegrateFwdFromBuffer(const tcnn::vec3& incidentDirection,
                                                                   float weight,
                                                                   uint32_t particleIdx, 
                                                                   TFeaturesVec& integratedFeatures) const {
        // 🚀 一体化高效接口：从缓冲区直接积分，减少中间操作
        particleFeaturesIntegrateFwdFromBuffer(
            *reinterpret_cast<const float3*>(&incidentDirection),
            weight,
            particleIdx,
            {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree},
            reinterpret_cast<float3*>(&integratedFeatures));
    }

    /**
     * 特征反向积分（梯度反向传播）
     * 
     * 功能：实现体积渲染积分操作的梯度反向传播
     * 关键：用于神经网络训练中的参数优化
     * 
     * 反向传播原理：
     * - 前向：integratedFeatures += alpha * features
     * - 反向：featuresGrad += alpha * integratedFeaturesGrad
     * -       alphaGrad += dot(features, integratedFeaturesGrad)
     * 
     * 数学背景：
     * - 这是链式法则的具体应用
     * - 每个中间变量都需要计算相对于输入的梯度
     * - 用于优化球谐系数和粒子透明度
     * 
     * @param alpha 粒子不透明度（前向值）
     * @param alphaGrad alpha的梯度（输入输出）
     * @param features 粒子特征（前向值）
     * @param featuresGrad 特征梯度（输入输出）
     * @param integratedFeatures 积分特征（前向值，可能被修改）
     * @param integratedFeaturesGrad 积分特征梯度（输入输出）
     */
    __forceinline__ __device__ void featuresIntegrateBwd(float alpha,
                                                         float& alphaGrad,
                                                         const TFeaturesVec& features,
                                                         TFeaturesVec& featuresGrad,
                                                         TFeaturesVec& integratedFeatures,
                                                         TFeaturesVec& integratedFeaturesGrad) const {
        if (TDifferentiable) {
            // 🎓 调用Slang实现的反向积分算法
            particleFeaturesIntegrateBwd(alpha,
                                         &alphaGrad,
                                         *reinterpret_cast<const float3*>(&features),
                                         reinterpret_cast<float3*>(&featuresGrad),
                                         reinterpret_cast<float3*>(&integratedFeatures),
                                         reinterpret_cast<float3*>(&integratedFeaturesGrad));
        }
        // 🚀 非可微分模式下，编译时分支消除
    }

    template <bool exclusiveGradient>
    /**
     * 📋 FEATURES反向传播调用链（第2层/5层）- C++到Slang桥接层
     * 
     * 🔄 调用路径：
     * 1. 上层：gutKBufferRenderer.cuh:172 → particles.featuresIntegrateBwdToBuffer<false>()
     * 2. 【当前层】shRadiativeGaussianParticles.cuh:753 → particleFeaturesIntegrateBwdToBuffer()  
     * 3. 下层：shRadiativeParticles.slang:485 → [Slang自动微分系统]
     * 
     * 🎯 本层职责：类型转换和参数封装
     * - TCNN向量类型 → CUDA原生float3类型
     * - 封装球谐参数缓冲区信息（数据指针+梯度指针+度数）
     * - 传递exclusiveGradient标志（控制原子操作vs直接写入）
     * 
     * 🧮 数学原理：
     * - 从∂L/∂RGB反向传播到∂L/∂(球谐系数)
     * - 链式法则：∂L/∂c_i = ∂L/∂RGB × ∂RGB/∂c_i
     * - ∂RGB/∂c_i = Y_i(ω) (球谐基函数在观察方向ω的值)
     */
    __forceinline__ __device__ void featuresIntegrateBwdToBuffer(const tcnn::vec3& incidentDirection,
                                                                 float alpha,
                                                                 float& alphaGrad,
                                                                 uint32_t particleIdx,
                                                                 const TFeaturesVec& features,
                                                                 TFeaturesVec& integratedFeatures,
                                                                 TFeaturesVec& integratedFeaturesGrad) const {

        if (TDifferentiable) {
            // 🌉 关键桥接：C++ → Slang，类型转换和参数封装
            particleFeaturesIntegrateBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                 alpha,
                                                 &alphaGrad,
                                                 particleIdx,
                                                 {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                                 *reinterpret_cast<const float3*>(&features),
                                                 reinterpret_cast<float3*>(&integratedFeatures),
                                                 reinterpret_cast<float3*>(&integratedFeaturesGrad));
        }
    }

    /**
     * 光线击中前向处理（高性能一体化接口）
     * 
     * 功能：组合了密度击中检测+特征计算+积分的一体化操作
     * 优势：减少函数调用开销，提高GPU执行效率
     * 关键：这是渲染循环中的核心热点函数
     * 
     * 工作流程：
     * 1. 调用densityHit检测光线与粒子是否相交
     * 2. 如果相交，获取粒子的辐射特征（颜色）
     * 3. 按alpha值进行透明度混合
     * 4. 更新透射率和累积颜色
     * 
     * 模板参数：
     * @tparam PerRayRadiance 是否使用每光线的辐射计算模式
     *   - true: 从缓冲区动态计算辐射（灵活，但慢）
     *   - false: 使用预计算的辐射值（快，但内存占用大）
     * 
     * @param rayOrigin 光线起点
     * @param rayDirection 光线方向
     * @param particleIdx 粒子索引
     * @param particleFeaturesPtr 预计算特征指针（仅PerRayRadiance=false时使用）
     * @param transmittance 透射率（输入输出）
     * @param features 累积特征（输入输出）
     * @param hitT 击中距离（输出）
     * @return 是否发生有效击中
     */
    template <bool PerRayRadiance>
    __forceinline__ __device__ bool processHitFwd(const tcnn::vec3& rayOrigin,
                                                  const tcnn::vec3& rayDirection,
                                                  uint32_t particleIdx,
                                                  const TFeaturesVec* particleFeaturesPtr,
                                                  float& transmittance,
                                                  TFeaturesVec& features,
                                                  float& hitT) const {
        // 🚀 调用高度优化的C++内核实现（包含所有渲染步骤）
        return threedgut::processHitFwd<ExtParams::KernelDegree, false, PerRayRadiance>(
            reinterpret_cast<const float3&>(rayOrigin),
            reinterpret_cast<const float3&>(rayDirection),
            particleIdx,
            m_densityRawParameters.ptr,
            PerRayRadiance ? reinterpret_cast<const float*>(m_featureRawParameters.ptr) : reinterpret_cast<const float*>(particleFeaturesPtr),
            ExtParams::MinParticleKernelDensity,  // 最小密度响应阈值
            ExtParams::AlphaThreshold,            // 最小透明度阈值
            m_featureActiveShDegree,
            &transmittance,
            reinterpret_cast<float3*>(&features),
            &hitT,
            nullptr);
    }

    /**
     * 反向传播粒子击中处理
     * 
     * 输入：
     *   rayOrigin, rayDirection     - 光线几何
     *   particleIdx                 - 粒子索引
     *   densityRawParameters        - 粒子几何参数(Forward值)
     *   particleFeatures            - 粒子特征(Forward值)  
     *   transmittanceBackward       - Forward计算的最终透射率
     *   featuresBackward            - Forward计算的最终特征
     *   hitTBackward               - Forward计算的最终击中距离
     *   transmittanceGradient       - ∂L/∂transmittance
     *   featuresGradient           - ∂L/∂features  
     *   hitTGradient               - ∂L/∂hitT
     * 
     * 输出：
     *   densityRawParametersGrad    - ∂L/∂(位置,旋转,缩放,密度)
     *   particleFeaturesGradPtr     - ∂L/∂特征 或 ∂L/∂球谐系数
     * 
     * 输入输出：
     *   transmittance, features, hitT - 当前累积状态，会被更新
     * 
     * 📋 调用链位置:
     *   1. evalBackwardNoKBuffer (gutKBufferRenderer.cuh:677) → particles.processHitBwd<>()
     *   2. 【当前层】C++包装 (shRadiativeGaussianParticles.cuh:858) → threedgut::processHitBwd<>()
     *   3. 🎯 最底层实现 (gaussianParticles.cuh:863) → 梯度计算数学公式
     * 
     * 🔄 本层职责: 类型适配和参数转换
     *   - 将TCNN向量类型转换为CUDA原生类型 (tcnn::vec3 → float3)  
     *   - 处理动态vs静态特征模式的分支逻辑
     *   - 传递模板参数（核函数度数、是否表面、是否动态特征）
     *   - 维护类型安全的高层API接口
     * 
     * ⚡ 性能优化:
     *   - reinterpret_cast零开销类型转换
     *   - constexpr条件编译时分支消除
     *   - 内联函数避免调用开销
     */
    template <bool PerRayRadiance>
    __forceinline__ __device__ void processHitBwd(
        const tcnn::vec3& rayOrigin,                    // 光线起点（TCNN向量格式）
        const tcnn::vec3& rayDirection,                 // 光线方向（TCNN向量格式）  
        uint32_t particleIdx,                           // 粒子索引
        const DensityRawParameters& densityRawParameters,    // 粒子几何参数（Forward值）
        DensityRawParameters* densityRawParametersGrad,      // 几何参数梯度输出缓冲区
        const TFeaturesVec& particleFeatures,               // 粒子特征（Forward值）
        TFeaturesVec* particleFeaturesGradPtr,              // 特征梯度输出缓冲区
        float& transmittance,                               // 当前透射率（会被修改）
        float transmittanceBackward,                        // Forward计算的透射率
        float transmittanceGradient,                        // 透射率梯度输入
        TFeaturesVec& features,                             // 当前特征（会被修改）
        const TFeaturesVec& featuresBackward,               // Forward计算的特征
        const TFeaturesVec& featuresGradient,               // 特征梯度输入
        float& hitT,                                        // 当前击中距离（会被修改）
        float hitTBackward,                                 // Forward计算的击中距离
        float hitTGradient                                  // 击中距离梯度输入
    ) const {
        
        // ========== 调用第3层：最底层C++反向传播实现 ==========
        // 🎯 关键转换: 高层模板化API → 底层优化实现
        threedgut::processHitBwd<ExtParams::KernelDegree, false, PerRayRadiance>(
            // 🔄 类型转换：TCNN向量 → CUDA原生类型
            reinterpret_cast<const float3&>(rayOrigin),     // tcnn::vec3 → float3
            reinterpret_cast<const float3&>(rayDirection),  // tcnn::vec3 → float3
            
            // 🏷️ 粒子标识和参数
            particleIdx,
            reinterpret_cast<const threedgut::ParticleDensity&>(densityRawParameters),     // 几何参数
            reinterpret_cast<threedgut::ParticleDensity*>(densityRawParametersGrad),       // 几何梯度
            
            // 🎨 特征数据分支：动态 vs 静态特征
            PerRayRadiance ? 
                reinterpret_cast<const float*>(m_featureRawParameters.ptr) :        // 动态：从球谐系数计算
                reinterpret_cast<const float*>(particleFeatures.data()),           // 静态：使用预计算特征
            PerRayRadiance ? 
                reinterpret_cast<float*>(m_featureRawParameters.gradPtr) :         // 动态：球谐梯度缓冲区
                reinterpret_cast<float*>(particleFeaturesGradPtr),                 // 静态：特征梯度缓冲区
            
            // 🎛️ 算法控制参数
            ExtParams::MinParticleKernelDensity,   // 最小核响应阈值（early culling）
            ExtParams::AlphaThreshold,             // 最小alpha阈值（透明度过滤）
            ExtParams::MinTransmittanceThreshold,  // 最小透射率阈值（early termination）
            m_featureActiveShDegree,               // 球谐函数活跃度数
            
            // 🔄 反向传播状态：Forward值 + 梯度输入
            transmittanceBackward,                  // Forward透射率
            transmittance,                          // 当前透射率（输入输出）
            transmittanceGradient,                  // 透射率梯度
            reinterpret_cast<const float3&>(featuresBackward),  // Forward特征
            reinterpret_cast<float3&>(features),                // 当前特征（输入输出） 
            reinterpret_cast<const float3&>(featuresGradient),  // 特征梯度
            hitT,                                   // 当前击中距离（输入输出）
            hitTBackward,                           // Forward击中距离
            hitTGradient);                          // 击中距离梯度
    }

    /**
     * 🚀 特征梯度更新 - 调用链第2层：最终执行层（无更深调用）
     * 
     * 📋 调用链位置:
     *   1. evalBackwardNoKBuffer (gutKBufferRenderer.cuh:699) → particles.processHitBwdUpdateFeaturesGradient()
     *   2. 🎯 【当前层】最终执行层 → 直接执行 Warp协作 + 原子操作
     * 
     * 🎓 论文技术实现: "部分负载均衡（Warp级别协作）"
     *   论文描述: "利用warp voting和shuffle指令在每个warp内重新分配剩余的工作负载"
     *   核心思想: 将32个线程的梯度先聚合，再一次性原子更新，减少内存竞争
     * 
     * ⚡ 性能优化原理:
     *   - 🔄 Warp内并行归约：32个线程协作计算总梯度
     *   - 🎯 单线程原子操作：仅第一个线程执行昂贵的atomicAdd
     *   - 📊 内存带宽优化：减少到全局内存的写入次数（32→1）
     *   - 🚀 竞争减少：降低原子操作的序列化开销
     * 
     * 🔬 算法细节:
     *   synchedThread=true:  O(log32) + 1次原子操作 = ~6次shuffle + 1次atomicAdd
     *   synchedThread=false: 32次原子操作 = 32次atomicAdd (用于对比)
     */
    template <bool synchedThread = true>
    __forceinline__ __device__ void processHitBwdUpdateFeaturesGradient(
        uint32_t particleIdx,           // 目标粒子索引
        TFeaturesVec& featuresGrad,     // 当前线程计算的特征梯度
        TFeaturesVec* featuresGradSum,  // 全局特征梯度累积数组
        uint32_t tileThreadIdx          // tile内线程索引（用于warp内定位）
    ) {
        if constexpr (synchedThread) {
            // ========== 第1步: Warp内并行归约（Butterfly Reduction Pattern） ==========
            // 🔄 使用shuffle指令实现高效的树形归约
            // 模式: 32线程 → 16线程 → 8线程 → 4线程 → 2线程 → 1线程
#pragma unroll
            for (int mask = 1; mask < warpSize; mask *= 2) {  // mask: 1,2,4,8,16 (log2(32)=5轮)
#pragma unroll
                for (int i = 0; i < ExtParams::FeaturesDim; ++i) {
                    // 📡 __shfl_xor_sync: 与距离mask的线程交换数据并累加
                    // 例如: thread0与thread1交换，thread2与thread3交换...
                    featuresGrad[i] += __shfl_xor_sync(0xffffffff, featuresGrad[i], mask);
                }
            }

            // ========== 第2步: 单线程原子更新（Warp Leader Optimization） ==========
            // 🎯 经过归约后，只有第一个线程（lane 0）拥有完整的warp总梯度
            if ((tileThreadIdx & (warpSize - 1)) == 0) {  // 检查是否为warp的第一个线程
#pragma unroll
                for (int i = 0; i < ExtParams::FeaturesDim; i++) {
                    // 💥 关键原子操作：将整个warp的梯度一次性累加到全局数组
                    atomicAdd(&featuresGradSum[particleIdx][i], featuresGrad[i]);
                }
            }
        } else {
            // ========== 备选方案: 直接原子更新（用于性能对比） ==========
            // ⚠️ 低效模式：每个线程独立执行原子操作，导致大量内存竞争
#pragma unroll
            for (int i = 0; i < ExtParams::FeaturesDim; ++i) {
                atomicAdd(&featuresGradSum[particleIdx][i], featuresGrad[i]);
            }
        }
    }

    /**
     * 🎛️ 密度梯度更新 - 调用链第2层：最终执行层（无更深调用）
     * 
     * 📋 调用链位置:
     *   1. evalBackwardNoKBuffer (gutKBufferRenderer.cuh:704) → particles.processHitBwdUpdateDensityGradient()
     *   2. 🎯 【当前层】最终执行层 → 直接执行 复合参数Warp协作
     * 
     * 🎓 论文技术实现: "Warp级别负载均衡的深度应用" 
     *   论文描述: "利用warp voting和shuffle指令在每个warp内重新分配剩余的工作负载"
     *   高级优化: 同时处理位置(3D)、四元数(4D)、缩放(3D)、密度(1D)共11个参数
     * 
     * ⚡ 相比特征梯度的复杂性:
     *   - 🎨 特征梯度: 3个float (RGB颜色)
     *   - 🔧 几何梯度: 11个float (位置3 + 四元数4 + 缩放3 + 密度1)
     *   - 📊 归约复杂度: ~3.7倍的数据量，但相同的warp协作效率
     * 
     * 🚀 性能关键: 几何参数直接影响粒子的3D变换，梯度精度对收敛至关重要
     */
    template <bool synchedThread = true>
    __forceinline__ __device__ void processHitBwdUpdateDensityGradient(
        uint32_t particleIdx,                       // 目标粒子索引
        DensityRawParameters& densityRawParameters, // 当前线程计算的几何参数梯度
        uint32_t tileThreadIdx                      // tile内线程索引（warp内定位）
    ) {
        if constexpr (synchedThread) {
            // ========== 第1步: 复合参数的Warp内并行归约 ==========
            // 🔄 对粒子的所有几何参数同时进行梯度聚合（11个标量）
            // 这是论文"第二阶段负载均衡"的具体实现：高维参数空间的协作优化
#pragma unroll
            for (int mask = 1; mask < warpSize; mask *= 2) {  // 5轮butterfly reduction
                // 位置梯度的warp归约
                densityRawParameters.position.x += __shfl_xor_sync(0xffffffff, densityRawParameters.position.x, mask);
                densityRawParameters.position.y += __shfl_xor_sync(0xffffffff, densityRawParameters.position.y, mask);
                densityRawParameters.position.z += __shfl_xor_sync(0xffffffff, densityRawParameters.position.z, mask);
                densityRawParameters.density += __shfl_xor_sync(0xffffffff, densityRawParameters.density, mask);
                // 旋转参数梯度的warp归约
                densityRawParameters.quaternion.x += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.x, mask);
                densityRawParameters.quaternion.y += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.y, mask);
                densityRawParameters.quaternion.z += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.z, mask);
                densityRawParameters.quaternion.w += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.w, mask);
                // 缩放参数梯度的warp归约
                densityRawParameters.scale.x += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.x, mask);
                densityRawParameters.scale.y += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.y, mask);
                densityRawParameters.scale.z += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.z, mask);
            }

            // First thread in the warp performs the atomic add
            if ((tileThreadIdx & (warpSize - 1)) == 0) {
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.x, densityRawParameters.position.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.y, densityRawParameters.position.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.z, densityRawParameters.position.z);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].density, densityRawParameters.density);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.x, densityRawParameters.quaternion.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.y, densityRawParameters.quaternion.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.z, densityRawParameters.quaternion.z);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.w, densityRawParameters.quaternion.w);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.x, densityRawParameters.scale.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.y, densityRawParameters.scale.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.z, densityRawParameters.scale.z);
            }
        } else {
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.x, densityRawParameters.position.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.y, densityRawParameters.position.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.z, densityRawParameters.position.z);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].density, densityRawParameters.density);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.x, densityRawParameters.quaternion.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.y, densityRawParameters.quaternion.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.z, densityRawParameters.quaternion.z);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.w, densityRawParameters.quaternion.w);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.x, densityRawParameters.scale.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.y, densityRawParameters.scale.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.z, densityRawParameters.scale.z);
        }
    }

    // ========== 内部数据管理：缓冲区和配置的封装 ==========
private:
    
    /**
     * 密度参数缓冲区管理器
     * 
     * 存储：
     * - 数据指针：指向包含所有粒子的几何参数的缓冲区
     * - 梯度指针：仅在TDifferentiable=true时分配，用于反向传播
     * 
     * 内容：每个粒子包含：
     * - position: float3 世界空间位置
     * - quaternion: float4 旋转四元数
     * - scale: float3 三轴缩放因子
     * - density: float 密度/不透明度
     */
    ShRadiativeGaussianParticlesBuffer<DensityRawParameters, TDifferentiable> m_densityRawParameters;

    /**
     * 当前激活的球谐阶数
     * 
     * 功能：控制球谐解码的计算复杂度和质量平衡
     * 取值范围：0-3（对应使用1, 4, 9, 16个系数）
     * 运行时可调：从全局参数缓冲区动态加载
     * 
     * 影响：
     * - 阶数越高，光照质量越好，但计算开销越大
     * - 阶数=0：仅常数项（类似传统RGB）
     * - 阶数=1：简单的方向性光照
     * - 阶数=2-3：复杂的光照环境效果
     */
    int m_featureActiveShDegree = 0;
    
    /**
     * 球谐系数缓冲区管理器
     * 
     * 存储：
     * - 数据类型：float3（RGB三通道的球谐系数）
     * - 存储布局：[particle0_coeff0, particle0_coeff1, ..., particle1_coeff0, ...]
     * - 系数数量：每个粒子RadianceMaxNumSphCoefficients个系数
     * - 梯度支持：仅在训练模式下分配梯度缓冲区
     * 
     * 数据结构：
     * - DC项：系数[0] - 常数光照分量
     * - 一阶项：系数[1-3] - 方向性光照
     * - 高阶项：系数[4-15] - 复杂光照环境
     */
    ShRadiativeGaussianParticlesBuffer<float3, TDifferentiable> m_featureRawParameters;
};
