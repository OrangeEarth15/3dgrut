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

#include <3dgut/renderer/renderParameters.h> // 渲染参数定义（相机、场景、输出配置）
#include <3dgut/utils/cuda/cudaBuffer.h> // GPU内存管理工具类

#include <json/json.hpp> // nlohmann::json库，用于配置文件解析

#include <memory> // 智能指针支持（std::unique_ptr）

namespace threedgut {

class GUTRenderer {
public:
    struct GutRenderForwardContext;

    struct Parameters {
        struct {
            uint32_t numParticles; // 粒子数量
            int radianceSphDegree; // 球谐函数阶数
        } values; // 存储渲染过程中的标量参数，这些值会被传输到GPU常量内存。

        struct {
            void* dptrValuesBuffer; // 粒子属性值缓冲区指针
            void* dptrDensityParameters; // 密度参数缓冲区指针
            void* dptrRadianceParameters; // 辐射参数缓冲区指针
        } parameters; // 参数缓冲区（全局参数、密度参数、辐射参数）

        struct {
            void* dptrDensityGradients; // 密度梯度缓冲区指针
            void* dptrRadianceGradients; // 辐射梯度缓冲区指针
        } gradients; // 梯度缓冲区（密度梯度、辐射梯度）

        // CUDA缓冲区管理，RAII内存管理对象，自动管理GPU内存的生命周期
        threedgut::CudaBuffer parametersBuffer; // 参数缓冲区（全局参数、密度参数、辐射参数）
        threedgut::CudaBuffer gradientsBuffer; // 梯度缓冲区（密度梯度、辐射梯度）
        threedgut::CudaBuffer valuesBuffer; // 粒子属性值缓冲区

        // 指针数组
        uint64_t* m_dptrParametersBuffer = nullptr; // 参数缓冲区指针
        uint64_t* m_dptrGradientsBuffer  = nullptr; // 梯度缓冲区指针
    };

private:
    Logger m_logger;
    // Context = "上下文" = "执行环境的状态快照"
    // 它包含了某个操作过程中需要的所有相关信息，就像一个"工作台"，上面放着完成任务所需的所有工具和材料。
    std::unique_ptr<GutRenderForwardContext> m_forwardContext;

public:
    GUTRenderer(const nlohmann::json& config, const Logger& logger);
    virtual ~GUTRenderer();

    /// march the scene according to the given camera and composite the result into the given cuda arrays
    Status renderForward(const RenderParameters& params,
                         const tcnn::vec3* sensorRayOriginCudaPtr,
                         const tcnn::vec3* sensorRayDirectionCudaPtr,
                         float* worldHitCountCudaPtr,
                         float* worldHitDistanceCudaPtr,
                         tcnn::vec4* radianceDensityCudaPtr,
                         int* particlesVisibilityCudaPtr,
                         Parameters& parameters,
                         int cudaDeviceIndex,
                         cudaStream_t cudaStream);

    Status renderBackward(const RenderParameters& params,
                          const tcnn::vec3* sensorRayOriginCudaPtr,
                          const tcnn::vec3* sensorRayDirectionCudaPtr,
                          const float* worldHitDistanceCudaPtr,
                          const float* worldHitDistanceGradientCudaPtr,
                          const tcnn::vec4* radianceDensityCudaPtr,
                          const tcnn::vec4* radianceDensityGradientCudaPtr,
                          tcnn::vec3* worldRayOriginGradientCudaPtr,
                          tcnn::vec3* worldRayDirectionGradientCudaPtr,
                          Parameters& parameters,
                          int cudaDeviceIndex,
                          cudaStream_t cudaStream);
};

} // namespace threedgut
