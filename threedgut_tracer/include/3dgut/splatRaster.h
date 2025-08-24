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

// 用于3D点云渲染的Rasterizer
#pragma once

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/utils/logger.h>

#include <json/json.hpp> // JSON配置支持
#include <pybind11_json/pybind11_json.hpp> // python绑定的JSON支持

#include <deque>
#include <string>

using TTimestamp = int64_t; // 时间戳类型，用于时序渲染

class SplatRaster final { // 定义一个不可继承的类（final）
private:
    uint8_t m_logLevel;
    threedgut::Logger m_logger;

    std::unique_ptr<threedgut::GUTRenderer> m_renderer;

    threedgut::GUTRenderer::Parameters m_parameters; // 渲染参数，用于配置渲染行为

    class CudaTimer; // 声明，内部计时器类，用于计时CUDA核心操作性能，具体实现不在本文件中

    bool m_enableKernelTimings  = false; // 是否启用内核计时
    std::map<std::string, float> m_timings; // 计时器映射，存储不同操作的耗时

    const size_t m_maxNumTimers = 256; // We only keep the most recent 256 timers
    std::deque<std::shared_ptr<CudaTimer>> m_timers;

public:
    SplatRaster(const nlohmann::json& config); // 构造函数，用JSON配置初始化该类实例，可以灵活配置各种渲染参数和行为。

    ~SplatRaster();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    trace(uint32_t frameNumber, int numActiveFeatures,
          // Particles
          torch::Tensor particleDensity,
          torch::Tensor particleRadiance,
          // Rays
          torch::Tensor rayOrigin,
          torch::Tensor rayDirection,
          torch::Tensor rayTimestamp, // 光线时间戳
          // Sensor
          threedgut::TSensorModel sensor, // 传感器模型
          TTimestamp startTimestamp, // 开始时间戳
          TTimestamp endTimestamp, // 结束时间戳
          torch::Tensor sensorsStartPose, // 传感器起始姿态
          torch::Tensor sensorsEndPose); // 传感器结束姿态

    std::tuple<torch::Tensor, torch::Tensor>
    traceBwd(uint32_t frameNumber, int numActiveFeatures,
             // Particles
             torch::Tensor particleDensity,
             torch::Tensor particleRadiance,
             // Rays
             torch::Tensor rayOrigin,
             torch::Tensor rayDirection,
             torch::Tensor rayTimestamp,
             // Sensor
             threedgut::TSensorModel sensorModel,
             TTimestamp startTimestamp,
             TTimestamp endTimestamp,
             torch::Tensor sensorsStartPose,
             torch::Tensor sensorsEndPose,
             // Gradients
             torch::Tensor rayRadianceDensity,
             torch::Tensor rayRadianceDensityGradient,
             torch::Tensor rayHitDistance,
             torch::Tensor rayHitDistanceGradient);

    std::map<std::string, float>
    collectTimes(); // 收集并返回所有计时器的耗时数据
};
