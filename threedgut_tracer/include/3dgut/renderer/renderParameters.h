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

#include <3dgut/sensors/sensors.h>
#include <3dgut/utils/bounding_box.h>

namespace threedgut {

using TTrackInstancePose = tcnn::vec<7>;

struct MemoryHandles {

    const uint64_t* handles;

    template <typename T>
    inline TCNN_HOST_DEVICE T* bufferPtr(int index) {
        return reinterpret_cast<T*>(handles[index]);
    }
};

struct RenderParameters {
    uint32_t id; // 渲染id
    tcnn::ivec2 resolution; // 分辨率
    float hitTransmittance; // 命中传输
    threedgut::BoundingBox objectAABB; // 对象包围盒
    // sensor.py
    // using TSensorPose = tcnn::vec<7>; // 3D position and 3D quaternion (x,y,z,w)
    //using TSensorModel = CameraModelParameters;

    TSensorModel sensorModel; // 传感器模型
    TSensorState sensorState; // 传感器状态
    // tcnn::mat4x3 colorCorrectionMatrix;
};

} // namespace threedgut
