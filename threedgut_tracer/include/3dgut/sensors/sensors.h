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

// clang-format off
#include <tiny-cuda-nn/common.h>    // TCNN库的通用定义，提供向量和矩阵运算
#include <3dgut/sensors/cameraModels.h>
// clang-format on

namespace threedgut {

// 向量切片工具函数 - 解决tcnn::vec::slice编译问题的替代方案
// 功能：从大向量中提取指定偏移和大小的子向量
// 模板参数：
//   Offset: 起始偏移量
//   OutSize: 输出向量大小
//   T: 数据类型（float/double等）
//   InSize: 输入向量大小
//   A: 内存对齐参数
template <uint32_t Offset, uint32_t OutSize, typename T, uint32_t InSize, size_t A>
inline TCNN_HOST_DEVICE tcnn::tvec<T, OutSize, A>& sliceVec(const tcnn::tvec<T, InSize, A>& vec) {
    // 通过指针偏移和类型转换实现向量切片
    // 例如：从vec7中提取前3个元素作为vec3
    return *(tcnn::tvec<T, OutSize, A>*)(vec.data() + Offset);
}

using TTimestamp = int64_t;


using TSensorPose = tcnn::vec<7>;

using TSensorModel = CameraModelParameters;

// ============= 传感器状态结构体 =============

// 传感器状态 - 描述传感器在一个时间段内的运动状态
// 支持运动模糊和rolling shutter效果的建模
struct TSensorState {
    TTimestamp startTimestamp;  // 曝光开始时间戳
    TSensorPose startPose;      // 曝光开始时的传感器姿态
    TTimestamp endTimestamp;    // 曝光结束时间戳  
    TSensorPose endPose;        // 曝光结束时的传感器姿态
    
    // 设计理念：
    // - 单一曝光可能跨越多个时间点（运动模糊）
    // - rolling shutter相机逐行曝光，不同行对应不同时间
    // - 通过起止姿态可以插值出任意时刻的精确姿态
};

// ============= 传感器姿态数学运算函数 =============

// 计算传感器姿态的逆变换
// 功能：从世界坐标系到传感器坐标系的变换矩阵
// 输入：传感器在世界坐标系中的姿态
// 输出：逆变换姿态，用于将世界坐标转换到传感器坐标
static inline TCNN_HOST_DEVICE TSensorPose sensorPoseInverse(const TSensorPose& pose) {
    // 第一步：提取四元数并转换为旋转矩阵
    // 注意：四元数存储顺序为[qx,qy,qz,qw]，但构造时需要[qw,qx,qy,qz]
    const tcnn::mat3 invRotation = tcnn::transpose(tcnn::to_mat3(tcnn::tquat{pose[6], pose[3], pose[4], pose[5]}));
    
    // 第二步：从逆旋转矩阵重新计算四元数
    const tcnn::quat invQuaternion = tcnn::quat{invRotation};
    
    // 第三步：构造逆姿态
    TSensorPose invPose;
    // 逆位置 = -R^T * t，其中R^T是逆旋转，t是原位置
    invPose.slice<0, 3>() = -1.0f * invRotation * pose.slice<0, 3>();
    // 逆四元数直接存储
    invPose.slice<3, 4>() = tcnn::vec4{invQuaternion.x, invQuaternion.y, invQuaternion.z, invQuaternion.w};
    
    return invPose;
}

// 在两个传感器姿态之间进行时间插值
// 功能：计算指定时间点的传感器姿态，用于处理运动模糊和rolling shutter
// 参数：
//   startPose: 起始姿态
//   endPose: 结束姿态  
//   relativeTime: 相对时间 [0,1]，0表示起始时间，1表示结束时间
// 返回：插值后的姿态
static inline TCNN_HOST_DEVICE TSensorPose interpolatedSensorPose(const TSensorPose& startPose,
                                                                  const TSensorPose& endPose,
                                                                  float relativeTime) {
    using namespace tcnn;

    // 第一步：四元数球面线性插值(SLERP)
    // SLERP保证旋转插值的平滑性和最短路径特性
    const quat interpolatedQuat = slerp(quat{startPose[6], startPose[3], startPose[4], startPose[5]},  // 起始四元数
                                        quat{endPose[6], endPose[3], endPose[4], endPose[5]},          // 结束四元数
                                        relativeTime);                                                 // 插值参数

    // 第二步：构造插值结果
    TSensorPose interpolated;
    // 位置线性插值：P(t) = (1-t)*P0 + t*P1
    interpolated.slice<0, 3>() = mix(startPose.slice<0, 3>(), endPose.slice<0, 3>(), relativeTime);
    // 四元数插值结果
    interpolated.slice<3, 4>() = vec4{interpolatedQuat.x, interpolatedQuat.y, interpolatedQuat.z, interpolatedQuat.w};

    return interpolated;
}

// 将传感器姿态转换为4x3变换矩阵
// 功能：生成从传感器坐标系到世界坐标系的变换矩阵
// 输出：4x3矩阵 [R|t]，其中R是3x3旋转矩阵，t是3x1平移向量
// 应用：将传感器坐标系中的点变换到世界坐标系
static inline TCNN_HOST_DEVICE tcnn::mat4x3 sensorPoseToMat(const TSensorPose& pose) {
    using namespace tcnn;

    // 第一步：从四元数提取旋转矩阵
    // 四元数到旋转矩阵的转换保证了旋转的正交性
    const mat3 rotation = to_mat3(quat{pose[6], pose[3], pose[4], pose[5]});
    
    // 第二步：构造齐次变换矩阵的前4行3列
    // 格式：[rotation[0], rotation[1], rotation[2], translation]
    // 即：[Rx, Ry, Rz, t]，其中Rx,Ry,Rz是旋转矩阵的列向量
    return mat4x3{rotation[0], rotation[1], rotation[2], pose.slice<0, 3>()};
}

} // namespace threedgut
