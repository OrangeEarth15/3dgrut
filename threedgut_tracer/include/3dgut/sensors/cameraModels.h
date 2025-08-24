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

// 两种主要的相机类型
// 1. 针孔相机模型（Pinhole Camera Model）- 标准相机
// 2. 鱼眼相机模型（Fisheye Camera Model）- 广角相机

#pragma once

#include <tiny-cuda-nn/vec.h>

namespace threedgut {

struct OpenCVPinholeProjectionParameters {
    tcnn::vec2 nearFar; // 近远平面距离，用于裁剪
    tcnn::vec2 principalPoint; // 主点，表示图像中心点相对于图像坐标系原点（通常是左上角）的偏移，单位是像素。它是相机成像光轴在图像上的投影点。
    tcnn::vec2 focalLength; // 焦距，分别对应于x轴和y轴的焦距，单位是像素
    tcnn::vec<6> radialCoeffs; // 径向畸变系数数组，有6个参数。径向畸变是光线经过镜头时沿径向发生的变形，常见的桶形畸变和枕形畸变属于此类，导致从图像中心到边缘的直线变曲线。
    tcnn::vec2 tangentialCoeffs; // 切向畸变系数，通常有两个参数。它描述镜头因制造不完美或安装不对称造成的非径向方向的图像变形。
    tcnn::vec4 thinPrismCoeffs; // 细棱镜畸变参数，包含四个系数，用于修正镜头产生的更复杂、细微的畸变，通常是光学系统额外的细节调整。
};

struct OpenCVFisheyeProjectionParameters {
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
    tcnn::vec4 radialCoeffs; // 鱼眼镜头的径向畸变参数，包含4个系数。鱼眼镜头的畸变通常更强烈，必须用专门的函数模型来描述和校正。
    float maxAngle; // 最大视角角度，表示鱼眼镜头能覆盖的最大入射光线角度，超过该角度的光线通常会被裁切或忽略，用来限定鱼眼镜头视场范围。
};

struct CameraModelParameters {
    enum ShutterType {
        RollingTopToBottomShutter,
        RollingLeftToRightShutter,
        RollingBottomToTopShutter,
        RollingRightToLeftShutter, // 滚动快门，因为传感器逐行曝光可能产生“果冻效应”。
        GlobalShutter // 全局快门，所有像素同时曝光
    } shutterType = GlobalShutter;

    enum ModelType {
        OpenCVPinholeModel,
        OpenCVFisheyeModel,
        EmptyModel,
        Unsupported
    } modelType = EmptyModel;

    union {
        OpenCVPinholeProjectionParameters ocvPinholeParams;
        OpenCVFisheyeProjectionParameters ocvFisheyeParams;
    };
};

} // namespace threedgut
