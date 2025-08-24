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

#ifdef _WIN32 // 解决windows系统的兼容性问题，防止windows头文件定义的min/max宏与C++标准库冲突
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <3dgut/sensors/cameraModels.h>

threedgut::CameraModelParameters
fromOpenCVPinholeCameraModelParameters(std::array<uint64_t, 2> _resolution, // 图像分辨率 [宽，高]
                                       threedgut::TSensorModel::ShutterType shutter_type, // 快门类型：全局快门/滚动快门
                                       std::array<float, 2> principal_point, // 主点坐标 [u, v]
                                       std::array<float, 2> focal_length, // 焦距 [fx, fy]
                                       std::array<float, 6> radial_coeffs, // 径向畸变系数 [k1, k2, k3, k4, k5, k6]
                                       std::array<float, 2> tangential_coeffs, // 切向畸变系数 [p1, p2]
                                       std::array<float, 4> thin_prism_coeffs) { // 薄棱镜畸变系数 [s1, s2, s3, s4]
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVPinholeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec<6>), "[3dgut] typing size mismatch");
    static_assert(sizeof(tangential_coeffs) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(thin_prism_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvPinholeParams.nearFar          = tcnn::vec2{0.01f, 100.0f};
    params.ocvPinholeParams.principalPoint   = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvPinholeParams.focalLength      = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvPinholeParams.radialCoeffs     = *reinterpret_cast<const tcnn::vec<6>*>(radial_coeffs.data());
    params.ocvPinholeParams.tangentialCoeffs = *reinterpret_cast<const tcnn::vec2*>(tangential_coeffs.data());
    params.ocvPinholeParams.thinPrismCoeffs  = *reinterpret_cast<const tcnn::vec4*>(thin_prism_coeffs.data());
    return params;
}

threedgut::CameraModelParameters
fromOpenCVFisheyeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::TSensorModel::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 4> radial_coeffs,
                                       float max_angle) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVFisheyeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvFisheyeParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvFisheyeParams.focalLength    = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvFisheyeParams.radialCoeffs   = *reinterpret_cast<const tcnn::vec4*>(radial_coeffs.data());
    params.ocvFisheyeParams.maxAngle       = max_angle;
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // 宏定义入口，指定生成的Python模块名（宏TORCH_EXTENSION_NAME通常由编译系统指定）。

    pybind11::class_<SplatRaster>(m, "SplatRaster") // 无需访问内部变量
        .def(pybind11::init<const nlohmann::json&>()) // 给 Python 绑定类构造函数，形参为 JSON 格式配置。
        .def("trace", &SplatRaster::trace) 
        .def("trace_bwd", &SplatRaster::traceBwd)
        .def("collect_times", &SplatRaster::collectTimes);

    py::enum_<threedgut::TSensorModel::ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", threedgut::TSensorModel::ShutterType::RollingTopToBottomShutter)
        .value("ROLLING_LEFT_TO_RIGHT", threedgut::TSensorModel::ShutterType::RollingLeftToRightShutter)
        .value("ROLLING_BOTTOM_TO_TOP", threedgut::TSensorModel::ShutterType::RollingBottomToTopShutter)
        .value("ROLLING_RIGHT_TO_LEFT", threedgut::TSensorModel::ShutterType::RollingRightToLeftShutter)
        .value("GLOBAL", threedgut::TSensorModel::ShutterType::GlobalShutter);

    // 不能不绑定
    // 一方面fromOpenCVPinholeCameraModelParameters返回的参数类型是threedgut::CameraModelParameters
    // 另一方面SplatRaster::trace 接受 CameraModelParameters 作为参数
    py::class_<threedgut::CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>());

    m.def("fromOpenCVPinholeCameraModelParameters", &fromOpenCVPinholeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("tangential_coeffs"),
          py::arg("thin_prism_coeffs"));

    m.def("fromOpenCVFisheyeCameraModelParameters", &fromOpenCVFisheyeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("max_angle"));
}
