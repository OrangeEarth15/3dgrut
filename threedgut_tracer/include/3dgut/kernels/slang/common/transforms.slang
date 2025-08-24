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

// ========== 几何变换模块 - 3D图形学中的核心数学变换 ==========
//
// 【模块作用】：
// 提供3D渲染管线中必需的几何变换功能，特别是旋转变换的高效实现
// - 四元数到旋转矩阵的转换（避免万向锁问题）
// - 支持自动微分的变换（用于神经网络训练）
// - 优化的数学实现（最小化计算复杂度）
//
// 【使用场景】：
// 在3D高斯溅射(3DGS)中，每个高斯粒子都有自己的旋转、位置和缩放
// 本模块提供了将存储的四元数旋转参数转换为实际变换矩阵的功能
//
// 【四元数优势】：
// - 存储紧凑：只需4个float vs 9个float（3x3矩阵）
// - 数值稳定：避免万向锁和插值问题
// - 计算高效：标准化和组合操作更简单

module transforms;

namespace transforms
{

// ========== 四元数到旋转矩阵转置的高效转换 ==========
//
// 【功能】：将四元数表示的旋转转换为旋转矩阵的转置
// 【为什么是转置】：
// - CUDA/GPU通常使用行主序(Row-Major)内存布局
// - Slang默认使用列主序(Column-Major)矩阵运算
// - 转置确保变换的正确性和性能
//
// 【数学原理】：
// 四元数 q = (w, x, y, z) 表示绕轴(x,y,z)旋转 2*arccos(w) 弧度
// 旋转矩阵 R = I + 2*sin(θ/2)*[v]× + 2*sin²(θ/2)*[v]×²
// 其中 [v]× 是反对称矩阵，θ 是旋转角度，v 是旋转轴
//
// [BackwardDifferentiable] - 支持自动微分，用于反向传播计算四元数参数的梯度
[BackwardDifferentiable]
float3x3 rotationMatrixTranspose(float4 quaternion)    // 输入：单位四元数 (w,x,y,z)
{
    // ========== 四元数分量提取和预计算 ==========
    // 四元数格式：quaternion = (w, x, y, z)
    // - w: 实部，表示旋转角度的余弦值
    // - x, y, z: 虚部，表示旋转轴方向
    //
    // 注意：预计算平方项和交叉项以减少重复乘法运算（性能优化）
    const float xx = quaternion.y * quaternion.y;    // x² - x分量的平方
    const float yy = quaternion.z * quaternion.z;    // y² - y分量的平方  
    const float zz = quaternion.w * quaternion.w;    // z² - z分量的平方
    const float xy = quaternion.y * quaternion.z;    // xy - x和y分量的乘积
    const float xz = quaternion.y * quaternion.w;    // xz - x和z分量的乘积
    const float yz = quaternion.z * quaternion.w;    // yz - y和z分量的乘积
    const float rx = quaternion.x * quaternion.y;    // wx - w和x分量的乘积（实部*x虚部）
    const float ry = quaternion.x * quaternion.z;    // wy - w和y分量的乘积（实部*y虚部）
    const float rz = quaternion.x * quaternion.w;    // wz - w和z分量的乘积（实部*z虚部）

    // ========== 旋转矩阵转置的构造 ==========
    // 使用四元数到旋转矩阵的标准公式，但直接构造转置形式
    // 
    // 标准旋转矩阵公式：R = I + 2*[q×] + 2*q₀*[q×]
    // 其中：
    // - I 是单位矩阵
    // - [q×] 是四元数虚部的反对称矩阵
    // - q₀ 是四元数的实部
    //
    // 优化后的显式矩阵元素计算（避免矩阵乘法）：
    // R^T = [ 1-2(y²+z²)   2(xy+wz)    2(xz-wy) ]
    //       [ 2(xy-wz)    1-2(x²+z²)   2(yz+wx) ]  
    //       [ 2(xz+wy)    2(yz-wx)    1-2(x²+y²)]
    return float3x3(
        // 第一行：变换后的X轴在原坐标系中的表示
        float3((1.f - 2.f * (yy + zz)),    // R^T[0,0] = 1 - 2(y²+z²)
               2.f * (xy + rz),             // R^T[0,1] = 2(xy + wz)  
               2.f * (xz - ry)),            // R^T[0,2] = 2(xz - wy)
        
        // 第二行：变换后的Y轴在原坐标系中的表示
        float3(2.f * (xy - rz),             // R^T[1,0] = 2(xy - wz)
               (1.f - 2.f * (xx + zz)),     // R^T[1,1] = 1 - 2(x²+z²)
               2.f * (yz + rx)),            // R^T[1,2] = 2(yz + wx)
        
        // 第三行：变换后的Z轴在原坐标系中的表示  
        float3(2.f * (xz + ry),             // R^T[2,0] = 2(xz + wy)
               2.f * (yz - rx),             // R^T[2,1] = 2(yz - wx)
               (1.f - 2.f * (xx + yy)))     // R^T[2,2] = 1 - 2(x²+y²)
    );
}

}
