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

// ========== 球谐函数模块 - 3D渲染中的方向性光照编码/解码核心 ==========
//
// 【模块作用】：
// 球谐函数(Spherical Harmonics, SH)是3D渲染中处理方向性光照的数学工具
// - 将复杂的方向性光照信息压缩为少量系数
// - 支持快速的视角相关效果计算（如光照、反射）
// - 特别适用于3D高斯溅射(3DGS)的颜色计算
//
// 【数学背景】：
// 球谐函数是定义在球面上的正交基函数族，类似于傅里叶变换中的正弦余弦函数
// 任何球面函数都可以表示为球谐基函数的线性组合：f(θ,φ) = Σ c_lm * Y_lm(θ,φ)
// 其中 c_lm 是球谐系数，Y_lm 是 l 阶 m 次的球谐基函数
//
// 【渲染应用】：
// 在3DGS中，每个高斯粒子存储一组球谐系数，用于计算视角相关的颜色
// 当从不同角度观察粒子时，通过球谐解码得到对应方向的颜色值

module sphericalHarmonics;

namespace sphericalHarmonics
{

// ========== 核心球谐解码函数 - 从系数重建方向性颜色 ==========
//
// 【功能】：根据观察方向和球谐系数，重建出该方向上的颜色值
// 【原理】：颜色(direction) = Σ coefficient_i * basisFunction_i(direction)
// 【优势】：
// - 存储高效：只需少量系数即可表示复杂的方向性变化
// - 计算快速：解码过程是简单的多项式计算
// - 连续平滑：球谐基函数保证了颜色的连续性
//
// [BackwardDifferentiable] - 支持自动微分，用于神经网络训练中的梯度反向传播
// let声明编译时已知的常量，可用于条件编译和优化
[BackwardDifferentiable]
vector<float, Dim> decode<let Dim : int, let MaxNumCoefficients : int>(
    no_diff in int degree,                                    // 球谐函数阶数(0-3)，决定表示精度，no_diff表示不参与微分
    in vector<float, Dim> coefficients[MaxNumCoefficients],   // 球谐系数数组，存储方向性光照信息
    in float3 direction)                                      // 观察方向向量(单位向量)，从粒子指向观察者
{
    // ========== 球谐基函数系数预计算 ==========
    // 这些常数是球谐基函数 Y_l^m(θ,φ) 的归一化系数，用于优化运行时计算
    // 数学来源：Y_l^m = N_l^m * P_l^m(cosθ) * e^(imφ)，其中 N_l^m 是归一化常数
    
    static const float C0 = 0.28209479177387814;            // Y₀⁰ 系数 = 1/(2√π) - 0阶：均匀分布
    static const float C1 = 0.4886025119029199;             // Y₁ᵐ 系数 = √(3/4π) - 1阶：线性梯度
    
    // 2阶球谐系数：Y₂⁻², Y₂⁻¹, Y₂⁰, Y₂¹, Y₂² 对应的归一化常数
    static const float C2[] = { 1.0925484305920792,         // Y₂⁻²: xy项
                               -1.0925484305920792,         // Y₂⁻¹: yz项  
                                0.31539156525252005,        // Y₂⁰: (2z²-x²-y²)项
                               -1.0925484305920792,         // Y₂¹: xz项
                                0.5462742152960396 };       // Y₂²: (x²-y²)项
    
    // 3阶球谐系数：Y₃⁻³ 到 Y₃³ 对应的归一化常数（7个系数）
    static const float C3[] = { -0.5900435899266435,        // Y₃⁻³: y(3x²-y²)项
                                 2.890611442640554,          // Y₃⁻²: xyz项
                                -0.4570457994644658,         // Y₃⁻¹: y(4z²-x²-y²)项
                                 0.3731763325901154,         // Y₃⁰: z(2z²-3x²-3y²)项
                                -0.4570457994644658,         // Y₃¹: x(4z²-x²-y²)项
                                 1.445305721320277,          // Y₃²: z(x²-y²)项
                                -0.5900435899266435 };       // Y₃³: x(x²-3y²)项

    // ========== 0阶球谐：常数项（环境光基础） ==========
    // Y₀⁰ = C0，表示各向同性的环境光照，不依赖于观察方向
    // 这是颜色的基础值，所有方向都能看到的"底色"
    vector<float, Dim> features = C0 * coefficients[0];
    
    // ========== 1阶球谐：线性项（基础方向性） ==========
    // Y₁ᵐ 项提供了基本的方向性变化，创建简单的梯度效果
    if (degree > 0)    
    {
        const float x = direction.x;    // 观察方向的x分量
        const float y = direction.y;    // 观察方向的y分量  
        const float z = direction.z;    // 观察方向的z分量
        
        // 1阶球谐基函数：Y₁⁻¹=-C1*y, Y₁⁰=C1*z, Y₁¹=-C1*x
        // 分别对应Y轴、Z轴、X轴方向的线性变化
        features = features - C1 * y * coefficients[1] +     // Y₁⁻¹项：Y方向梯度
                               C1 * z * coefficients[2] -     // Y₁⁰项：Z方向梯度  
                               C1 * x * coefficients[3];      // Y₁¹项：X方向梯度

        // ========== 2阶球谐：二次项（双线性变化和细节） ==========
        // Y₂ᵐ 项添加了更复杂的方向性变化，如"明暗分界线"效果
        if (degree > 1)    
        {
            // 预计算二次项，减少重复计算
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            
            // 2阶球谐基函数组合：
            // Y₂⁻²=C2[0]*xy, Y₂⁻¹=C2[1]*yz, Y₂⁰=C2[2]*(2z²-x²-y²), Y₂¹=C2[3]*xz, Y₂²=C2[4]*(x²-y²)
            features = features + C2[0] * xy * coefficients[4] +               // Y₂⁻²：XY交叉项
                                  C2[1] * yz * coefficients[5] +               // Y₂⁻¹：YZ交叉项  
                                  C2[2] * (2.0 * zz - xx - yy) * coefficients[6] + // Y₂⁰：Z轴偏向项
                                  C2[3] * xz * coefficients[7] +               // Y₂¹：XZ交叉项
                                  C2[4] * (xx - yy) * coefficients[8];         // Y₂²：X-Y对比项

            // ========== 3阶球谐：三次项（高频细节和复杂光照） ==========
            // Y₃ᵐ 项提供最精细的方向性变化，用于表示复杂的光照效果
            if (degree > 2)    
            {
                // 3阶球谐基函数：复杂的三次多项式组合
                // 这些项能够表示更加复杂和精细的方向性光照变化
                features = features + C3[0] * y * (3.0 * xx - yy) * coefficients[9] +           // Y₃⁻³
                                     C3[1] * xy * z * coefficients[10] +                        // Y₃⁻²  
                                     C3[2] * y * (4.0 * zz - xx - yy) * coefficients[11] +     // Y₃⁻¹
                                     C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[12] + // Y₃⁰
                                     C3[4] * x * (4.0 * zz - xx - yy) * coefficients[13] +     // Y₃¹
                                     C3[5] * z * (xx - yy) * coefficients[14] +                // Y₃²
                                     C3[6] * x * (xx - 3.0 * yy) * coefficients[15];          // Y₃³
            }
        }
    }
    
    // ========== 最终激活函数：确保颜色值非负 ==========
    // 应用修正的ReLU激活：features + 0.5 然后截断到非负值
    // +0.5 提供了一个基础偏移，避免完全黑色的区域
    // max() 确保最终颜色值为非负（物理上合理的亮度值）
    return max(features + 0.5, vector<float, Dim>(0));
}

}