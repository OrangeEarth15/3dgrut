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

#include <3dgut/kernels/cuda/common/mathUtils.cuh>

namespace threedgut {

// ========== 3D高斯粒子系统核心数学库 ==========
// 📄 【系统概述】
// 本文件实现了3DGUT渲染管线的数学核心，包括：
// 1. 高斯粒子的几何变换和密度计算
// 2. 球谐函数（Spherical Harmonics）的编码/解码
// 3. 广义高斯核函数的响应计算
// 4. 前向和反向传播的完整数学实现
//
// 【数据结构设计】两种表示格式的分工：
// ParticleDensity - 存储和传输格式
// - GPU内存中的原始数据格式，紧凑存储
// - 网络传输、文件存储的标准格式
// - 内存占用小，利于带宽优化
// - 四元数表示旋转，避免万向锁问题
//
// ParticeFetchedDensity - 计算处理格式
// - GPU kernel内部的工作格式，计算友好
// - 直接用于矩阵运算，避免重复转换
// - 旋转矩阵已预计算，便于向量化计算
// - 优化内存布局，提升计算效率
// ========== 粒子密度存储格式 ==========
// 📦 紧凑的GPU内存存储格式，优化带宽使用
// 内存布局：3+1+4+3+1 = 12个float (48字节)
struct ParticleDensity {
    float3 position;   // 🌍 粒子在世界坐标系中的3D位置
    float density;     // 📊 粒子密度值 (对应不透明度α的基础值)
    float4 quaternion; // 🔄 旋转四元数 (w,x,y,z格式，避免万向锁)
    float3 scale;      // 📏 各轴向的缩放因子 (椭球体的半轴长度)
    float padding;     // 🔧 内存对齐填充，确保GPU访问效率
};

// ========== 粒子密度计算格式 ==========
// ⚡ GPU kernel内部的计算优化格式
// 内存布局：3+3+9+1 = 16个float (64字节)
struct ParticeFetchedDensity {
    float3 position;   // 🌍 粒子位置 (从存储格式直接复制)
    float3 scale;      // 📏 缩放因子 (从存储格式直接复制)
    float33 rotationT; // 🔄 预计算的3x3旋转矩阵转置 (避免运行时四元数转换)
    float density;     // 📊 密度值 (从存储格式直接复制)
};

// ========== 四元数到旋转矩阵转换 ==========
// 🔄 【数学原理】四元数到旋转矩阵的标准转换公式
// 输入：四元数 q = (w,x,y,z) 其中 w=real部分，(x,y,z)=虚部
// 输出：3x3旋转矩阵 R，满足 R * v = 旋转后的向量
//
// 【公式推导】基于四元数旋转公式 v' = q * v * q^(-1)
// 转换为矩阵形式，避免每次都进行四元数乘法运算
__forceinline__ __device__ void quaternionWXYZToMatrix(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float rx = r * x;
    const float ry = r * y;
    const float rz = r * z;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
    ret[1] = make_float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
    ret[2] = make_float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));
}

// ========== 球谐函数系数常量 ==========
// 📐 【数学背景】球谐函数Y_l^m的标准化系数
// 这些常量来自球谐函数的解析表达式，用于高效计算辐射传输
//
// 【物理意义】在3D渲染中，球谐函数用于表示:
// - 环境光照的频域表示
// - 材质BRDF的低频近似  
// - 视角相关的辐射度分布
static constexpr __device__ float SpHCoeff0   = 0.28209479177387814f;    // Y_0^0 系数 (常数项)
static constexpr __device__ float SpHCoeff1   = 0.4886025119029199f;     // Y_1^m 系数 (线性项)
static constexpr __device__ float SpHCoeff2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                                 -1.0925484305920792f, 0.5462742152960396f}; // Y_2^m 系数 (二次项)
static constexpr __device__ float SpHCoeff3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                                 -0.5900435899266435f, 1.445305721320277f, -0.5900435899266435f}; // Y_3^m 系数 (三次项)

// ========== 球谐函数辐射度计算 ==========
// 🌟 【核心功能】从球谐系数重建视角相关的辐射度（颜色）
// 
// 【数学原理】球谐重建公式：
// L(ω) = Σ_{l=0}^{deg} Σ_{m=-l}^{l} c_l^m * Y_l^m(ω)
// 其中：L(ω) = 方向ω的辐射度，c_l^m = 球谐系数，Y_l^m = 球谐基函数
//
// 【应用场景】
// - 3DGS中每个高斯粒子存储球谐系数
// - 根据观察方向动态计算颜色
// - 实现视角相关的外观效果（如反射、高光）
__device__ float3 radianceFromSpH(int deg,                        // 球谐函数的最大阶数 (0-3)
                                  const float3* sphCoefficients, // 球谐系数数组 [deg+1]个float3
                                  const float3& rdir,            // 观察方向向量 (归一化)
                                  bool clamped = true) {         // 是否限制输出为正值
    
    // 从球谐函数系数和方向计算颜色
    float3 rad = SpHCoeff0 * sphCoefficients[0];
    if (deg > 0) {
        const float3& dir = rdir;

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad           = rad - SpHCoeff1 * y * sphCoefficients[1] + SpHCoeff1 * z * sphCoefficients[2] -
              SpHCoeff1 * x * sphCoefficients[3];

        if (deg > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SpHCoeff2[0] * xy * sphCoefficients[4] + SpHCoeff2[1] * yz * sphCoefficients[5] +
                  SpHCoeff2[2] * (2.0f * zz - xx - yy) * sphCoefficients[6] +
                  SpHCoeff2[3] * xz * sphCoefficients[7] + SpHCoeff2[4] * (xx - yy) * sphCoefficients[8];

            if (deg > 2) {
                rad = rad + SpHCoeff3[0] * y * (3.0f * xx - yy) * sphCoefficients[9] +
                      SpHCoeff3[1] * xy * z * sphCoefficients[10] +
                      SpHCoeff3[2] * y * (4.0f * zz - xx - yy) * sphCoefficients[11] +
                      SpHCoeff3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12] +
                      SpHCoeff3[4] * x * (4.0f * zz - xx - yy) * sphCoefficients[13] +
                      SpHCoeff3[5] * z * (xx - yy) * sphCoefficients[14] +
                      SpHCoeff3[6] * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }
    rad += 0.5f;
    return clamped ? maxf3(rad, make_float3(0.0f)) : rad;
}

// ========== 球谐系数梯度累加 ==========
// ⚡ 【优化策略】支持原子操作的梯度累加，处理多线程写入冲突
//
// 【应用场景】
// - 反向传播时，多个光线可能击中同一个粒子
// - 需要将梯度安全地累加到该粒子的球谐系数上
// - Atomic=true: 使用原子操作，适用于并发写入
// - Atomic=false: 直接累加，适用于独占访问
template <bool Atomic = false>
static inline __device__ void addSphCoeffGrd(float3* sphCoefficientsGrad, // 球谐系数梯度数组
                                             int idx,                      // 系数索引
                                             const float3& val) {         // 要累加的梯度值
    if constexpr (Atomic) {
        // type atomicAdd(type* address, type val);
        atomicAdd(&sphCoefficientsGrad[idx].x, val.x);
        atomicAdd(&sphCoefficientsGrad[idx].y, val.y);
        atomicAdd(&sphCoefficientsGrad[idx].z, val.z);
    } else {
        sphCoefficientsGrad[idx] += val;
    }
}

// ========== 球谐函数反向传播（高层接口）==========
// 🔄 【反向传播】计算球谐系数的梯度，支持权重调制
//
// 【数学原理】链式法则应用：
// ∂L/∂c_l^m = ∂L/∂L(ω) * ∂L(ω)/∂c_l^m * weight
// 其中：∂L(ω)/∂c_l^m = Y_l^m(ω)
//
// 【功能特色】
// - 自动处理Clamp操作的梯度（正值mask）
// - 支持权重调制（如alpha混合权重）
// - 返回未截断的原始辐射度值
template <bool Atomic = false>
static inline __device__ float3 radianceFromSpHBwd(int deg,                          // 球谐函数最大阶数
                                                   const float3* sphCoefficients,   // 当前球谐系数
                                                   const float3& rdir,              // 观察方向
                                                   float weight,                     // 混合权重
                                                   const float3& rayRadGrd,         // 从上层传来的辐射度梯度
                                                   float3* sphCoefficientsGrad) {   // 输出：球谐系数梯度
    
    // radiance unclamped
    const float3 gradu = radianceFromSpH(deg, sphCoefficients, rdir, false); // 不clamp，不限制范围
    radianceFromSpHBwd<Atomic>(deg, rdir, rayRadGrd * weight, sphCoefficientsGrad, gradu);
    return make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                       gradu.y > 0.0f ? gradu.y : 0.0f,
                       gradu.z > 0.0f ? gradu.z : 0.0f);
}

// ========== 球谐函数反向传播（核心实现）==========
// 🧮 【核心算法】逐阶计算球谐系数的偏导数
//
// 【计算流程】
// 1. 处理Clamp操作的梯度mask（只对正值传播梯度）
// 2. 逐阶计算Y_l^m(ω)的值
// 3. 应用链式法则：∂L/∂c_l^m = rayRadGrd * Y_l^m(ω)
//
// 【优化特色】
// - 展开循环，减少分支预测开销
// - 复用中间计算结果（x²，y²，xy等）
// - 支持原子操作，处理并发写入
template <bool Atomic = false>
static inline __device__ void radianceFromSpHBwd(int deg,                      // 球谐函数最大阶数
                                                 const float3& rdir,           // 观察方向向量
                                                 const float3& rayRadGrd,      // 辐射度梯度
                                                 float3* sphCoefficientsGrad,  // 输出：系数梯度数组
                                                 const float3& gradu) {        // 未截断的辐射度值
    //
    float3 dL_dRGB = rayRadGrd; // 从上层传下来的颜色梯度
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0); // 只对正值进行梯度计算
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ---> rayRad = weight * grad = weight * explu(gsph0 * SpHCoeff0 +
    // 0.5,SHRadMinBound) with explu(x,a) = x if x > a else a*e(x-a)
    // ===> d_rayRad / d_gsph0 =   weight * SpHCoeff0
    addSphCoeffGrd(sphCoefficientsGrad, 0, SpHCoeff0 * dL_dRGB);

    if (deg > 0) {
        // const float3 sphdiru = gpos - rori;
        // const float3 sphdir = safe_normalize(sphdiru);
        const float3& sphdir = rdir;

        float x = sphdir.x;
        float y = sphdir.y;
        float z = sphdir.z;

        float dRGBdsh1 = -SpHCoeff1 * y;
        float dRGBdsh2 = SpHCoeff1 * z;
        float dRGBdsh3 = -SpHCoeff1 * x;

        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 3, dRGBdsh3 * dL_dRGB);

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SpHCoeff2[0] * xy;
            float dRGBdsh5 = SpHCoeff2[1] * yz;
            float dRGBdsh6 = SpHCoeff2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SpHCoeff2[3] * xz;
            float dRGBdsh8 = SpHCoeff2[4] * (xx - yy);

            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 8, dRGBdsh8 * dL_dRGB);

            if (deg > 2) {
                float dRGBdsh9  = SpHCoeff3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SpHCoeff3[1] * xy * z;
                float dRGBdsh11 = SpHCoeff3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SpHCoeff3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SpHCoeff3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SpHCoeff3[5] * z * (xx - yy);
                float dRGBdsh15 = SpHCoeff3[6] * x * (xx - 3.f * yy);

                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 15, dRGBdsh15 * dL_dRGB);
            }
        }
    }
}

// ========== 粒子密度参数提取 ==========
// 🔄 【格式转换】从存储格式转换为计算格式
//
// 【转换操作】
// 1. 直接复制位置、缩放、密度
// 2. 四元数 → 3x3旋转矩阵转换
// 3. 内存布局优化，便于后续计算
//
// 【性能考虑】
// - 一次性转换，避免重复计算
// - 旋转矩阵缓存，减少三角函数调用
static inline __device__ void fetchParticleDensity(
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    float3& particlePosition,
    float3& particleScale,
    float33& particleRotation,
    float& particleDensity) {
    const ParticleDensity particleData = particlesDensity[particleIdx];

    particlePosition = particleData.position;
    particleScale    = particleData.scale;
    quaternionWXYZToMatrix(particleData.quaternion, particleRotation);
    particleDensity = particleData.density;
}

// ========== 粒子球谐系数提取 ==========
// 📦 【数据获取】从全局数组中提取特定粒子的球谐系数
//
// 【数据布局】全局数组格式：
// [粒子0的系数0-RGB, 粒子0的系数1-RGB, ..., 粒子1的系数0-RGB, ...]
// 每个粒子有 PARTICLE_RADIANCE_NUM_COEFFS 个系数，每个系数3个分量（RGB）
//
// 【内存访问优化】
// - 连续内存读取，利用GPU缓存
// - 展开循环，减少分支开销
static inline __device__ void fetchParticleSphCoefficients(
    const int32_t particleIdx,
    const float* particlesSphCoefficients,
    float3* sphCoefficients) {
    const uint32_t particleOffset = particleIdx * PARTICLE_RADIANCE_NUM_COEFFS * 3;
#pragma unroll
    for (unsigned int i = 0; i < PARTICLE_RADIANCE_NUM_COEFFS; ++i) {
        const int offset   = i * 3;
        sphCoefficients[i] = make_float3(
            particlesSphCoefficients[particleOffset + offset + 0],
            particlesSphCoefficients[particleOffset + offset + 1],
            particlesSphCoefficients[particleOffset + offset + 2]);
    }
}


/*
在3D渲染中，粒子（如高斯椭球）需要一个函数来描述它们的"影响范围"和"强度分布"。广义高斯核函数就是这样一个数学工具，它决定了：
- 粒子在空间中如何衰减（从中心向外逐渐变弱）
- 衰减的"陡峭程度"（是平滑过渡还是急剧下降）

广义高斯核函数的一般形式是：
G(x) = exp(-a * |x|^b)
梯度：∂G/∂x = -a * b * |x|^(b-1) * sign(x) * G(x)
其中：
- a 是缩放系数，决定了衰减速度
- b 是阶数，决定了衰减的"陡峭程度"
- sign(x) 是符号函数，决定了梯度的方向


当 b=0 时，G(x) = 1，表示线性衰减
当 b=1 时，G(x) = exp(-a * |x|)，表示拉普拉斯衰减
当 b=2 时，G(x) = exp(-a * |x|^2)，表示高斯衰减
当 b>2 时，G(x) = exp(-a * |x|^b)，表示更陡峭的衰减

不同的 b 值会产生不同的衰减效果：
- b=0: 线性衰减，边缘清晰但不光滑
- b=1: 拉普拉斯衰减，适度光滑
- b=2: 高斯衰减，平滑衰减
- b>2: 更陡峭的衰减，更紧凑的支撑域

在3D渲染中，广义高斯核函数常用于：
- 粒子密度场的平滑过渡
- 粒子位置的平滑插值
- 粒子影响的平滑衰减

*/
// ========== 广义高斯核函数梯度计算 ==========
// 📊 【数学背景】广义高斯核函数的偏导数计算
//
// 【核函数定义】G(x) = exp(-a * |x|^b)
// 其中：a = 缩放系数，b = 广义高斯阶数，x = 归一化距离
//
// 【梯度公式】∂G/∂x = -a * b * |x|^(b-1) * sign(x) * G(x)
// 对于2范数距离：∂G/∂(x²) = -a * (b/2) * |x|^(b-2) * G(x)
//
// 【应用场景】反向传播中计算粒子影响函数的梯度
template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponseGrd(float grayDist,  // 归一化距离平方
                                                   float gres,       // 当前响应值G(x)
                                                   float gresGrd) {  // 从上层传来的梯度
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241 * (0.5f * 8);
        const float grayDistSq = grayDist * grayDist;
        return s * grayDistSq * grayDist * gres * gresGrd;
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185 * (0.5f * 5);
        return s * grayDist * sqrtf(grayDist) * gres * gresGrd;
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556 * (0.5f * 4);
        return s * grayDist * gres * gresGrd;
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667 * (0.5f * 3);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f * (0.5f * 1);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return gres > 0.f ? (0.5f * s * rsqrtf(grayDist)) * gresGrd : 0.f;
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return s * gres * gresGrd;
    }
    }
}

// ========== 广义高斯核函数响应计算 ==========
// 🎯 【核心算法】计算粒子对光线的影响强度
//
// 【数学模型】G(d) = exp(-s * d^n)
// 其中：d = 归一化距离，n = 广义高斯阶数，s = 缩放系数 = -4.5/3^n
//
// 【阶数效果】
// - n=0 (Linear): 线性衰减，硬边界
// - n=1 (Laplacian): 拉普拉斯分布，中等衰减
// - n=2 (Gaussian): 标准高斯分布，平滑衰减
// - n>2: 更陡峭的衰减，更紧凑的支撑域
//
// 【应用】确定粒子对光线的不透明度贡献
template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponse(float grayDist) {  // 输入：归一化距离平方
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241f;
        const float grayDistSq = grayDist * grayDist;
        return expf(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185f;
        return expf(s * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556f;
        return expf(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667f;
        return expf(s * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f;
        return expf(s * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return expf(s * grayDist);
    }
    }
}

// ========== 自适应广义高斯核函数 ==========
// ⚙️ 【高级功能】支持动态调制的广义高斯响应计算
//
// 【功能特色】
// 1. 最小响应阈值控制（LOD优化）
// 2. 响应调制因子（密度缩放）
// 3. 数值稳定性保护（避免log(0)）
//
// 【应用场景】
// - 自适应级别细节（LOD）渲染
// - 粒子密度的动态调节
// - 数值稳定性优化
template <int GeneralizedGaussianDegree = 2, bool clamped>
static inline __device__ float particleScaledResponse(float grayDist,            // 归一化距离平方
                                                     float modulatedMinResponse, // 调制后的最小响应
                                                     float responseModulation = 1.0f) { // 响应调制因子

    const float minResponse    = fminf(modulatedMinResponse / responseModulation, 0.97f);
    const float logMinResponse = clamped ? logf(minResponse) : modulatedMinResponse;

    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        const float grayDistSq = grayDist * grayDist;
        return expf(logMinResponse * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        return expf(logMinResponse * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        return expf(logMinResponse * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        return expf(logMinResponse * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        return expf(logMinResponse * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = (1.0f - minResponse) / 3.0f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        return expf(logMinResponse * grayDist);
    }
    }
}

// ========== 粒子击中前向处理 ==========
// 🎯 【完整流程】处理光线与粒子的交互（前向传播）
//
// 【处理步骤】
// 1. 几何变换：世界坐标 → 粒子局部坐标
// 2. 距离计算：光线到粒子中心的最短距离
// 3. 响应计算：基于距离的影响强度
// 4. 颜色混合：球谐函数计算视角相关颜色
// 5. 透射率更新：alpha混合公式
//
// 【模板参数】
// - ParticleKernelDegree: 广义高斯核函数阶数
// - SurfelPrimitive: 是否使用面片模式（vs 体积模式）
// - PerRayRadiance: 是否动态计算辐射度（vs 预计算）
template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false, bool PerRayRadiance = true>
__device__ inline bool processHitFwd(
    const float3& rayOrigin,                            // 世界坐标系中的光线起点
    const float3& rayDirection,                         // 世界坐标系中的光线方向（单位向量）
    const int32_t particleIdx,                          // 当前处理的粒子在数组中的索引
    const ParticleDensity* particlesDensity,            // 全局粒子数据数组指针
    const float* particlesSphCoefficients,              // 球谐系数数组指针（用于颜色计算）
    const float minParticleKernelDensity,               // 最小核函数密度阈值（早期剔除优化）
    const float minParticleAlpha,                       // 最小透明度阈值（早期剔除优化）
    const int32_t sphEvalDegree,                        // 球谐函数计算的最大度数
    float* transmittance,                               // [输入输出] 当前累积透射率
    float3* radiance,                                   // [输入输出] 当前累积辐射度（颜色）
    float* depth,                                       // [输入输出] 当前累积深度值
    float3* normal) {                                   // [输入输出] 当前累积法向量（可选）

    // =============== 步骤1: 粒子数据提取 ===============
    // 从紧凑存储格式中提取粒子的几何参数
    float3 particlePosition;      // 粒子在世界坐标系中的中心位置
    float3 particleScale;         // 粒子椭球的三轴缩放系数
    float33 particleRotation;     // 粒子的3x3旋转矩阵（从四元数转换而来）
    float particleDensity;        // 粒子的基础密度值

    // 调用数据提取函数：解包ParticleDensity结构体，并将四元数转换为旋转矩阵
    fetchParticleDensity(
        particleIdx,              // 粒子索引
        particlesDensity,         // 粒子数据数组
        particlePosition,         // [输出] 粒子位置
        particleScale,            // [输出] 粒子缩放
        particleRotation,         // [输出] 旋转矩阵
        particleDensity);         // [输出] 密度值

    // =============== 步骤2: 坐标系变换 ===============
    // 【核心思想】将复杂的椭球-光线相交问题转换为简单的单位球-光线相交问题
    // 变换链：世界坐标 → 粒子坐标 → 标准化椭球 → 单位球

    // 2.1 计算缩放逆变换系数
    // 目的：将椭球在各轴向上标准化为单位球
    const float3 giscl = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    
    // 2.2 平移变换：将光线起点移动到以粒子为原点的坐标系
    const float3 gposc = (rayOrigin - particlePosition);
    
    // 2.3 旋转变换：将相对位置向量旋转到粒子的局部坐标系
    // 使用旋转矩阵将世界坐标转换为粒子局部坐标
    const float3 gposcr = (gposc * particleRotation);
    
    // 2.4 缩放变换：在局部坐标系中应用逆缩放，将椭球变为单位球
    const float3 gro = giscl * gposcr;

    // 2.5 对光线方向进行相同的变换（注意：方向向量不需要平移）
    const float3 rayDirR = rayDirection * particleRotation;  // 旋转光线方向
    const float3 grdu = giscl * rayDirR;                     // 应用逆缩放
    const float3 grd = safe_normalize(grdu);                 // 重新标准化方向向量

    // =============== 步骤3: 距离计算 ===============
    // 根据渲染模式选择不同的距离计算方法
    const float3 gcrod = SurfelPrimitive ? 
        // 面片模式：计算光线与z=0平面的交点，然后计算交点到原点的距离
        // 数学公式：交点 = 光线起点 + 方向 * t，其中 t = -起点.z / 方向.z
        gro + grd * (-gro.z / grd.z) : 
        // 体积模式：使用向量叉积计算光线到粒子中心的最短距离向量
        // 数学公式：距离向量 = 方向 × 起点向量
        cross(grd, gro);
    
    // 计算距离的平方（避免开方运算以提升性能）
    const float grayDist = dot(gcrod, gcrod);

    // =============== 步骤4: 响应计算和早期剔除 ===============
    // 4.1 使用广义高斯核函数计算粒子对光线的影响强度
    // 不同的ParticleKernelDegree会产生不同形状的衰减曲线
    const float gres = particleResponse<ParticleKernelDegree>(grayDist);
    
    // 4.2 计算最终的alpha值：响应强度 × 粒子密度，并限制在[0, 0.99]范围内
    // 限制上限是为了避免完全不透明导致的数值问题
    const float galpha = fminf(0.99f, gres * particleDensity);

    // 4.3 早期剔除测试：如果响应太小或alpha太小就跳过后续计算
    // 这是重要的性能优化，避免处理对最终结果贡献微小的粒子
    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    
    if (acceptHit) {  // 只有通过剔除测试的粒子才进行完整计算
        
        // =============== 步骤5: 权重和深度计算 ===============
        // 5.1 计算这个粒子对最终像素的贡献权重
        // 权重 = 粒子透明度 × 当前累积透射率
        const float weight = galpha * (*transmittance);

        // 5.2 计算光线到粒子中心投影点的实际距离（用于深度缓冲）
        const float3 grds = particleScale * grd * (SurfelPrimitive ? 
            // 面片模式：使用z轴投影距离
            -gro.z / grd.z : 
            // 体积模式：使用光线方向在起点向量上的投影
            dot(grd, -1 * gro));
        const float hitT = sqrtf(dot(grds, grds));  // 计算距离的模长

        // =============== 步骤6: 颜色计算 ===============
        if constexpr (PerRayRadiance) {
            // 动态模式：实时计算球谐函数以获得视角相关的颜色
            // 6.1 从全局数组中提取当前粒子的球谐系数
            float3 sphCoefficients[PARTICLE_RADIANCE_NUM_COEFFS];
            fetchParticleSphCoefficients(
                particleIdx,
                particlesSphCoefficients,
                &sphCoefficients[0]);
            
            // 6.2 根据光线方向计算球谐函数值，得到视角相关的颜色
            *radiance += weight * radianceFromSpH(sphEvalDegree, &sphCoefficients[0], rayDirection);
        } else {
            // 预计算模式：直接使用预存储的颜色值（性能更高但视角无关）
            *radiance += weight * reinterpret_cast<const float3*>(particlesSphCoefficients)[particleIdx];
        }

        // =============== 步骤7: 状态更新 ===============
        // 7.1 更新透射率：实现标准的alpha混合公式 T_new = T_old × (1 - α)
        *transmittance *= (1 - galpha);
        
        // 7.2 累加深度值：使用权重平均计算最终深度
        *depth += hitT * weight;

        // 7.3 法向量计算（如果需要）
        if (normal) {
            constexpr float ellispoidSqRadius = 9.0f;  // 椭球半径平方的阈值常数
            // 将粒子缩放因子变换回世界坐标系
            const float3 particleScaleRotated = (particleRotation * particleScale);
            
            *normal += weight * (SurfelPrimitive ? 
                // 面片模式：法向量始终沿z轴方向，根据视角决定正负
                make_float3(0, 0, (grd.z > 0 ? 1 : -1) * particleScaleRotated.z) : 
                // 体积模式：计算椭球表面的真实法向量
                // 公式：法向量 = normalize(表面点 - 椭球中心) × 缩放因子
                safe_normalize((gro + grd * (dot(grd, -1 * gro) - sqrtf(ellispoidSqRadius - grayDist))) * particleScaleRotated));
        }
    }

    // 返回是否有有效贡献（用于统计和调试）
    return acceptHit;
}

// ========== 自定义粒子相交检测 ==========
// 🔍 【几何算法】光线-椭球相交的高效检测
//
// 【算法原理】
// 1. 将光线变换到粒子局部坐标系
// 2. 在局部坐标系中椭球变为单位球
// 3. 计算光线到球心的最短距离
// 4. 检测距离是否在阈值范围内
//
// 【优化特色】
// - 避免完整的二次方程求解
// - 提前剔除不可能相交的情况
// - 支持距离范围限制（近远平面）
//
// 【应用场景】
// - 光线跟踪中的相交检测
// - 碰撞检测系统
// - 可见性测试
__device__ inline bool intersectCustomParticle(
    const float3& rayOrigin,                     // 光线起点（世界坐标）
    const float3& rayDirection,                  // 光线方向（世界坐标，单位向量）
    const int32_t particleIdx,                   // 要检测的粒子索引
    const ParticleDensity* particlesDensity,     // 粒子数据数组
    const float minHitDistance,                  // 最近击中距离（近平面）
    const float maxHitDistance,                  // 最远击中距离（远平面）
    const float maxParticleSquaredDistance,      // 粒子影响范围的平方距离阈值
    float& hitDistance) {                        // [输出] 击中点到光线起点的距离

    // =============== 步骤1: 粒子数据提取 ===============
    float3 particlePosition;      // 粒子中心位置
    float3 particleScale;         // 粒子椭球缩放系数
    float33 particleRotation;     // 粒子旋转矩阵
    float particleDensity;        // 粒子密度（此函数中未使用，但API保持一致性）
    
    // 提取粒子的几何参数
    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleRotation,
        particleDensity);

    // =============== 步骤2: 坐标系变换 ===============
    // 将光线从世界坐标系变换到粒子局部坐标系，使椭球变为单位球
    
    // 2.1 计算逆缩放系数
    const float3 giscl = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    
    // 2.2 平移：将光线起点移动到以粒子为原点的坐标系
    const float3 gposc = (rayOrigin - particlePosition);
    
    // 2.3 旋转：将相对位置向量旋转到粒子局部坐标系
    const float3 gposcr = (gposc * particleRotation);
    
    // 2.4 缩放：在局部坐标系中应用逆缩放，将椭球标准化为单位球
    const float3 gro = giscl * gposcr;

    // 2.5 对光线方向进行相同的变换
    const float3 rayDirR = rayDirection * particleRotation;  // 旋转方向
    const float3 grdu = giscl * rayDirR;                     // 应用逆缩放
    const float3 grd = safe_normalize(grdu);                 // 重新标准化

    // =============== 步骤3: 计算光线上最接近粒子中心的点 ===============
    // 【数学原理】对于光线 P(t) = 起点 + t * 方向，找到使 |P(t) - 中心| 最小的 t
    // 通过求导并令导数为0：d/dt |P(t)|² = 0
    // 解得：t = -dot(方向, 起点向量) / |方向|²
    // 由于方向是单位向量，所以分母为1
    const float grp = -dot(grd, gro);  // 投影参数 t

    // =============== 步骤4: 计算击中距离 ===============
    // 将局部坐标系中的距离转换回世界坐标系
    const float3 grds = particleScale * grd * grp;          // 应用缩放变换
    hitDistance = (grp < 0.f ? -1.f : 1.f) * sqrtf(dot(grds, grds));  // 计算实际距离，保留符号信息

    // =============== 步骤5: 距离范围检测 ===============
    // 检查击中点是否在指定的距离范围内（近平面到远平面之间）
    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        
        // =============== 步骤6: 精确相交检测 ===============
        // 计算光线到粒子中心的最短距离
        const float3 gcrod = cross(grd, gro);                    // 使用叉积计算垂直距离向量
        const float grayDist = dot(gcrod, gcrod);                // 距离的平方
        
        // 检查距离是否在粒子的影响范围内
        return (grayDist < maxParticleSquaredDistance);
    }
    
    // 距离超出范围，无相交
    return false;
}

// ========== 实例粒子相交检测 ==========
// ⚡ 【优化版本】针对已变换到粒子空间的光线
//
// 【应用场景】
// - 光线已经通过逆变换到粒子局部空间
// - 粒子参数已经归一化处理
// - 快速批量相交检测
//
// 【算法简化】
// - 跳过坐标变换步骤
// - 直接在局部空间计算
// - 更高的计算效率
//
// 【性能优势】
// - 避免重复的矩阵变换
// - 减少内存访问
// - 适用于已知变换的批量处理
__device__ inline bool intersectInstanceParticle(
    const float3& particleRayOrigin,           // 已变换到粒子局部空间的光线起点
    const float3& particleRayDirection,        // 已变换到粒子局部空间的光线方向
    const int32_t particleIdx,                 // 粒子索引（此版本中未使用）
    const float minHitDistance,                // 最近击中距离阈值
    const float maxHitDistance,                // 最远击中距离阈值
    const float maxParticleSquaredDistance,    // 粒子影响范围的平方距离阈值
    float& hitDistance) {                      // [输出] 击中距离

    // =============== 步骤1: 计算光线上最接近原点的点 ===============
    // 【数学原理】对于光线 P(t) = 起点 + t * 方向，找到使 |P(t)| 最小的参数 t
    // 通过最小化 |起点 + t * 方向|²：
    // d/dt [|起点|² + 2t * dot(起点,方向) + t² * |方向|²] = 0
    // 解得：t = -dot(起点, 方向) / |方向|²
    
    const float numerator = -dot(particleRayOrigin, particleRayDirection);     // 分子：-dot(起点,方向)
    const float denominator = 1.f / dot(particleRayDirection, particleRayDirection);  // 分母：1/|方向|²
    
    // 计算击中参数 t（光线上最接近原点的点的参数）
    hitDistance = numerator * denominator;

    // =============== 步骤2: 距离范围检测 ===============
    // 检查击中点是否在有效的距离范围内
    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        
        // =============== 步骤3: 精确相交检测 ===============
        // 计算光线到原点的最短距离的平方
        // 【几何原理】叉积的模长等于两个向量构成的平行四边形面积
        // 对于单位向量，叉积的模长等于两向量间的垂直距离
        
        // 3.1 标准化光线方向（确保叉积计算的准确性）
        const float3 normalizedDirection = safe_normalize(particleRayDirection);
        
        // 3.2 使用叉积计算垂直距离向量
        const float3 gcrod = cross(normalizedDirection, particleRayOrigin);
        
        // 3.3 计算距离平方并应用方向长度校正
        // 乘以 denominator 是为了补偿非单位方向向量的影响
        const float distanceSquared = dot(gcrod, gcrod) * denominator;
        
        // 3.4 检查距离是否在粒子的影响范围内
        return (distanceSquared < maxParticleSquaredDistance);
    }
    
    // 击中点不在有效距离范围内
    return false;
}

// ========== 粒子击中反向处理 ==========
// 🔄 【反向传播】计算所有参数的梯度（用于神经网络训练）
//
// 【梯度计算链】
// 1. 颜色梯度 → 球谐系数梯度
// 2. 透射率梯度 → 不透明度梯度
// 3. 深度梯度 → 位置/旋转/缩放梯度
// 4. 响应梯度 → 几何参数梯度
//
// 【数学复杂性】
// - 涉及多重链式法则
// - 矩阵微分（旋转矩阵梯度）
// - 四元数微分（旋转参数梯度）
// - 所有梯度需要原子累加（多线程安全）
//
// 【应用】3DGS模型的端到端训练
//
// 【核心思想】
// 反向传播是前向传播的逆过程，通过链式法则将输出梯度
// 传播到所有可训练参数上，实现端到端的神经网络训练
template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false, bool PerRayRadiance = true>
__device__ inline void processHitBwd(
    const float3& rayOrigin,                      // 光线起点（世界坐标）
    const float3& rayDirection,                   // 光线方向（世界坐标）
    uint32_t particleIdx,                         // 粒子索引
    const ParticleDensity& particleData,          // 粒子参数（前向传播时的值）
    ParticleDensity* particleDensityGradPtr,      // [输出] 粒子参数的梯度
    const float* particleRadiancePtr,             // 球谐系数（前向传播时的值）
    float* particleRadianceGradPtr,               // [输出] 球谐系数的梯度
    float minParticleKernelDensity,               // 最小核函数密度阈值
    float minParticleAlpha,                       // 最小透明度阈值
    float minTransmittance,                       // 最小透射率阈值（早停条件）
    int32_t sphEvalDegree,                        // 球谐函数计算度数
    float integratedTransmittance,                // 累积透射率（用于梯度计算）
    float& transmittance,                         // [输入输出] 当前透射率
    float transmittanceGrad,                      // 透射率的梯度（从输出反传）
    float3 integratedRadiance,                    // 累积辐射度（用于梯度计算）
    float3& radiance,                             // [输入输出] 当前辐射度
    float3 radianceGrad,                          // 辐射度的梯度（从输出反传）
    float integratedDepth,                        // 累积深度（用于梯度计算）
    float& depth,                                 // [输入输出] 当前深度
    float depthGrad) {                            // 深度的梯度（从输出反传）
    // =============== 步骤1: 粒子参数提取和前向计算重现 ===============
    // 【重要说明】反向传播需要重现前向传播的所有中间计算结果
    // 这些值将用于计算各个参数的梯度
    
    float3 particlePosition;      // 粒子位置
    float3 gscl;                  // 粒子缩放系数
    float33 particleRotation;     // 粒子旋转矩阵
    float particleDensity;        // 粒子密度
    float4 grot;                  // 原始四元数（用于梯度计算）

    {   // 数据提取块：从输入参数中提取粒子的几何属性
        particlePosition = particleData.position;
        gscl             = particleData.scale;
        grot             = particleData.quaternion;
        quaternionWXYZToMatrix(grot, particleRotation);  // 四元数转旋转矩阵
        particleDensity  = particleData.density;
    }

    // =============== 步骤2: 坐标变换（重现前向传播的几何计算） ===============
    // 这些计算与前向传播完全一致，用于获得梯度计算所需的中间变量
    
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);  // 逆缩放系数
    const float3 gposc   = (rayOrigin - particlePosition);                     // 平移变换
    const float3 gposcr  = (gposc * particleRotation);                        // 旋转变换
    const float3 gro     = giscl * gposcr;                                     // 缩放标准化
    const float3 rayDirR = rayDirection * particleRotation;                    // 光线方向变换
    const float3 grdu    = giscl * rayDirR;                                    // 方向标准化
    const float3 grd     = safe_normalize(grdu);                               // 最终方向单位向量
    
    // 距离计算（根据渲染模式选择）
    const float3 gcrod = SurfelPrimitive ? 
        gro + grd * (-gro.z / grd.z) :  // 面片模式
        cross(grd, gro);                // 体积模式
    const float grayDist = dot(gcrod, gcrod);  // 距离平方

    // =============== 步骤3: 响应计算和早期剔除检测 ===============
    // 重现前向传播的响应计算，用于确定是否需要计算梯度
    
    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);  // 核函数响应
    const float galpha = fminf(0.99f, gres * particleDensity);              // 最终alpha值

    // 只有通过早期剔除测试的粒子才计算梯度（与前向传播逻辑一致）
    if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha)) {

        // =============== 步骤4: 深度和权重计算（重现前向传播） ===============
        
        // 4.1 计算击中距离相关的中间变量
        const float3 grdd   = grd * (SurfelPrimitive ? 
            -gro.z / grd.z :           // 面片模式：z轴投影
            dot(grd, -1 * gro));       // 体积模式：方向投影
        const float3 grds   = gscl * grdd;          // 应用缩放变换
        const float gsqdist = dot(grds, grds);      // 距离平方
        const float gdist   = sqrtf(gsqdist);       // 实际距离

        // 4.2 计算权重和透射率
        const float weight = galpha * transmittance;           // 当前粒子的贡献权重
        const float nextTransmit = (1 - galpha) * transmittance;  // 更新后的透射率

        // =============== 步骤5: 深度梯度计算 ===============
        // 【数学原理】深度的前向公式：depth += weight * gdist
        // 反向传播时需要计算：∂L/∂weight 和 ∂L/∂gdist
        
        // 5.1 更新深度值（重现前向传播）
        depth += weight * gdist;
        
        // 5.2 计算剩余深度（用于梯度计算）
        // 【物理含义】当前透射率小于阈值时，后续光线贡献可忽略
        const float residualHitT = fmaxf(
            (nextTransmit <= minTransmittance ? 0 : (integratedDepth - depth) / nextTransmit), 
            0);

        // =============== 步骤6: 深度对alpha的梯度计算 ===============
        // 【数学推导】深度的完整公式：
        // hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        //
        // 【链式法则】∂L/∂galpha = ∂L/∂hitT * ∂hitT/∂galpha
        // ∂hitT/∂galpha = gdist * prevTrm - residualHitT * prevTrm
        //                = (gdist - residualHitT) * prevTrm
        const float galphaRayHitGrd = (gdist - residualHitT) * transmittance * depthGrad;

        // =============== 步骤7: 深度对几何参数的梯度计算 ===============
        // 【数学推导】深度对距离平方的梯度：
        // ∂hitT/∂gsqdist = ∂hitT/∂gdist * ∂gdist/∂gsqdist
        //                 = weight * (1 / (2*sqrt(gsqdist)))
        //                 = weight / (2*gdist)
        //
        // 【距离向量的梯度】∂gsqdist/∂grds = 2 * grds
        const float3 grdsRayHitGrd = gsqdist > 0.0f ? 
            ((2 * grds * weight) / (2 * gdist)) * depthGrad : 
            make_float3(0.0f);

        // =============== 步骤8: 几何变换的反向梯度传播 ===============
        // 【变换链条】grds = gscl * grdd = gscl * grd * dot(grd, -gro)
        //
        // 8.1 对缩放参数的梯度：∂grds/∂gscl = grdd
        const float3 gsclRayHitGrd = grdd * grdsRayHitGrd;
        
        // 8.2 对方向向量的梯度：∂grds/∂grd（需要处理点积的梯度）
        // 【复杂推导】由于grds依赖于grd的点积，需要考虑两个贡献：
        // - 直接贡献：gscl * dot(grd, -gro)
        // - 点积贡献：gscl * grd * (-gro) 的各分量梯度
        const float3 grdRayHitGrd = -gscl * make_float3(
            2 * grd.x * gro.x + grd.y * gro.y + grd.z * gro.z,  // x分量的偏导数
            grd.x * gro.x + 2 * grd.y * gro.y + grd.z * gro.z,  // y分量的偏导数
            grd.x * gro.x + grd.y * gro.y + 2 * grd.z * gro.z   // z分量的偏导数
        ) * grdsRayHitGrd;
        
        // 8.3 对局部位置的梯度：∂grds/∂gro = -gscl * grd * grd（点积梯度）
        const float3 groRayHitGrd = -gscl * grd * grd * grdsRayHitGrd;

        // =============== 步骤9: 透射率梯度计算 ===============
        // 【数学推导】透射率的前向公式：
        // rayDensity = 1 - prevTransmittance * (1-galpha) * nextTransmittance
        //            = 1 - (1-galpha) * prevTransmittance * nextTransmittance
        //
        // 【链式法则】∂L/∂galpha += ∂L/∂rayDensity * ∂rayDensity/∂galpha
        // ∂rayDensity/∂galpha = prevTransmittance * nextTransmittance = residualTransmittance
        
        // 9.1 计算剩余透射率（数值稳定性处理）
        const float residualTrm = galpha < 0.999999f ? 
            integratedTransmittance / (1 - galpha) :  // 标准情况
            transmittance;                            // 避免除零的边界情况
        
        // 9.2 透射率对alpha的梯度贡献
        const float galphaRayDnsGrd = residualTrm * (-transmittanceGrad);

        // =============== 步骤10: 球谐函数和颜色梯度计算 ===============
        // 【目标】计算辐射度对球谐系数和几何参数的梯度
        
        float3 grad;  // 存储球谐函数对视角方向的梯度
        
        if constexpr (PerRayRadiance) {
            // 动态模式：实时计算球谐函数的反向传播
            
            // 10.1 获取球谐系数（重现前向传播）
            float3 sphCoefficients[PARTICLE_RADIANCE_NUM_COEFFS];
            fetchParticleSphCoefficients(
                particleIdx,
                particleRadiancePtr,
                &sphCoefficients[0]);
            
            // 10.2 球谐函数的反向传播
            // 【复杂计算】同时计算：
            // - 辐射度对球谐系数的梯度（存储到particleRadianceGradPtr）
            // - 辐射度对视角方向的梯度（返回值grad，用于位置梯度计算）
            grad = radianceFromSpHBwd<true>(
                sphEvalDegree, 
                &sphCoefficients[0], 
                rayDirection,                   // 视角方向
                weight,                         // 权重系数
                radianceGrad,                   // 输入的辐射度梯度
                (float3*)&particleRadianceGradPtr[particleIdx * PARTICLE_RADIANCE_NUM_COEFFS * 3]);  // 输出：系数梯度
        } else {
            // 预计算模式：直接使用预存储的颜色值
            
            // 10.1 获取预计算的颜色值
            grad = reinterpret_cast<const float3*>(particleRadiancePtr)[0];
            
            // 10.2 直接计算颜色梯度（无球谐计算）
            particleRadianceGradPtr[0] = radianceGrad.x * weight;  // R分量梯度
            particleRadianceGradPtr[1] = radianceGrad.y * weight;  // G分量梯度
            particleRadianceGradPtr[2] = radianceGrad.z * weight;  // B分量梯度
        }

        // =============== 步骤11: 辐射度梯度的后续处理 ===============
        // 【数学推导】辐射度的前向公式：
        // rayRadiance = accumulatedRayRad + weight * rayRad + (1-galpha) * transmit * residualRayRad
        
        // 11.1 更新当前辐射度（重现前向传播）
        const float3 rayRad = weight * grad;
        radiance += rayRad;
        
        // 11.2 计算剩余辐射度（用于梯度计算）
        const float3 residualRayRad = maxf3(
            (nextTransmit <= minTransmittance ? 
                make_float3(0) : 
                (integratedRadiance - radiance) / nextTransmit),
            make_float3(0));

        // =============== 步骤12: 粒子密度梯度的总和计算 ===============
        // 【复杂推导】密度梯度需要汇总多个来源的贡献：
        // 1. 深度梯度的贡献（galphaRayHitGrd）
        // 2. 透射率梯度的贡献（galphaRayDnsGrd）
        // 3. 颜色梯度的贡献（通过alpha对颜色的影响）
        //
        // 【数学公式】对于每个颜色分量：
        // ∂rayRadiance/∂density = gres * transmittance * (grad - residualRayRad)
        // 其中 grad 是球谐函数值，residualRayRad 是剩余辐射度
        
        particleDensityGradPtr->density = gres * (
            galphaRayHitGrd +                                                     // 深度梯度贡献
            galphaRayDnsGrd +                                                     // 透射率梯度贡献
            transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +        // R分量颜色梯度贡献
            transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +        // G分量颜色梯度贡献
            transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);        // B分量颜色梯度贡献

        // =============== 步骤13: 核函数响应梯度的计算 ===============
        // 【数学推导】响应函数对透射率和颜色的影响：
        // ∂rayDensity/∂gres = residualTransmittance * particleDensity
        // ∂rayRadiance/∂gres = particleDensity * transmittance * (grad - residualRayRad)
        //
        // 【总梯度汇总】响应函数梯度来自三个方面：
        // 1. 深度梯度通过alpha的影响
        // 2. 透射率梯度通过alpha的影响
        // 3. 颜色梯度通过alpha的影响
        
        const float gresGrd = particleDensity * (
            galphaRayHitGrd +                                                     // 深度→alpha→响应 的梯度链
            galphaRayDnsGrd +                                                     // 透射率→alpha→响应 的梯度链
            transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +        // 颜色R→alpha→响应 的梯度链
            transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +        // 颜色G→alpha→响应 的梯度链
            transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);        // 颜色B→alpha→响应 的梯度链

        // =============== 步骤14: 距离梯度的计算 ===============
        // 【数学推导】广义高斯核函数的梯度：
        // 对于 gres = exp(-s * grayDist^n)：
        // ∂gres/∂grayDist = -s * n * grayDist^(n-1) * exp(-s * grayDist^n)
        //                 = -s * n * grayDist^(n-1) * gres
        //
        // 【具体实现】particleResponseGrd函数封装了不同阶数的梯度计算
        const float grayDistGrd = particleResponseGrd<ParticleKernelDegree>(grayDist, gres, gresGrd);

        // =============== 步骤15: 几何距离的反向梯度传播 ===============
        // 【分支处理】根据渲染模式选择不同的梯度计算方法
        
        float3 grdGrd, groGrd;  // 存储对方向向量和位置向量的梯度
        
        if (SurfelPrimitive) {
            // ========== 面片模式的梯度计算 ==========
            // 【几何原理】面片模式将粒子视为与z=0平面相交的椭圆
            
            // 15.1 定义面片法向量和相关计算
            const float3 surfelNm    = make_float3(0, 0, 1);        // 面片法向量（z轴）
            const float doSurfelGro  = dot(surfelNm, gro);          // 起点在法向量上的投影
            const float dotSurfelGrd = dot(surfelNm, grd);          // 方向在法向量上的投影（不能为0）
            const float ghitT        = -doSurfelGro / dotSurfelGrd;  // 光线与面片的交点参数
            const float3 ghitPos     = gro + grd * ghitT;           // 交点位置

            // 15.2 距离对交点位置的梯度
            // 【数学推导】grayDist = dot(ghitPos, ghitPos) = |ghitPos|²
            // ∂grayDist/∂ghitPos = 2 * ghitPos
            const float3 ghitPosGrd = 2 * ghitPos * grayDistGrd;

            // 15.3 交点位置对几何参数的梯度
            // 【数学推导】ghitPos = gro + grd * ghitT
            // ∂ghitPos/∂gro = 1（单位矩阵）
            // ∂ghitPos/∂grd = ghitT（标量乘以单位矩阵）
            groGrd = ghitPosGrd;                    // 对起点位置的直接梯度
            grdGrd = ghitT * ghitPosGrd;            // 对方向向量的梯度
            
            // ∂ghitPos/∂ghitT = grd
            const float ghitTGrd = sum(grd * ghitPosGrd);  // 对交点参数的梯度

            // 15.4 交点参数对几何参数的梯度
            // 【数学推导】ghitT = -dot(surfelNm, gro) / dot(surfelNm, grd)
            // ∂ghitT/∂gro = -surfelNm / dot(surfelNm, grd)
            // ∂ghitT/∂(dot(surfelNm, grd)) = dot(surfelNm, gro) / (dot(surfelNm, grd))²
            groGrd += (-surfelNm * ghitTGrd) / dotSurfelGrd;
            const float dotSurfelGrdGrd = (doSurfelGro * ghitTGrd) / (dotSurfelGrd * dotSurfelGrd);
            
            // ∂(dot(surfelNm, grd))/∂grd = surfelNm
            grdGrd += surfelNm * dotSurfelGrdGrd;
            
        } else {
            // ========== 体积模式的梯度计算 ==========
            // 【几何原理】体积模式使用叉积计算光线到粒子中心的最短距离
            
            // 15.1 重新计算叉积（重现前向传播）
            const float3 gcrod = cross(grd, gro);

            // 15.2 距离对叉积向量的梯度
            // 【数学推导】grayDist = dot(gcrod, gcrod) = |gcrod|²
            // ∂grayDist/∂gcrod = 2 * gcrod
            const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

            // 15.3 叉积向量对输入向量的梯度
            // 【向量微分】对于 c = a × b：
            // ∂c/∂a = [∂c/∂aₓ, ∂c/∂aᵧ, ∂c/∂aᵤ]
            // ∂c/∂b = [∂c/∂bₓ, ∂c/∂bᵧ, ∂c/∂bᵤ]
            //
            // 【具体计算】gcrod = grd × gro
            // gcrod.x = grd.y * gro.z - grd.z * gro.y
            // gcrod.y = grd.z * gro.x - grd.x * gro.z  
            // gcrod.z = grd.x * gro.y - grd.y * gro.x
            
            // 对方向向量grd的梯度
            grdGrd = make_float3(
                gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,    // ∂gcrod/∂grd.x
                gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,    // ∂gcrod/∂grd.y
                gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);   // ∂gcrod/∂grd.z
            
            // 对位置向量gro的梯度
            groGrd = make_float3(
                gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,    // ∂gcrod/∂gro.x
                gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,    // ∂gcrod/∂gro.y
                gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);   // ∂gcrod/∂gro.z
        }

        // =============== 步骤16: 坐标变换的反向梯度传播链 ===============
        // 【变换序列回溯】按照前向传播的逆序进行梯度传播：
        // gro ← gposcr ← gposc ← particlePosition
        // grd ← grdu ← rayDirR ← rayDirection
        
        // 16.1 局部位置对缩放参数的梯度
        // 【数学推导】gro = giscl * gposcr = (1/gscl) * gposcr
        // ∂gro/∂gscl = ∂/∂gscl[(1/gscl) * gposcr] = -gposcr / gscl²
        const float3 gsclGrdGro = make_float3(
            (-gposcr.x / (gscl.x * gscl.x)),    // x分量的缩放梯度
            (-gposcr.y / (gscl.y * gscl.y)),    // y分量的缩放梯度
            (-gposcr.z / (gscl.z * gscl.z))     // z分量的缩放梯度
        ) * (groGrd + groRayHitGrd);  // 汇总来自几何计算和深度计算的贡献

        // ∂gro/∂gposcr = 1/gscl = giscl
        const float3 gposcrGrd = giscl * (groGrd + groRayHitGrd);

        // 16.2 旋转变换的反向传播
        // 【数学推导】gposcr = gposc * particleRotation（矩阵乘法）
        // ∂gposcr/∂gposc = particleRotation^T（转置矩阵）
        // ∂gposcr/∂particleRotation 需要通过四元数微分计算
        const float3 gposcGrd = matmul_bw_vec(particleRotation, gposcrGrd);          // 对位置向量的梯度
        const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);         // 对四元数的梯度（位置分支）

        // 16.3 平移变换的反向传播
        // 【数学推导】gposc = rayOrigin - particlePosition
        // ∂gposc/∂particlePosition = -1（负单位矩阵）
        const float3 rayMoGPosGrd = -gposcGrd;
        particleDensityGradPtr->position = rayMoGPosGrd;  // 最终的位置梯度

        // =============== 步骤17: 方向向量的反向梯度传播链 ===============
        
        // 17.1 方向标准化的反向传播
        // 【数学推导】grd = safe_normalize(grdu)
        // safe_normalize_bw函数实现了标准化操作的反向传播
        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);

        // 17.2 方向缩放的反向传播
        // 【数学推导】grdu = giscl * rayDirR = (1/gscl) * rayDirR
        // ∂grdu/∂gscl = -rayDirR / gscl²
        // ∂grdu/∂rayDirR = 1/gscl = giscl
        
        // 汇总缩放参数的所有梯度贡献
        particleDensityGradPtr->scale = 
            gsclRayHitGrd +                              // 来自深度计算的贡献
            gsclGrdGro +                                 // 来自位置变换的贡献
            (-rayDirR / (gscl * gscl)) * grduGrd;        // 来自方向变换的贡献
        
        const float3 rayDirRGrd = giscl * grduGrd;       // 对旋转后方向向量的梯度

        // 17.3 方向旋转的反向传播
        // 【数学推导】rayDirR = rayDirection * particleRotation
        // 对四元数的梯度需要通过专用函数计算
        const float4 grotGrdRayDirR = matmul_bw_quat(rayDirection, rayDirRGrd, grot);  // 对四元数的梯度（方向分支）
        
        // 17.4 汇总四元数的最终梯度
        // 【梯度汇总】四元数同时影响位置变换和方向变换，需要将两个分支的梯度相加
        particleDensityGradPtr->quaternion.x = grotGrdPoscr.x + grotGrdRayDirR.x;  // x分量梯度
        particleDensityGradPtr->quaternion.y = grotGrdPoscr.y + grotGrdRayDirR.y;  // y分量梯度
        particleDensityGradPtr->quaternion.z = grotGrdPoscr.z + grotGrdRayDirR.z;  // z分量梯度
        particleDensityGradPtr->quaternion.w = grotGrdPoscr.w + grotGrdRayDirR.w;  // w分量梯度

        // =============== 步骤18: 透射率状态更新 ===============
        // 【重要】更新透射率状态，为下一个粒子的处理做准备
        transmittance = nextTransmit;
    }
    // 【函数结束】反向传播完成，所有梯度已计算并存储到输出缓冲区
}

} // namespace threedgut