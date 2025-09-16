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

#include <optix.h>

#include <3dgrt/mathUtils.h>
#include <3dgrt/particleDensity.h>

void quaternionWXYZToMatrix(const float4& q, float33& ret) {
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

static constexpr float SH_C0   = 0.28209479177387814f;
static constexpr float SH_C1   = 0.4886025119029199f;
static constexpr float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                  -1.0925484305920792f, 0.5462742152960396f};
static constexpr float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

static inline __device__ float3
radianceFromSpH(int deg, const float3* sphCoefficients, const float3& rdir, bool clamped = true) {
    float3 rad = SH_C0 * sphCoefficients[0];
    if (deg > 0) {
        const float3& dir = rdir;

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad           = rad - SH_C1 * y * sphCoefficients[1] + SH_C1 * z * sphCoefficients[2] -
              SH_C1 * x * sphCoefficients[3];

        if (deg > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SH_C2[0] * xy * sphCoefficients[4] + SH_C2[1] * yz * sphCoefficients[5] +
                  SH_C2[2] * (2.0f * zz - xx - yy) * sphCoefficients[6] +
                  SH_C2[3] * xz * sphCoefficients[7] + SH_C2[4] * (xx - yy) * sphCoefficients[8];

            if (deg > 2) {
                rad = rad + SH_C3[0] * y * (3.0f * xx - yy) * sphCoefficients[9] +
                      SH_C3[1] * xy * z * sphCoefficients[10] +
                      SH_C3[2] * y * (4.0f * zz - xx - yy) * sphCoefficients[11] +
                      SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12] +
                      SH_C3[4] * x * (4.0f * zz - xx - yy) * sphCoefficients[13] +
                      SH_C3[5] * z * (xx - yy) * sphCoefficients[14] +
                      SH_C3[6] * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }
    rad += 0.5f;
    return clamped ? maxf3(rad, make_float3(0.0f)) : rad;
}

static inline __device__ void addSphCoeffGrd(float3* sphCoefficientsGrad, int idx, const float3& val) {
    atomicAdd(&sphCoefficientsGrad[idx].x, val.x);
    atomicAdd(&sphCoefficientsGrad[idx].y, val.y);
    atomicAdd(&sphCoefficientsGrad[idx].z, val.z);
}

/**
 * 球谐函数反向传播函数 - 计算球谐系数的梯度
 * 
 * 🎯 球谐函数基础知识:
 * 球谐函数(Spherical Harmonics)是定义在球面上的正交函数基，可以高效表示方向相关的函数。
 * 在3D渲染中，用于表示环境光照和粒子的方向相关颜色特性。
 * 
 * 🔢 数学公式:
 * RGB(direction) = Σ(coefficient_i × Y_i(θ, φ))
 * 其中 Y_i 是第i个球谐基函数，coefficient_i 是对应系数
 * 
 * 🎨 实际应用:
 * - degree 0: 环境光 (1个系数) - 常数项，各向同性
 * - degree 1: 线性光照 (3个系数) - 主要光源方向
 * - degree 2: 二次光照 (5个系数) - 复杂光照效果
 * - degree 3: 三次光照 (7个系数) - 更精细的光照细节
 */
static inline __device__ float3 radianceFromSpHBwd(
    int deg,                           // 球谐函数的最高度数 (0-3)
    const float3* sphCoefficients,     // 球谐系数数组 (每个float3对应RGB三通道)
    const float3& rdir,                // 观察方向 (标准化的3D向量)
    float weight,                      // 权重系数 (来自体积渲染的alpha*transmittance)
    const float3& rayRadGrd,           // 从后续计算传来的辐射度梯度 ∂L/∂radiance
    float3* sphCoefficientsGrad        // 输出: 球谐系数的梯度累积数组
) {
    // =============== 步骤1: 前向传播重现 ===============
    // 🔄 重新计算前向传播的结果，获得未裁剪的原始辐射度
    // 命名结构其实是gaussian_radianc_unclamped
    const float3 gradu = radianceFromSpH(deg, sphCoefficients, rdir, false);

    // =============== 步骤2: clamp裁剪操作 ===============  
    // 🎯 物理含义: 确保辐射度为非负值 (光不能是负的)
    // 📐 数学含义: ReLU函数 f(x) = max(0, x)
    float3 grad = make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                              gradu.y > 0.0f ? gradu.y : 0.0f,
                              gradu.z > 0.0f ? gradu.z : 0.0f);

    // =============== 步骤3: 梯度预处理 ===============
    // 🔗 链式法则第一步: 考虑权重和ReLU的梯度
    // dL/dRadiance_raw = dL/dRadiance_final × weight × ReLU'(x)
    // 其中 ReLU'(x) = 1 if x > 0, else 0
    float3 dL_dRGB = rayRadGrd * weight;        // 权重缩放
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0);     // clamp梯度 (x通道)
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);     // clamp梯度 (y通道) 
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);     // clamp梯度 (z通道)

    // =============== 步骤4: Degree 0 - 环境光系数的梯度 ===============
    // 🌍 物理含义: 球谐函数的第0项是常数项，表示各向同性的环境光
    // 📐 数学公式: Y_0^0 = SH_C0 (常数 ≈ 0.28209)
    // 
    // 🔗 链式法则推导:
    // RGB = coefficient_0 × SH_C0 + 其他项...
    // ∂RGB/∂coefficient_0 = SH_C0
    // ∂L/∂coefficient_0 = ∂L/∂RGB × ∂RGB/∂coefficient_0 = dL_dRGB × SH_C0
    addSphCoeffGrd(sphCoefficientsGrad, 0, SH_C0 * dL_dRGB);

    if (deg > 0) {
        // =============== 步骤5: Degree 1 - 线性光照系数的梯度 ===============
        // 🎯 物理含义: 线性项描述主要光源的方向性，类似于兰伯特光照
        // 📐 数学基础: degree 1 有3个基函数，对应 x, y, z 方向的线性分量
        
        const float3& sphdir = rdir;  // 观察方向 (已标准化)
        
        // 提取方向分量 (球坐标系中的单位向量)
        float x = sphdir.x;  // X方向分量
        float y = sphdir.y;  // Y方向分量  
        float z = sphdir.z;  // Z方向分量

        // 🔗 Degree 1 球谐基函数的导数计算:
        // Y_1^{-1} = SH_C1 × y     =>  ∂RGB/∂coeff_1 = -SH_C1 × y
        // Y_1^0    = SH_C1 × z     =>  ∂RGB/∂coeff_2 = SH_C1 × z  
        // Y_1^1    = SH_C1 × x     =>  ∂RGB/∂coeff_3 = -SH_C1 × x
        float dRGBdsh1 = -SH_C1 * y;  // 对系数1的偏导数
        float dRGBdsh2 = SH_C1 * z;   // 对系数2的偏导数
        float dRGBdsh3 = -SH_C1 * x;  // 对系数3的偏导数

        // 应用链式法则: ∂L/∂coefficient = ∂L/∂RGB × ∂RGB/∂coefficient
        addSphCoeffGrd(sphCoefficientsGrad, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 3, dRGBdsh3 * dL_dRGB);

        if (deg > 1) {
            // =============== 步骤6: Degree 2 - 二次光照系数的梯度 ===============
            // 🎯 物理含义: 二次项捕获更复杂的光照效果，如边缘光、反射等
            // 📐 数学基础: degree 2 有5个基函数，涉及二次项组合
            
            // 预计算二次项 (优化性能)
            float xx = x * x, yy = y * y, zz = z * z;  // 平方项
            float xy = x * y, yz = y * z, xz = x * z;  // 交叉项

            // 🔗 Degree 2 球谐基函数的导数:
            // Y_2^{-2} ∝ xy           =>  ∂RGB/∂coeff_4 = SH_C2[0] × xy
            // Y_2^{-1} ∝ yz           =>  ∂RGB/∂coeff_5 = SH_C2[1] × yz
            // Y_2^0    ∝ (3z²-1)      =>  ∂RGB/∂coeff_6 = SH_C2[2] × (2z²-x²-y²)
            // Y_2^1    ∝ xz           =>  ∂RGB/∂coeff_7 = SH_C2[3] × xz  
            // Y_2^2    ∝ (x²-y²)      =>  ∂RGB/∂coeff_8 = SH_C2[4] × (x²-y²)
            float dRGBdsh4 = SH_C2[0] * xy;                    // xy项的梯度
            float dRGBdsh5 = SH_C2[1] * yz;                    // yz项的梯度
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);  // z主导项的梯度
            float dRGBdsh7 = SH_C2[3] * xz;                    // xz项的梯度
            float dRGBdsh8 = SH_C2[4] * (xx - yy);             // x-y差项的梯度

            // 累积梯度到对应的球谐系数
            addSphCoeffGrd(sphCoefficientsGrad, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 8, dRGBdsh8 * dL_dRGB);

            if (deg > 2) {
                // =============== 步骤7: Degree 3 - 三次光照系数的梯度 ===============
                // 🎯 物理含义: 三次项提供最精细的光照细节和高频特征
                // 📐 数学基础: degree 3 有7个基函数，涉及三次项组合
                // 🎨 应用场景: 高质量渲染中的精细光照效果
                
                // 🔗 Degree 3 球谐基函数的导数 (复杂的三次多项式):
                // 每个公式都是对应球谐基函数对方向向量的偏导数
                float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);          // Y_3^{-3}: y(3x²-y²)
                float dRGBdsh10 = SH_C3[1] * xy * z;                        // Y_3^{-2}: xyz
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);      // Y_3^{-1}: y(4z²-x²-y²)
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy); // Y_3^0: z(2z²-3x²-3y²)
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);      // Y_3^1: x(4z²-x²-y²)
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);                 // Y_3^2: z(x²-y²)
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);          // Y_3^3: x(x²-3y²)

                // 🔗 链式法则: 将每个偏导数与损失梯度相乘并累积
                addSphCoeffGrd(sphCoefficientsGrad, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 15, dRGBdsh15 * dL_dRGB);
            }
        }
    }

    // =============== 步骤8: 返回裁剪后的辐射度值 ===============
    // 🎯 返回值含义: 经过ReLU裁剪的最终辐射度，用于后续的体积渲染计算
    // 📊 用途: 这个值会与权重相乘，贡献到最终的像素颜色
    return grad;
}

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

static inline __device__ void fetchParticleSphCoefficients(
    const int32_t particleIdx,
    const float* particlesSphCoefficients,
    float3* sphCoefficients) {
    const uint32_t particleOffset = particleIdx * SPH_MAX_NUM_COEFFS * 3;
#pragma unroll
    for (unsigned int i = 0; i < SPH_MAX_NUM_COEFFS; ++i) {
        const int offset   = i * 3;
        sphCoefficients[i] = make_float3(
            particlesSphCoefficients[particleOffset + offset + 0],
            particlesSphCoefficients[particleOffset + offset + 1],
            particlesSphCoefficients[particleOffset + offset + 2]);
    }
}

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponseGrd(float grayDist, float gres, float gresGrd) {
    /// generalized gaussian of degree b : scaling a = -4.5/3^b
    /// d_e^{a*|x|^b}/d_x^2 = a*(0.5*b)*x^{b-2}*e^{a*|x|^b}
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

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponse(float grayDist) {
    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
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

template <int GeneralizedGaussianDegree = 2, bool clamped>
static inline __device__ float particleScaledResponse(float grayDist, float modulatedMinResponse, float responseModulation = 1.0f) {

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

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
__device__ inline bool processHit(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float* particlesSphCoefficients,
    const float minParticleKernelDensity,
    const float minParticleAlpha,
    const int32_t sphEvalDegree,
    float* transmittance,
    float3* radiance,
    float* depth,
    float3* normal) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleRotation;
    float particleDensity;

    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleRotation,
        particleDensity);

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);

    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    if (acceptHit) {
        const float weight = galpha * (*transmittance);

        // distance to the gaussian center projection on the ray
        const float3 grds = particleScale * grd * (SurfelPrimitive ? -gro.z / grd.z : dot(grd, -1 * gro));
        const float hitT  = sqrtf(dot(grds, grds));

        // radiance from sph coefficients
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particlesSphCoefficients,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpH(sphEvalDegree, &sphCoefficients[0], rayDirection);

        *radiance += grad * weight;
        *transmittance *= (1 - galpha);
        *depth += hitT * weight;

        if (normal) {
            constexpr float ellispoidSqRadius = 9.0f;
            const float3 particleScaleRotated = (particleRotation * particleScale);
            *normal += weight * (SurfelPrimitive ? make_float3(0, 0, (grd.z > 0 ? 1 : -1) * particleScaleRotated.z) : safe_normalize((gro + grd * (dot(grd, -1 * gro) - sqrtf(ellispoidSqRadius - grayDist))) * particleScaleRotated));
        }
    }

    return acceptHit;
}

__device__ inline bool intersectCustomParticle(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleRotation;
    float particleDensity;
    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleRotation,
        particleDensity);

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);

    // distance to the gaussian center projection on the ray
    const float grp   = -dot(grd, gro);
    const float3 grds = particleScale * grd * grp;
    hitDistance       = (grp < 0.f ? -1.f : 1.f) * sqrtf(dot(grds, grds));

    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        const float3 gcrod   = cross(grd, gro);
        const float grayDist = dot(gcrod, gcrod);
        return (grayDist < maxParticleSquaredDistance);
    }
    return false;
}

__device__ inline bool intersectInstanceParticle(
    const float3& particleRayOrigin,
    const float3& particleRayDirection,
    const int32_t particleIdx,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    const float numerator   = -dot(particleRayOrigin, particleRayDirection);
    const float denominator = 1.f / dot(particleRayDirection, particleRayDirection);
    hitDistance             = numerator * denominator;
    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        const float3 gcrod = cross(safe_normalize(particleRayDirection), particleRayOrigin);
        return (dot(gcrod, gcrod) * denominator < maxParticleSquaredDistance);
    }
    return false;
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
__device__ inline void processHitBwd(
    const float3& rayOrigin,
    const float3& rayDirection,
    int32_t particleIdx,
    const ParticleDensity* particleDensityPtr,
    ParticleDensity* particleDensityGradPtr,
    const float* particleRadiancePtr,
    float* particleRadianceGradPtr,
    float minParticleKernelDensity,
    float minParticleAlpha,
    float minTransmittance,
    int32_t sphEvalDegree,
    float integratedTransmittance,
    float& transmittance,
    float transmittanceGrad,
    float3 integratedRadiance,
    float3& radiance,
    float3 radianceGrad,
    float integratedDepth,
    float& depth,
    float depthGrad) {
    float3 particlePosition;
    float3 gscl;
    float33 particleRotation;
    float particleDensity;
    float4 grot;

    {
        const ParticleDensity particleData = particleDensityPtr[particleIdx];
        particlePosition                   = particleData.position;
        gscl                               = particleData.scale;
        grot                               = particleData.quaternion;
        quaternionWXYZToMatrix(grot, particleRotation);
        particleDensity = particleData.density;
    }

    // project ray in the gaussian
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha)) {
        ParticleDensity& particleDensityGrad = particleDensityGradPtr[particleIdx];

        const float3 grdd   = grd * (SurfelPrimitive ? -gro.z / grd.z : dot(grd, -1 * gro));
        const float3 grds   = gscl * grdd;
        const float gsqdist = dot(grds, grds);
        const float gdist   = sqrtf(gsqdist);

        const float weight = galpha * transmittance;

        const float nextTransmit = (1 - galpha) * transmittance;

        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        depth += weight * gdist;
        const float residualHitT =
            fmaxf((nextTransmit <= minTransmittance ? 0 : (integratedDepth - depth) / nextTransmit),
                  0);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        //
        // ===> d_hitT / d_galpha = gdist * prevTrm - residualHitT * prevTrm
        //                        = (gdist - residualHitT) * prevTrm
        //
        const float galphaRayHitGrd = (gdist - residualHitT) * transmittance * depthGrad;
        //
        // ===> d_hitT / d_gsqdist = weight / (2*gdist)
        // ===> d_gsqdist / d_grds =  2 * grds
        const float3 grdsRayHitGrd = gsqdist > 0.0f ? ((2 * grds * weight) / (2 * gdist)) * depthGrad : make_float3(0.0f);

        // ---> grds = gscl * grd * dot(grd, -1 * gro)
        //
        // ===> d_grds / d_gscl =  grd * dot(grd, -1 * gro)
        const float3 gsclRayHitGrd = grdd * grdsRayHitGrd;
        // ===> d_grds / d_grd =  - gscl * grd * (2 dot(grd, -1 * gro)
        const float3 grdRayHitGrd = -gscl * make_float3(2 * grd.x * gro.x + grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + 2 * grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + grd.y * gro.y + 2 * grd.z * gro.z) * grdsRayHitGrd;
        //
        // ===> d_grds / d_gro = - gscl * grd * grd
        const float3 groRayHitGrd = -gscl * grd * grd * grdsRayHitGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_galpha = prevTrm * nextTrm = residualTrm
        const float residualTrm     = galpha < 0.999999f ? integratedTransmittance / (1 - galpha) : transmittance;
        const float galphaRayDnsGrd = residualTrm * -transmittanceGrad;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // compute the gradient wrt to the sph coefficients and position (through the sph view
        // direction)
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particleRadiancePtr,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpHBwd(sphEvalDegree, &sphCoefficients[0], rayDirection, weight, radianceGrad, (float3*)&particleRadianceGradPtr[particleIdx * SPH_MAX_NUM_COEFFS * 3]);

        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
        const float3 rayRad = weight * grad;
        radiance += rayRad;
        const float3 residualRayRad = maxf3((nextTransmit <= minTransmittance ? make_float3(0) : (integratedRadiance - radiance) / nextTransmit),
                                            make_float3(0));

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gdns = residualTrm * gres
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
        atomicAdd(
            &particleDensityGrad.density,
            gres * (galphaRayHitGrd + galphaRayDnsGrd + transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                    transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                    transmittance * (grad.z - residualRayRad.z) * radianceGrad.z));

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gres = residualTrm * gdns
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
        const float gresGrd =
            particleDensity * (galphaRayHitGrd + galphaRayDnsGrd + transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                               transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                               transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gres = exp(-0.0555 * grayDist * grayDist)
        // ===> d_gres / d_grayDist = -0.111 * grayDist * exp(-0.555 * grayDist * grayDist)
        //                          = -0.111 * grayDist * gres
        const float grayDistGrd = particleResponseGrd<PARTICLE_KERNEL_DEGREE>(grayDist, gres, gresGrd);

        float3 grdGrd, groGrd;
        if (SurfelPrimitive) {
            const float3 surfelNm    = make_float3(0, 0, 1);
            const float doSurfelGro  = dot(surfelNm, gro);
            const float dotSurfelGrd = dot(surfelNm, grd); // cannot be null otherwise no hit
            const float ghitT        = -doSurfelGro / dotSurfelGrd;
            const float3 ghitPos     = gro + grd * ghitT;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grayDist = dot(ghitPos, ghitPos)
            //               = ghitPos.x^2 + ghitPos.y^2 + ghitPos.z^2
            // ===> d_grayDist / d_ghitPos = 2*ghitPos
            const float3 ghitPosGrd = 2 * ghitPos * grayDistGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> ghitPos = gro + grd * ghitT
            //
            // ===> d_ghitPos / d_gro = 1
            // ===> d_ghitPos / d_grd = ghitT
            groGrd = ghitPosGrd;
            grdGrd = ghitT * ghitPosGrd;
            // ===> d_ghitPos / d_ghitT = grd
            const float ghitTGrd = sum(grd * ghitPosGrd);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> ghitT = -dot(surfelNm, gro) / dot(surfNm, grd)
            //
            // ===> d_ghitT / d_gro = -surfelNm / dot(surfNm, grd)
            // ===> d_ghitT / d_dotSurfelGrd = dot(surfelNm, gro) / dotSurfelGrd^2
            groGrd += (-surfelNm * ghitTGrd) / dotSurfelGrd;
            const float dotSurfelGrdGrd = (doSurfelGro * ghitTGrd) / (dotSurfelGrd * dotSurfelGrd);
            // ===> d_dotSurfelGrd / d_grd = surfelNm
            grdGrd += surfelNm * dotSurfelGrdGrd;
        } else {
            const float3 gcrod = cross(grd, gro);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grayDist = dot(gcrod, gcrod)
            //               = gcrod.x^2 + gcrod.y^2 + gcrod.z^2
            // ===> d_grayDist / d_gcrod = 2*gcrod
            const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gcrod = cross(grd, gro)
            // ---> gcrod.x = grd.y * gro.z - grd.z * gro.y
            // ---> gcrod.y = grd.z * gro.x - grd.x * gro.z
            // ---> gcrod.z = grd.x * gro.y - grd.y * gro.x
            grdGrd = make_float3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                 gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                 gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
            groGrd = make_float3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                 gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                 gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);
        }

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gro = (1/gscl)*gposcr
        // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
        // ===> d_gro / d_gposcr = (1/gscl)
        const float3 gsclGrdGro = make_float3((-gposcr.x / (gscl.x * gscl.x)),
                                              (-gposcr.y / (gscl.y * gscl.y)),
                                              (-gposcr.z / (gscl.z * gscl.z))) *
                                  (groGrd + groRayHitGrd);
        const float3 gposcrGrd = giscl * (groGrd + groRayHitGrd);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposcr = matmul(gposc, grotMat)
        // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
        // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
        const float3 gposcGrd     = matmul_bw_vec(particleRotation, gposcrGrd);
        const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposc = rayOri - gpos
        // ===> d_gposc / d_gpos = -1
        const float3 rayMoGPosGrd = -gposcGrd;
        atomicAdd(&particleDensityGrad.position.x, rayMoGPosGrd.x);
        atomicAdd(&particleDensityGrad.position.y, rayMoGPosGrd.y);
        atomicAdd(&particleDensityGrad.position.z, rayMoGPosGrd.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grd = safe_normalize(grdu)
        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grdu = (1/gscl)*rayDirR
        // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
        // ===> d_grdu / d_rayDirR = (1/gscl)
        atomicAdd(&particleDensityGrad.scale.x, gsclRayHitGrd.x + gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
        atomicAdd(&particleDensityGrad.scale.y, gsclRayHitGrd.y + gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
        atomicAdd(&particleDensityGrad.scale.z, gsclRayHitGrd.z + gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
        const float3 rayDirRGrd = giscl * grduGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDirR = matmul(rayDir, grotMat)
        // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
        const float4 grotGrdRayDirR = matmul_bw_quat(rayDirection, rayDirRGrd, grot);
        atomicAdd(&particleDensityGrad.quaternion.x, grotGrdPoscr.x + grotGrdRayDirR.x);
        atomicAdd(&particleDensityGrad.quaternion.y, grotGrdPoscr.y + grotGrdRayDirR.y);
        atomicAdd(&particleDensityGrad.quaternion.z, grotGrdPoscr.z + grotGrdRayDirR.z);
        atomicAdd(&particleDensityGrad.quaternion.w, grotGrdPoscr.w + grotGrdRayDirR.w);

        transmittance = nextTransmit;
    }
}
