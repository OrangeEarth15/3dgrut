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

// ========== 球谐辐射粒子模型 - 3DGS中的高级光照计算核心 ==========
//
// 【模块作用】：
// 球谐辐射粒子模型(Spherical Harmonic Radiative Particle Model)是3D高斯溅射(3DGS)
// 中用于实现复杂光照效果的核心组件。它将传统的RGB颜色扩展为方向性光照表示。
//
// 【技术原理】：
// - 使用球谐函数编码方向性光照信息（替代简单的RGB值）
// - 每个高斯粒子存储一组球谐系数，而不是固定的颜色
// - 根据观察方向动态计算颜色，实现视角相关的光照效果
// - 支持复杂的光照现象：镜面反射、次表面散射、各向异性材质等
//
// 【与传统3DGS的区别】：
// 传统3DGS：每个粒子 → 固定RGB颜色
// 球谐3DGS：每个粒子 → 球谐系数数组 → 基于观察方向计算颜色
//
// 【计算流程】：
// 1. 存储：每个粒子存储RadianceMaxNumSphCoefficients个球谐系数
// 2. 查询：根据入射光线方向计算该方向的辐射亮度
// 3. 积分：在体积渲染中累积多个粒子的贡献
// 4. 训练：通过自动微分优化球谐系数参数
//
// 【性能优势】：
// - 紧凑表示：16个系数可表示复杂的方向性光照（vs 每方向一个RGB）
// - 连续插值：球谐基函数保证光照的平滑变化
// - GPU友好：所有计算都是简单的多项式运算
//
// 【应用场景】：
// - 高质量的神经渲染和新视角合成
// - 复杂材质的实时渲染（金属、玻璃、皮肤等）
// - 环境光照的高效近似
// - 光照感知的场景重建

/**
 * 球谐辐射粒子模型 (Spherical Harmonic Radiative Particle Model)
 * 
 * 本模块实现了基于球谐函数的辐射粒子系统，用于高效表示和计算
 * 各向异性的光照效果。球谐函数提供了一种紧凑的方式来编码
 * 复杂的方向性光照信息。
 * 
 * 主要功能：
 * - 球谐系数的存储和管理
 * - 球谐基函数的编码/解码
 * - 支持自动微分的前向和反向传播
 * - 体积渲染的积分计算
 */

// 导入球谐函数模块
#include <3dgut/kernels/slang/common/sphericalHarmonics.slang>

namespace shRadiativeParticle
{

/**
 * 球谐辐射参数缓冲区管理结构
 * 
 * 管理球谐系数数据的存储和梯度更新，支持两种梯度更新模式：
 * 1. 排他性模式：单线程或无竞争环境下的直接更新
 * 2. 并发模式：多线程环境下使用原子操作的安全更新
 */
struct ParametersBuffer
{
    const vector<float, Dim>* _dataPtr;     // 球谐系数数据指针（只读）
    vector<float, Dim> *_gradPtr;           // 球谐系数梯度指针（可写）
    bool exclusiveGradient;                 // 是否使用排他性梯度更新（避免原子操作）
};

/**
 * 球谐辐射粒子的通用参数
 * 
 * 包含渲染和计算所需的全局配置参数
 */
struct CommonParameters
{
    ParametersBuffer parametersBuffer;      // 参数缓冲区管理器
    int sphDegree;                         // 球谐函数阶数（决定表示精度）
};

/**
 * 球谐辐射参数结构 - 存储所有球谐系数
 * 
 * 实现IDifferentiable接口，支持自动微分。
 * 球谐系数数组用于存储方向性辐射信息，系数数量由最大支持的球谐阶数决定。
 * 
 * Dim: 每个系数的维度（通常为3，对应RGB三通道）
 * RadianceMaxNumSphCoefficients: 最大球谐系数数量
 */
struct Parameters : IDifferentiable
{
    vector<float, Dim> sphCoefficients[RadianceMaxNumSphCoefficients];  // 球谐系数数组
};

/**
 * 从缓冲区获取球谐参数的前向函数
 * 
 * 根据粒子索引从全局缓冲区中提取对应的球谐系数。
 * 支持自动微分，可在反向传播中计算梯度。
 * 
 * @param particleIdx 粒子索引（不参与微分）
 * @param parametersBuffer 参数缓冲区（不参与微分）
 * @return 包含所有球谐系数的Parameters结构体
 */
[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff ParametersBuffer parametersBuffer) 
{
    Parameters parameters;
    // 计算粒子在缓冲区中的偏移位置
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    
    // 循环展开优化：逐个复制球谐系数
    [unroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        parameters.sphCoefficients[i] = parametersBuffer._dataPtr[particleOffset + i];
    }
    return parameters;
}

/**
 * 球谐参数获取函数的反向传播版本
 * 
 * 实现fetchParametersFromBuffer函数的梯度反向传播。
 * 将计算得到的梯度累加到全局梯度缓冲区中。
 * 
 * @param particleIdx 粒子索引
 * @param parametersBuffer 参数缓冲区
 * @param parametersGrad 从后续计算传回的梯度
 */
[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] 
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad) 
{
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    
    // 将梯度写回到全局缓冲区
    [unroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        const vector<float, Dim> coeffs = parametersGrad.sphCoefficients[i];
        
        if (parametersBuffer.exclusiveGradient) {
            // 排他性模式：直接累加梯度（性能更好，但需要保证线程安全）
            [unroll] for (int j = 0; j < Dim; ++j) {
                parametersBuffer._gradPtr[particleOffset + i][j] += coeffs[j];
            }
        } else {
            // 并发模式：使用原子操作累加梯度（线程安全，但性能略低）
            [unroll] for (int j = 0; j < Dim; ++j) {
                InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i][j], coeffs[j]);
            }
        }
    }
}

/**
 * 从缓冲区计算球谐辐射亮度
 * 
 * 根据入射光线方向和球谐系数，计算粒子在该方向上的辐射亮度。
 * 使用球谐基函数解码获得最终的颜色值。
 * 
 * @param particleIdx 粒子索引
 * @param incidentDirection 入射光线方向（单位向量）
 * @param sphDegree 球谐函数阶数
 * @param parametersBuffer 参数缓冲区
 * @return 对应方向的辐射亮度值
 */
[BackwardDifferentiable] [ForceInline]
vector<float, Dim> radianceFromBuffer(no_diff uint32_t particleIdx,
                                      no_diff float3 incidentDirection,
                                      no_diff uint32_t sphDegree,
                                      no_diff ParametersBuffer parametersBuffer)
{
    // 使用球谐基函数解码系数为辐射亮度
    return sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
               sphDegree,
               fetchParametersFromBuffer(particleIdx, parametersBuffer).sphCoefficients,
               incidentDirection);
}

/**
 * 积分辐射亮度（体积渲染混合）
 * 
 * 实现体积渲染中的辐射亮度积分，支持两种混合模式：
 * 1. 前向后模式：直接累加权重值
 * 2. 后向前模式：使用线性插值混合（适用于预积分的alpha混合）
 * 
 * @param weight 混合权重（alpha值或透射率加权）
 * @param radiance 当前粒子的辐射亮度
 * @param integratedRadiance 累积的辐射亮度（输入输出参数）
 */
[BackwardDifferentiable][ForceInline]
void integrateRadiance<let backToFront : bool>(float weight,
                                               in vector<float, Dim> radiance,
                                               inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        if (backToFront)
        {
            // 后向前模式：线性插值混合（Over算子）
            integratedRadiance = lerp(integratedRadiance, radiance, weight);
        }
        else
        {
            // 前向后模式：直接累加（正向穿越）
            integratedRadiance += radiance * weight;
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromParameters<let backToFront : bool>(float3 incidentDirection,
                                                             no_diff uint32_t sphDegree,
                                                             float weight,
                                                             Parameters parameters,
                                                             inout vector<float, Dim> integratedRadiance) 
{
    if (weight > 0.0f) 
    {
        integrateRadiance<backToFront>(
            weight,
            sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
                sphDegree,
                parameters.sphCoefficients,
                incidentDirection),
            integratedRadiance
        );
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromBuffer<let backToFront : bool>(no_diff float3 incidentDirection,
                                                         no_diff uint32_t sphDegree,
                                                         float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout vector<float, Dim> integratedRadiance)
{
    integrateRadianceFromParameters<backToFront>(incidentDirection,
                                                 sphDegree,
                                                 weight,
                                                 fetchParametersFromBuffer(particleIdx, parametersBuffer),
                                                 integratedRadiance);
}

} // namespace shRadiativeParticle

// ------------------------------------------------------------------------------------------------------------------
// 导出函数入口点 (CUDA Device Export Functions)
//
// 以下函数是为CUDA设备端导出的入口点，可以被C++/CUDA代码调用。
// 这些函数封装了核心的球谐辐射计算功能，提供统一的接口。

/**
 * 从缓冲区获取粒子特征（辐射亮度）
 * 
 * CUDA设备端导出函数，用于获取指定粒子在特定入射方向上的辐射特征。
 * 
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数（包含缓冲区和球谐阶数）
 * @param incidentDirection 入射光线方向
 * @return 辐射亮度特征向量
 */
[CudaDeviceExport]
inline vector<float, shRadiativeParticle.Dim> particleFeaturesFromBuffer(in uint32_t particleIdx,
                                         shRadiativeParticle.CommonParameters commonParameters,
                                         in float3 incidentDirection)
{
    return sphericalHarmonics.decode<shRadiativeParticle.Dim, shRadiativeParticle.RadianceMaxNumSphCoefficients>(
        commonParameters.sphDegree,
        shRadiativeParticle.fetchParametersFromBuffer(particleIdx, commonParameters.parametersBuffer).sphCoefficients,
        incidentDirection);
}

/**
 * 粒子特征前向积分
 * 
 * 对粒子的辐射特征进行前向积分（前向后遍历模式）。
 * 用于体积渲染中的颜色积分计算。
 * 
 * @param weight 混合权重
 * @param features 当前粒子的辐射特征
 * @param integratedFeatures 累积的辐射特征（输入输出）
 */
[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in vector<float, shRadiativeParticle.Dim> features,
                                         inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadiance<false>(
        weight,
        features,
        integratedFeatures
    );
}

/**
 * 直接从缓冲区进行粒子特征前向积分
 * 
 * 组合了参数获取和特征积分的操作，提高效率。
 * 适用于渲染循环中的直接调用。
 * 
 * @param incidentDirection 入射光线方向
 * @param weight 混合权重
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数
 * @param integratedFeatures 累积的辐射特征（输入输出）
 */
[CudaDeviceExport] inline void particleFeaturesIntegrateFwdFromBuffer(in float3 incidentDirection,
                                                   in float weight,
                                                   in uint32_t particleIdx,
                                                   shRadiativeParticle.CommonParameters commonParameters,
                                                   inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadianceFromBuffer<false>(
        incidentDirection,
        commonParameters.sphDegree,
        weight,
        particleIdx,
        commonParameters.parametersBuffer,
        integratedFeatures);
}

/**
 * 粒子特征反向积分（梅度计算）
 * 
 * 实现体积渲染积分的反向传播，计算alpha和特征的梯度。
 * 使用后向前模式的混合公式进行反向重构。
 * 
 * @param alpha 不透明度值
 * @param alphaGrad alpha的梯度（输入输出）
 * @param features 粒子辐射特征
 * @param featuresGrad 特征梯度（输入输出）
 * @param integratedFeatures 积分的特征（输入输出）
 * @param integratedFeaturesGrad 积分特征的梯度（输入输出）
 */
[CudaDeviceExport] void particleFeaturesIntegrateBwd(
    in float alpha,
    inout float alphaGrad,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> featuresGrad,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad)
{
    if(alpha > 0.0f)
    {
        // 创建微分对（值+梯度）
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> featuresDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(features, featuresGrad);

        // 后向前模式的反向重构：从积分结果恢复前一步的状态
        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        // 调用反向微分函数
        bwd_diff(shRadiativeParticle.integrateRadiance<true>)(
            alphaDiff,
            featuresDiff,
            integratedFeaturesDiff);

        // 提取梯度结果
        alphaGrad = alphaDiff.getDifferential();
        featuresGrad = featuresDiff.getDifferential();
        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
    }
}

/**
 * 直接向缓冲区进行粒子特征反向积分
 * 
 * 组合了反向积分和梯度回写的操作，直接将梯度更新到全局缓冲区。
 * 适用于训练阶段的梯度积累。
 * 
 * @param incidentDirection 入射光线方向
 * @param alpha 不透明度值
 * @param alphaGrad alpha的梯度（输入输出）
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数
 * @param features 粒子辐射特征
 * @param integratedFeatures 积分的特征（输入输出）
 * @param integratedFeaturesGrad 积分特征的梯度（输入输出）
 */
[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad) 
{
    if (alpha > 0.0f)
    {
        // 创建微分对
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        
        // 后向前模式的反向重构
        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        // 直接调用带缓冲区更新的反向微分函数
        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirection,
            commonParameters.sphDegree,
            alphaDiff,
            particleIdx,
            commonParameters.parametersBuffer,
            integratedFeaturesDiff);

        // 提取梯度结果
        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
        alphaGrad = alphaDiff.getDifferential();
    }
}

/**
 * 将粒子特征梯度反向传播到缓冲区
 * 
 * 直接将特征梯度反向传播到球谐系数，更新全局梯度缓冲区。
 * 适用于不需要中间alpha梯度的情况。
 * 
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数
 * @param featuresGrad 特征梯度
 * @param incidentDirection 入射光线方向
 */
[CudaDeviceExport] void particleFeaturesBwdToBuffer(
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> featuresGrad,
    in float3 incidentDirection
)
{
    // 直接调用辐射计算的反向微分函数
    bwd_diff(shRadiativeParticle.radianceFromBuffer)(
        particleIdx,
        incidentDirection,
        commonParameters.sphDegree,
        commonParameters.parametersBuffer,
        featuresGrad);
}