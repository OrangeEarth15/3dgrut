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
 * 📋 FEATURES反向传播调用链（最终层）- 体积渲染积分的数学实现
 * 
 * 🔄 调用路径（到达算法终点）：
 * 1. 上4层：gutKBufferRenderer.cuh:172 → ... → shRadiativeParticles.slang:320
 * 2. 【当前层】shRadiativeParticles.slang:244 → 体积渲染的积分数学公式
 * 
 * 🎯 本层职责：3DGUT体积渲染的数学核心
 * - 实现两种积分模式的数学公式
 * - 处理多层透明粒子的颜色合成
 * - 支持自动微分的反向传播
 * - 编译时优化的模板特化
 * 
 * 🧮 体积渲染积分的数学原理：
 * 
 * 前向后模式（K-Buffer标准）：
 * ∫ color += particle_color × weight
 * 特点：简单累加，适合无序透明渲染
 * 
 * 后向前模式（深度排序）：
 * ∫ color = lerp(old_color, particle_color, weight)
 * 特点：线性插值，适合有序透明渲染
 * 
 * 🔗 反向传播梯度计算：
 * - 前向后：∂∫color/∂particle_color = weight
 * - 后向前：∂∫color/∂particle_color = weight, ∂∫color/∂old_color = (1-weight)
 * 
 * ⚡ 性能优化技术：
 * - 早期退出：weight <= 0.0f 零开销跳过
 * - 编译时分支：let模板参数消除运行时判断
 * - 强制内联：函数调用开销完全消除
 * - 自动微分：Slang编译器自动生成反向版本
 * 
 * @param weight 混合权重（alpha × transmittance，决定粒子贡献）
 * @param radiance 当前粒子的辐射亮度（球谐解码后的RGB值）
 * @param integratedRadiance 累积辐射亮度（输入输出，最终像素颜色）
 */
[BackwardDifferentiable][ForceInline]
void integrateRadiance<let backToFront : bool>(float weight,
                                               in vector<float, Dim> radiance,
                                               inout vector<float, Dim> integratedRadiance)
{
    // ========== 最终层步骤1：早期退出优化 ==========
    // 🚀 GPU性能关键：只有有效权重的粒子才进行颜色混合
    // 数值含义：weight = alpha × transmittance，表示粒子的实际贡献度
    if (weight > 0.0f)
    {
        // ========== 最终层步骤2：体积渲染积分的核心数学公式 ==========
        // 🧮 这里是整个3DGUT体积渲染的数学终点！
        // 所有复杂的球谐光照、几何变换、透明度计算最终归结为这两个简单公式
        
        if (backToFront)
        {
            // 🔄 后向前模式：Over算子（深度排序时使用）
            // 数学公式：C_final = C_old × (1-w) + C_particle × w
            // 物理意义：线性插值混合，保持颜色的平滑过渡
            // 反向梯度：∂C_final/∂C_particle = w, ∂C_final/∂C_old = (1-w)
            integratedRadiance = lerp(integratedRadiance, radiance, weight);
        }
        else
        {
            // ⚡ 前向后模式：累积算子（K-Buffer标准模式）
            // 数学公式：C_final += C_particle × w
            // 物理意义：直接累加加权贡献，实现多粒子透明效果
            // 反向梯度：∂C_final/∂C_particle = w（更简单的梯度计算）
            // 🎯 3DGUT的主要使用模式：高效的无序透明渲染
            integratedRadiance += radiance * weight;
        }
    }
    // 🚫 weight <= 0：粒子距离太远或alpha太小，对最终颜色无贡献
    // 编译器优化：这个分支在GPU上接近零开销
}

/**
 * 📋 FEATURES反向传播调用链（第5层/5层）- 球谐解码与积分的数学核心
 * 
 * 🔄 调用路径（到达最终层）：
 * 1. 上4层：gutKBufferRenderer.cuh:172 → ... → shRadiativeParticles.slang:325
 * 2. 【当前层】shRadiativeParticles.slang:274 → sphericalHarmonics.decode<>() + integrateRadiance<backToFront>()
 * 
 * 🎯 本层职责：球谐光照的数学实现核心
 * - 球谐解码：将紧凑的系数数组解码为指定方向的RGB值
 * - 体积积分：将解码的RGB按权重积分到最终像素颜色
 * - 自动微分：支持系数到RGB到像素的完整梯度链
 * 
 * 🧮 数学原理 - 球谐光照的完整实现：
 * 1. 球谐解码：RGB(ω) = Σ(i=0 to N-1) c_i × Y_i(ω)
 *    - c_i：第i个球谐系数（3D向量，对应RGB）
 *    - Y_i(ω)：第i个球谐基函数在方向ω的值
 *    - N：系数数量，由球谐阶数决定（1,4,9,16...）
 * 
 * 2. 体积积分：pixel_color += weight × RGB(ω)
 *    - weight：粒子贡献权重（alpha × transmittance）
 *    - 实现视角相关的光照效果累积
 * 
 * 🔗 反向传播梯度流：
 * ∂L/∂pixel → ∂L/∂RGB → ∂L/∂c_i (通过球谐基函数导数)
 * 
 * ⚡ 性能优化：
 * - 早期退出：weight <= 0.0f 直接跳过计算
 * - 强制内联：消除函数调用开销
 * - 模板特化：编译时确定积分模式
 * 
 * @param incidentDirection 入射光线方向（观察方向，影响球谐基函数值）
 * @param sphDegree 球谐函数阶数（决定使用的系数数量和计算精度）
 * @param weight 混合权重（alpha × transmittance，决定粒子贡献）
 * @param parameters 球谐系数结构体（包含所有系数的完整数组）
 * @param integratedRadiance 累积辐射亮度（输入输出，最终像素颜色）
 */
[BackwardDifferentiable][ForceInline]
void integrateRadianceFromParameters<let backToFront : bool>(float3 incidentDirection,
                                                             no_diff uint32_t sphDegree,
                                                             float weight,
                                                             Parameters parameters,
                                                             inout vector<float, Dim> integratedRadiance) 
{
    // ========== 早期退出优化 ==========
    if (weight > 0.0f) 
    {
        // ========== 第5层核心：球谐解码 + 体积积分 ==========
        integrateRadiance<backToFront>(
            weight,                     // 混合权重（决定粒子对最终颜色的贡献度）
            
            // 🌟 关键：球谐基函数解码
            // 功能：从紧凑的系数数组计算指定方向的RGB颜色
            // 数学：RGB = Σ(c_i × Y_i(ω))，其中ω是观察方向
            sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
                sphDegree,                      // 球谐阶数（控制计算精度）
                parameters.sphCoefficients,     // 球谐系数数组（粒子的光照参数）
                incidentDirection),             // 观察方向（影响最终颜色）
            
            integratedRadiance              // 累积结果（最终像素颜色）
        );
    }
    // 否则：weight <= 0，粒子无贡献，直接跳过所有计算
}

/**
 * 📋 FEATURES反向传播调用链（第4层/5层）- 球谐缓冲区接口层
 * 
 * 🔄 调用路径：
 * 1. 上3层：gutKBufferRenderer.cuh:172 → shRadiativeGaussianParticles.cuh:780 → shRadiativeParticles.slang:534
 * 2. 【当前层】shRadiativeParticles.slang:294 → integrateRadianceFromParameters<backToFront>()
 * 3. 下层：shRadiativeParticles.slang:301 → integrateRadianceFromParameters<backToFront>() + sphericalHarmonics.decode<>()
 * 
 * 🎯 本层职责：从缓冲区获取参数的中转接口
 * - 粒子参数获取：从全局缓冲区提取指定粒子的所有球谐系数
 * - 函数转发：将获取的参数传递给真正的积分算法
 * - 类型保持：维持自动微分兼容性 [BackwardDifferentiable]
 * 
 * 🔗 数据流向：
 * particleIdx + parametersBuffer → fetchParametersFromBuffer() → Parameters结构 → 积分计算
 * 
 * 📊 性能特点：
 * - [ForceInline] 强制内联：消除函数调用开销
 * - 纯转发函数：实际计算在下一层完成
 * - 自动微分透传：保持梯度计算能力
 */
[BackwardDifferentiable][ForceInline]
void integrateRadianceFromBuffer<let backToFront : bool>(no_diff float3 incidentDirection,
                                                         no_diff uint32_t sphDegree,
                                                         float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout vector<float, Dim> integratedRadiance)
{
    // 🔄 核心转发：缓冲区访问 → 参数结构 → 积分计算
    // 1. fetchParametersFromBuffer：从全局缓冲区获取粒子的所有球谐系数
    // 2. integrateRadianceFromParameters：使用获取的参数进行球谐积分计算
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

// ========== particleFeaturesIntegrateFwd调用链 - 第3层：Slang导出接口 ==========
//
// 🎨 【调用链位置】：
// 1. K-Buffer (gutKBufferRenderer.cuh:264) → particles.featureIntegrateFwd()
// 2. C++包装 (shRadiativeGaussianParticles.cuh:661) → particleFeaturesIntegrateFwd()
// 3. 【当前层】Slang导出 (shRadiativeParticles.slang:299) → shRadiativeParticle.integrateRadiance<false>()
// 4. 核心实现 (shRadiativeParticles.slang:205) → 颜色混合算法
//
// 【本层作用】：辐射特征积分的跨语言导出接口
// - [CudaDeviceExport] 使得Slang函数可以被CUDA C++代码调用
// - 提供稳定的ABI边界，隔离底层球谐光照实现细节
// - 纯转发函数，直接调用球谐辐射粒子的积分算法
// - 确保向量类型兼容性和参数传递正确性
//
// 【渲染模式】：
// - 使用前向后遍历模式 (backToFront=false)
// - 直接累积模式：integratedFeatures += features * weight
// - 适用于正向穿越的颜色积分 (front-to-back color integration)
//
// 🌌 【球谐光照原理】：
// - shRadiativeParticle 代表支持球谐基函数的辐射粒子
// - 每个粒子存储一组球谐系数，而不是固定的RGB值
// - 根据观察方向动态计算颜色，实现视角相关的光照效果
// - 支持复杂的光照现象：镜面反射、次表面散射、各向异性材质
//
// 📊 【核心公式】（将在第4层实现）：
// if (weight > 0.0f) {
//     integratedFeatures += features * weight;  // 直接累积模式
// }
//
/**
 * 粒子特征前向积分 - Slang导出接口
 * 
 * 功能：为CUDA C++代码提供调用Slang实现的特征积分功能的接口
 * 作用：这是一个纯转发函数，将调用直接传递给内部的shRadiativeParticle.integrateRadiance()
 * 
 * 对粒子的辐射特征进行前向积分（前向后遍历模式）。
 * 用于体积渲染中的颜色积分计算。
 * 
 * @param weight 混合权重（由前面的densityIntegrateHit计算得出）
 * @param features 当前粒子的辐射特征向量（RGB或球谐系数）
 * @param integratedFeatures 累积的辐射特征（输入输出，会被修改）
 */
[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in vector<float, shRadiativeParticle.Dim> features,
                                         inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    // ========== 直接转发到球谐辐射粒子的核心积分算法 ==========
    // 功能：将所有参数传递给真正的球谐辐射积分实现
    // 模板参数：<false> 表示使用前向后渲染模式的直接累积
    // 返回类型：void，直接修改integratedFeatures参数
    shRadiativeParticle.integrateRadiance<false>(
        weight,              // 混合权重（alpha * transmittance）
        features,            // 当前粒子的辐射特征
        integratedFeatures   // 累积的辐射特征（输入输出）
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
 * 📋 FEATURES反向传播调用链（第3层/5层）- Slang自动微分入口
 * 
 * 🔄 调用路径：
 * 1. 上层：gutKBufferRenderer.cuh:172 → particles.featuresIntegrateBwdToBuffer<false>()
 * 2. 上层：shRadiativeGaussianParticles.cuh:780 → particleFeaturesIntegrateBwdToBuffer()
 * 3. 【当前层】shRadiativeParticles.slang:485 → bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)()
 * 4. 下层：shRadiativeParticles.slang:294 → integrateRadianceFromBuffer<backToFront>()
 * 5. 下层：shRadiativeParticles.slang:301 → integrateRadianceFromParameters<backToFront>() + sphericalHarmonics.decode<>()
 * 
 * 🎯 本层职责：Slang自动微分系统的关键入口
 * - 设置微分对（DifferentialPair）进行自动微分
 * - 重构前向状态（为反向传播准备中间值）
 * - 调用Slang编译器生成的反向微分函数 bwd_diff()
 * - 自动将梯度传播到球谐系数缓冲区
 * 
 * 🧮 反向微分原理：
 * - Slang编译器自动生成 bwd_diff(forward_function) 反向版本
 * - DifferentialPair<T> 封装值和梯度，支持链式法则自动应用
 * - 前向重构：从后向前结果推导前向中间状态（逆向工程）
 * - 梯度传播：∂L/∂RGB → ∂L/∂球谐系数 (通过球谐基函数的导数)
 * 
 * 📊 数学核心：
 * - 前向：RGB = Σ(c_i × Y_i(ω))  [球谐解码]
 * - 反向：∂L/∂c_i = ∂L/∂RGB × Y_i(ω)  [链式法则]
 * - 其中 Y_i(ω) 是球谐基函数在观察方向ω的值
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
        // ========== 步骤1：创建自动微分对象 ==========
        // DifferentialPair封装前向值和对应梯度，支持自动链式法则
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        
        // ========== 步骤2：前向状态重构 ==========
        // 从后向前积分结果反推前向中间状态（逆向工程）
        // 数学：如果 result = old + alpha*features，则 old = (result - alpha*features)/(1-alpha)
        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        // ========== 步骤3：调用Slang自动微分引擎 ==========
        // 🚀 关键：bwd_diff() 是Slang编译器自动生成的反向微分版本
        // 功能：自动应用链式法则，将RGB梯度传播到球谐系数梯度
        // 结果：直接更新全局球谐参数缓冲区的梯度
        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirection,                      // 观察方向（影响球谐基函数值）
            commonParameters.sphDegree,             // 球谐阶数（决定系数数量）
            alphaDiff,                             // alpha的微分对
            particleIdx,                           // 粒子索引
            commonParameters.parametersBuffer,      // 球谐系数缓冲区（数据+梯度）
            integratedFeaturesDiff);               // 积分特征的微分对

        // ========== 步骤4：提取梯度结果 ==========
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