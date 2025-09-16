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

/**
 * 高斯粒子模型 (Gaussian Particle Model)
 * 
 * 本模块实现了基于高斯函数的粒子系统，用于高质量的体积渲染。
 * 高斯粒子通过三维高斯球来表示，支持旋转、缩放和平移变换。
 * 
 * 主要功能：
 * - 高斯粒子参数管理（位置、旋转、缩放、密度）
 * - 射线-高斯球相交测试
 * - 多种核函数支持（从线性到高阶多项式）
 * - 自动微分和梯度计算
 * - Surfel模式（二维面片）支持
 */

// 导入几何变换模块
#include <3dgut/kernels/slang/common/transforms.slang>

namespace gaussianParticle {

/**
 * 原始粒子参数结构 - 紧凑存储格式
 * 
 * 存储高斯粒子的原始参数，用于高效的内存存储和传输。
 * 实现IDifferentiable接口，支持自动微分计算。
 * 
 * 注意：四元数使用(w,x,y,z)顺序，其中w是实部。
 */
 // IDifferentiable声明这个结构体支持自动微分
struct RawParameters : IDifferentiable { // 3 + 1 + 4 + 3 + 1 = 12
    float3 position;    // 粒子3D位置坐标（世界空间）
    float density;      // 粒子密度（不透明度，控制透射率）
    float4 quaternion;  // 粒子旋转四元数 (w,x,y,z)，单位四元数
    float3 scale;       // 粒子3D缩放因子（沿三个主轴的半径）
    float padding;      // 内存对齐填充（GPU内存对齐优化）
};

/**
 * 原始参数缓冲区管理结构
 * 
 * 管理高斯粒子参数的存储和梯度更新。支持两种梯度更新模式：
 * 1. 排他性模式：单线程或无竞争环境，直接更新（性能更好）
 * 2. 并发模式：多线程环境，使用原子操作（线程安全）
 */
struct RawParametersBuffer {
    const RawParameters* _dataPtr;        // 指向参数数据的常量指针（只读）
    RawParameters *_gradPtr;              // 指向梯度数据的指针（可写，用于梯度积累）
    bool exclusiveGradient;               // 是否使用排他性梯度更新（避免原子操作）
};

/**
 * 通用参数包装结构
 */
struct CommonParameters {
    RawParametersBuffer parametersBuffer;  // 参数缓冲区管理器
};

/**
 * 从缓冲区获取原始粒子参数的前向函数
 *
 * 根据粒子索引从全局缓冲区中获取原始参数。
 * 支持Surfel模式，在该模式下将Z轴缩放设为极小值以实现2D面片效果。
 *
 * @param particleIdx 粒子索引（不参与微分计算）
 * @param parametersBuffer 参数缓冲区（不参与微分计算）
 * @return 获取的原始参数（可能经Surfel模式修改）
 */
 //[BackwardDifferentiable] 含义：声明此函数支持反向自动微分
 // 作用：Slang 编译器会自动生成对应的梯度计算代码
 // 实际效果：训练时可以自动计算参数梯度，无需手写求导
[BackwardDifferentiable][ForceInline] RawParameters fetchParametersFromBuffer(
    no_diff uint32_t particleIdx,              // 粒子索引（不参与微分）
    no_diff RawParametersBuffer parametersBuffer) {  // 参数缓冲区（不参与微分）
    
    // 从全局缓冲区读取指定粒子的参数
    RawParameters rawParameters = parametersBuffer._dataPtr[particleIdx];
    
    // Surfel模式：将三维高斯球压缩为二维面片
    if (Surfel) {
        rawParameters.scale.z = 1e-06f;  // 将Z轴厚度设为接近0（极薄面片）
    }
    return rawParameters;
}

/**
 * fetchParametersFromBuffer函数的反向传播版本
 *
 * 实现参数获取的梯度反向传播，将从后续计算传回的梯度累加到全局梯度缓冲区。
 * 对Surfel模式做特殊处理，不更新Z轴缩放梯度。
 *
 * @param particleIdx 粒子索引
 * @param parametersBuffer 参数缓冲区
 * @param rawParameters 从后续计算传回的梯度
 */
// 模式1:自动生成梯度（只需 [BackwardDifferentiable]）
// 模式2:自定义梯度逻辑（需要成对出现[BackwardDerivativeOf(xxx)]和[BackwardDifferentiable]）

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] void fetchParametersFromBufferBwd(
    no_diff uint32_t particleIdx,              // 粒子索引
    no_diff RawParametersBuffer parametersBuffer,  // 参数缓冲区
    RawParameters rawParameters) {              // 梯度输入（从后续计算传回）
    
    // 根据梯度更新模式选择不同的累加策略
    if (parametersBuffer.exclusiveGradient) {
        // 排他性模式：直接累加，无需原子操作（单线程或无竞争）
        parametersBuffer._gradPtr[particleIdx].density += rawParameters.density;
        parametersBuffer._gradPtr[particleIdx].position += rawParameters.position;
        parametersBuffer._gradPtr[particleIdx].quaternion += rawParameters.quaternion;
        if (!Surfel) {
            // 3D模式：更新所有缩放分量
            parametersBuffer._gradPtr[particleIdx].scale += rawParameters.scale;
        } else {
            // Surfel模式：只更新XY缩放，忽略Z轴（保持面片特性）
            parametersBuffer._gradPtr[particleIdx].scale.xy += rawParameters.scale.xy;
        }
    } else {
        // 非排他性模式：使用原子操作防止竞争（多线程并发安全）
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].density, rawParameters.density);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.x, rawParameters.position.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.y, rawParameters.position.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.z, rawParameters.position.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.x, rawParameters.quaternion.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.y, rawParameters.quaternion.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.z, rawParameters.quaternion.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.w, rawParameters.quaternion.w);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.x, rawParameters.scale.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.y, rawParameters.scale.y);
        if (!Surfel) {
            InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.z, rawParameters.scale.z);
        }
    }
}

/**
 * 计算优化的粒子参数结构 - 展开格式便于计算
 * 
 * 将原始参数转换为优化的计算格式，提高计算效率。
 * 主要优化：将四元数转换为旋转矩阵转置，减少重复计算。
 */
struct Parameters : IDifferentiable {
    float3 position;      // 粒子位置（世界空间坐标）
    float3 scale;         // 粒子缩放（三个主轴的半径）
    float3x3 rotationT;   // 旋转矩阵转置（优化后的格式，用于快速变换）
    float density;        // 粒子密度（控制不透明度）
};

/**
 * 获取计算优化的粒子参数（将四元数转换为旋转矩阵）
 * 
 * 将原始的紧凑参数格式转换为计算优化的格式。
 * 主要优化：将四元数预计算为旋转矩阵转置，避免在渲染循环中重复计算。
 * 
 * @param particleIdx 粒子索引
 * @param parametersBuffer 参数缓冲区
 * @return 优化后的参数结构
 */
[BackwardDifferentiable][ForceInline] Parameters fetchParameters(
    no_diff uint32_t particleIdx,                    // 粒子索引
    no_diff RawParametersBuffer parametersBuffer) {  // 参数缓冲区
    
    // 先获取原始参数
    const RawParameters rawParameters = fetchParametersFromBuffer(particleIdx, parametersBuffer);
    
    // 返回优化的计算格式，将四元数转换为旋转矩阵转置
    return {
        rawParameters.position,                                           // 直接复制位置
        rawParameters.scale,                                             // 直接复制缩放
        transforms.rotationMatrixTranspose(rawParameters.quaternion),    // 四元数→旋转矩阵转置
        rawParameters.density                                            // 直接复制密度
    };
}
/**
 * 将世界空间射线转换为粒子局部空间的标准化射线
 * 
 * 这是高斯相交测试的核心步骤：将任意射线转换到单位球空间，
 * 这样可以用统一的算法处理不同形状和大小的高斯球。
 * 
 * 变换步骤：
 * 1. 平移：将射线移动到以粒子为原点的坐标系
 * 2. 旋转：应用粒子的旋转矩阵转置
 * 3. 缩放：按缩放因子的倒数缩放
 * 
 * @param rayOrigin 世界空间射线起点
 * @param rayDirection 世界空间射线方向（单位向量）
 * @param parameters 粒子参数
 * @param particleRayOrigin 输出：标准化空间射线起点
 * @param particleRayDirection 输出：标准化空间射线方向单位向量
 */
[BackwardDifferentiable][ForceInline] void cannonicalRay(
    in float3 rayOrigin,
    in float3 rayDirection,
    in Parameters parameters,
    out float3 particleRayOrigin,
    out float3 particleRayDirection, ) {
    
    // 1. 计算缩放因子的倒数（用于将高斯椭球标准化为单位球）
    const float3 giscl  = float3(1.0f) / parameters.scale;
    
    // 2. 平移：射线起点相对于粒子中心的位置
    const float3 gposc  = (rayOrigin - parameters.position);
    
    // 3. 旋转：应用旋转矩阵转置将射线转换到粒子的局部坐标系
    const float3 gposcr = mul(parameters.rotationT, gposc);
    
    // 4. 缩放：按比例缩放，将高斯椭球标准化为单位球
    particleRayOrigin   = giscl * gposcr;

    // 对射线方向进行相同的旋转和缩放变换
    const float3 rayDirR = mul(parameters.rotationT, rayDirection);
    const float3 grdu    = giscl * rayDirR;
    particleRayDirection = normalize(grdu);  // 重新单位化保证方向向量的正确性
}

/**
 * 计算标准化空间中射线到原点的最小平方距离
 * 
 * 在标准化的单位球空间中，计算射线到球心（原点）的最小距离的平方。
 * 这个距离用于判断射线是否与高斯椭球相交。
 * 
 * 算法：使用向量叉积计算点到直线的距离
 * |cross(direction, origin)| = |origin| * sin(θ)
 * 其中θ是origin向量与direction向量的夹角
 * 
 * @param canonicalRayOrigin 标准化空间中的射线起点
 * @param canonicalRayDirection 标准化空间中的射线方向（单位向量）
 * @return 最小距离的平方值
 */
[BackwardDifferentiable][ForceInline] float canonicalRayMinSquaredDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    // 使用向量叉积计算垂直距离向量
    const float3 gcrod = cross(canonicalRayDirection, canonicalRayOrigin);
    // 返回距离的平方值（避免开平方根运算提高性能）
    return dot(gcrod, gcrod);
}

/**
 * 计算标准化空间中射线的最大核函数响应
 * 
 * 根据不同的核函数类型，计算射线对高斯粒子的最大影响。
 * 支持多种核函数，从线性到高阶多项式，提供不同的光滑度特性。
 * 
 * 核函数类型：
 * - 0: Linear - 线性函数，边缘清晰但不光滑
 * - 1: Laplacian - 拉普拉斯函数，适度光滑
 * - 2: Quadratic (默认) - 二次函数，平衡性能和质量
 * - 3: Cubic - 三次函数，更光滑
 * - 4: Tesseractic - 四次函数
 * - 5: Quintic - 五次函数
 * - 8: Zenzizenzizenzic - 八次函数，极高光滑度
 * 
 * @param canonicalRayOrigin 标准化空间中的射线起点
 * @param canonicalRayDirection 标准化空间中的射线方向
 * @return 核函数响应值（范围[0,1]）
 */
[BackwardDifferentiable][ForceInline] float canonicalRayMaxKernelResponse<let KernelDegree : int>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    // 计算射线到粒子中心的最小平方距离
    const float grayDist = canonicalRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection);

    // 广义高斯函数，阶数n的缩放因子: s = -4.5/3^n
    switch (KernelDegree) {
    case 8: // Zenzizenzizenzic（八次幂）- 最高光滑度
    {
        /*static const*/ float s = -0.000685871056241;  // -4.5 / 3^8
        const float grayDistSq   = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq);  // exp(s * d^8)
    }
    case 5: // Quintic（五次幂）
    {
        /*static const*/ float s = -0.0185185185185;    // -4.5 / 3^5
        return exp(s * grayDist * grayDist * sqrt(grayDist));  // exp(s * d^5)
    }
    case 4: // Tesseractic（四次幂）
    {
        /*static const*/ float s = -0.0555555555556;    // -4.5 / 3^4
        return exp(s * grayDist * grayDist);  // exp(s * d^4)
    }
    case 3: // Cubic（三次幂）
    {
        /*static const*/ float s = -0.166666666667;     // -4.5 / 3^3
        return exp(s * grayDist * sqrt(grayDist));  // exp(s * d^3)
    }
    case 1: // Laplacian（拉普拉斯）
    {
        /*static const*/ float s = -1.5f;               // -4.5 / 3^1
        return exp(s * sqrt(grayDist));  // exp(s * d^1)
    }
    case 0: // Linear（线性）- 非高斯函数，有界支持
    {
        /* static const */ float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);  // 线性递减，下界为0
    }
    default: // Quadratic（二次幂，默认）- 最常用的高斯函数
    {
        /*static const*/ float s = -0.5f;               // -4.5 / 3^2
        return exp(s * grayDist);  // exp(s * d^2)，标准高斯函数
    }
    }
}

// ========== 核心hitT计算函数 - K-Buffer排序的关键！ ==========
//
// 【功能】：计算光线与高斯粒子相交的参数化距离（hitT）
// 【重要性】：这是K-Buffer局部重排序的核心计算，直接影响渲染质量！
//
// 【数学原理】：
// 在标准化空间中，计算光线到粒子中心投影点的距离
// 光线方程：P(t) = origin + t * direction
// 投影点：t_opt = -dot(origin, direction) / dot(direction, direction)
// 距离：|P(t_opt) - center|，其中center在标准化空间为原点
//
// 【与globalDepth的关键区别】：
// - globalDepth：粒子中心到相机的欧几里得距离（用于粗排序）
// - hitT：光线参数化距离，表示光线上的具体击中位置（用于精排序）
//
// 【为什么需要hitT】：
// 对于大粒子或椭球粒子，粒子中心的深度可能与实际击中表面的深度差异很大
// 例如：粒子中心深度=10，但光线击中前表面深度=8，击中后表面深度=12
// K-Buffer需要精确的击中距离来正确排序透明混合
//
[BackwardDifferentiable][ForceInline] float canonicalRayDistance(
    float3 canonicalRayOrigin,      // 输入：标准化空间中的光线起点
    float3 canonicalRayDirection,   // 输入：标准化空间中的光线方向（单位向量）
    float3 scale) {                 // 输入：粒子的三轴缩放参数
    
    // ========== hitT计算的核心算法 ==========
    // 
    // 步骤1：计算光线方向与起点的点积投影
    // dot(canonicalRayDirection, -canonicalRayOrigin) 得到光线到原点的最近点参数
    const float projectionParam = dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    
    // 步骤2：计算加权方向向量
    // scale * canonicalRayDirection 按粒子缩放调整方向
    // 再乘以投影参数得到最终的加权距离向量
    const float3 grds = scale * canonicalRayDirection * projectionParam;
    
    // 步骤3：计算最终距离
    // sqrt(dot(grds, grds)) 得到向量的模长，即光线参数化距离
    // 这个距离会被返回作为hitT，用于K-Buffer的精确排序
    return sqrt(dot(grds, grds));
    
    // 注意：这个函数的返回值直接成为hit()函数中的depth参数，
    // 也就是K-Buffer中的hitParticle.hitT，是排序的关键依据！
}

// ========== 表面法线计算函数 - 光照和阴影的基础 ==========
//
// 【功能】：计算高斯粒子表面在击中点的法线向量
// 【作用】：用于光照计算、阴影生成和材质渲染
//
// 【Surfel模式说明】：
// - Surfel = Surface Element，表示二维面片而不是三维体积
// - 在Surfel模式下，粒子被压缩为极薄的面片（z轴缩放≈0）
// - 法线计算简化为面片的朝向（通常是z轴方向）
//
// 【法线计算原理】：
// 1. 在Surfel模式下，法线固定为z轴方向（面片法线）
// 2. 应用粒子的缩放和旋转变换到世界空间
// 3. 确保法线朝向与光线方向相反（背面剔除）
//
// 【应用场景】：
// - 光照计算：Lambert光照模型需要表面法线
// - 阴影映射：法线用于计算阴影的软硬程度
// - 材质渲染：镜面反射、折射等需要准确的法线
//
[BackwardDifferentiable][ForceInline] float3 canonicalRayNormal<let Surfel : bool>(
    float3 canonicalRayOrigin,      // 输入：标准化空间中的光线起点
    float3 canonicalRayDirection,   // 输入：标准化空间中的光线方向
    float3 scale,                   // 输入：粒子的三轴缩放参数
    float3x3 rotationT) {           // 输入：粒子的旋转矩阵转置
    
    // ========== Surfel法线计算 ==========
    // 
    // 步骤1：设置基础法线方向
    // 在Surfel模式下，面片的法线默认为Z轴正方向
    // TODO: 未来可能需要支持从几何数据中获取真实法线
    float3 surfelNm = float3(0, 0, 1);  // 基础面片法线（Z轴正方向）
    
    // 步骤2：法线方向歧义解决
    // 确保法线朝向与光线入射方向相反，实现背面剔除效果
    // 如果法线与光线方向夹角小于90度，则翻转法线
    if (dot(surfelNm, canonicalRayDirection) > 0) {
        surfelNm *= -1.0f;  // 翻转法线方向，使其背向光线
    }
    
    // 步骤3：变换到世界空间
    // 应用粒子的缩放和旋转变换，将局部法线转换为世界空间法线
    // surfelNm * scale: 按缩放因子调整法线（对于各向异性缩放很重要）
    // mul(..., rotationT): 应用旋转矩阵转置，将法线转换到世界坐标系
    const float3 worldNormal = mul(surfelNm * scale, rotationT);
    
    // 步骤4：单位化法线
    // 确保返回的法线是单位向量，满足光照计算的要求
    return normalize(worldNormal);
    
    // 注意：这个法线将用于：
    // - Lambert漫反射光照：I = Ia + Id * max(0, dot(N, L))
    // - Phong镜面反射：Is = Ip * pow(max(0, dot(R, V)), shininess)
    // - 阴影映射：确定表面朝向以计算阴影强度
}

/**
 * 射线-高斯粒子相交测试
 * 
 * 核心的相交测试函数，判断射线是否与高斯粒子相交，并计算相关参数。
 * 相交测试基于核函数响应和密度阈值。
 * 
 * 算法步骤：
 * 1. 将世界空间射线转换到粒子的标准化空间
 * 2. 计算核函数响应值
 * 3. 结合密度计算alpha值（不透明度）
 * 4. 如果超过阈值，计算深度和法线
 * 
 * @param rayOrigin 世界空间射线起点
 * @param rayDirection 世界空间射线方向单位向量
 * @param parameters 粒子参数
 * @param alpha 输出：计算得到的不透明度值
 * @param depth 输入输出：命中深度（仅在命中时更新）
 * @param enableNormal 是否需要计算法线
 * @param normal 输入输出：表面法线（仅在命中且enableNormal=true时更新）
 * @return 是否命中（true表示相交）
 */
// ========== densityHit调用链 - 第4层：核心实现（hitT计算的真正位置！）==========
//
// 调用链结构：
// 1. K-Buffer (gutKBufferRenderer.cuh:326) → particles.densityHit()
// 2. C++包装 (shRadiativeGaussianParticles.cuh:115) → particleDensityHit()
// 3. Slang导出 (gaussianParticles.slang:568) → gaussianParticle.hit()
// 4. 【当前层】核心实现 (gaussianParticles.slang:357) → 实际计算hitT
//
// 【本层作用】：hitT的实际计算逻辑
// - 这里是K-Buffer中hitParticle.hitT的真正计算位置！
// - 实现3D高斯粒子的射线相交算法
// - 计算精确的光线参数化距离用于K-Buffer重排序

[BackwardDifferentiable][ForceInline]
bool hit(
    float3 rayOrigin,               // 输入：光线起点（世界空间）
    float3 rayDirection,            // 输入：光线方向（世界空间）
    Parameters parameters,          // 输入：3D高斯粒子参数（位置、旋转、缩放、密度）
    out float alpha,               // 输出：不透明度[0,1]
    inout float depth,             // 输出：hitT - K-Buffer的关键排序参数！
    no_diff bool enableNormal,     // 输入：是否计算法线
    inout float3 normal) {         // 输出：表面法线

    // ========== 第1步：坐标空间变换 ==========
    // 目的：将椭球形高斯粒子变换为单位球，简化相交计算
    // 原理：应用逆变换矩阵，将射线从世界空间变换到粒子的标准化空间
    float3 canonicalRayOrigin;      // 变换后的光线起点
    float3 canonicalRayDirection;   // 变换后的光线方向
    cannonicalRay(
        rayOrigin,                  // 世界空间光线起点
        rayDirection,               // 世界空间光线方向
        parameters,                 // 粒子变换参数（位置、旋转、缩放）
        canonicalRayOrigin,         // 输出：标准化空间起点
        canonicalRayDirection);     // 输出：标准化空间方向

    // ========== 第2步：核函数响应计算 ==========
    // 目的：计算3D高斯核函数在最近点的响应值
    // 原理：exp(-0.5 * minSquaredDistance)，其中minSquaredDistance是光线到粒子中心的最小距离平方
    // 这个响应值决定了粒子对光线的影响强度
    const float maxResponse = canonicalRayMaxKernelResponse<KernelDegree>(
        canonicalRayOrigin,         // 标准化空间的光线起点
        canonicalRayDirection);     // 标准化空间的光线方向

    // ========== 第3步：不透明度计算 ==========
    // 目的：将核函数响应转换为实际的不透明度值
    // 公式：alpha = min(MaxAlpha, kernelResponse * particleDensity)
    // 限制最大值以保证反向传播的数值稳定性
    alpha = min(MaxParticleAlpha, maxResponse * parameters.density);
    
    // ========== 第4步：相交有效性验证 ==========
    // 使用双重阈值检查确保相交的有效性：
    // 1. maxResponse > MinParticleKernelDensity：核函数响应足够强
    // 2. alpha > MinParticleAlpha：最终不透明度足够高
    const bool acceptHit = ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));
    
    if (acceptHit)
    {
        // ========== 第5步：关键！计算hitT距离 ==========
        // 这是K-Buffer排序的核心：计算光线参数化距离
        // depth = canonicalRayDistance() 返回的是光线参数t，使得：
        //   hitPoint = rayOrigin + t * rayDirection
        // 
        // 重要区别：
        // - globalDepth（全局排序）= distance(camera, particleCenter)
        // - hitT（K-Buffer排序）= 光线与粒子表面相交的参数化距离
        // 
        // 为什么需要hitT而不是globalDepth？
        // 例子：大粒子的情况下，粒子中心可能很远，但光线击中粒子前表面很近
        depth = canonicalRayDistance(canonicalRayOrigin,     // 标准化光线起点
                                    canonicalRayDirection,   // 标准化光线方向  
                                    parameters.scale);       // 粒子缩放参数
        
        // ========== 第6步：可选的法线计算 ==========
        // 仅在需要时计算表面法线（节省计算资源）
        if (enableNormal)
        {
            normal = canonicalRayNormal<Surfel>(canonicalRayOrigin,     // 标准化光线起点
                                              canonicalRayDirection,   // 标准化光线方向
                                              parameters.scale,        // 粒子缩放
                                              parameters.rotationT);   // 粒子旋转转置矩阵
        }
    }
    
    // 返回相交有效性
    // 如果返回true，则depth参数包含了K-Buffer需要的精确hitT值！
    return acceptHit;
}

// ========== integrateHit调用链 - 第4层：核心实现算法 ==========
//
// 🗺️ 【调用链终点】：
// 1. K-Buffer (gutKBufferRenderer.cuh:244) → particles.densityIntegrateHit()
// 2. C++包装 (shRadiativeGaussianParticles.cuh:344) → particleDensityIntegrateHit()
// 3. Slang导出 (gaussianParticles.slang:873) → gaussianParticle.integrateHit<false>()
// 4. 【当前层】核心实现 (gaussianParticles.slang:551) → 实际Alpha混合计算
//
// 【本层作用】：体积渲染核心算法 - Alpha混合的数学实现
// - 这是整个调用链的最底层，包含真正的数学计算逻辑
// - 实现标准的Alpha混合公式和透射率更新
// - 支持双向渲染模式：前向后（K-Buffer使用）和后向前
// - 通过模板参数<backToFront>在编译时生成不同的优化版本
//
// 🎨 【数学原理】：
// Alpha混合公式（Porter-Duff理论）：
// C_out = α × C_src + (1-α) × C_dst  // 标准Alpha混合
// T_out = T_in × (1-α)                    // 透射率更新
// weight = α × T_in                        // 实际贡献权重
//
// 【渲染模式对比】：
// 前向后模式 (backToFront=false)：
//   weight = alpha * transmittance           // 正确的物理权重
//   integratedDepth += depth * weight        // 累积平均深度
//   transmittance *= (1 - alpha)             // 透射率衰减
// 后向前模式 (backToFront=true)：
//   weight = alpha                           // 直接使用alpha值
//   integratedDepth = lerp(old, new, alpha)  // 线性插值混合
//   transmittance *= (1 - alpha)             // 同样的透射率更新
//
// 📊 【性能优化】：
// - [BackwardDifferentiable] 支持自动微分，用于神经网络训练
// - [ForceInline] 强制内联展开，减少函数调用开销
// - 模板特化 <let backToFront : bool> 编译时分支消除
// - no_diff 标记不参与梯度计算的参数，优化内存使用
//
/**
 * 积分单次相交结果（体积渲染混合）- 核心实现算法
 *
 * 【核心功能】：
 * 这是3DGUT渲染系统中最关键的数学实现，实现体积渲染中单个粒子的贡献积分。
 * 支持两种混合模式，通过模板参数在编译时优化：
 * 1. 前向后模式：用于正向穿越，直接累加权重值 (K-Buffer使用)
 * 2. 后向前模式：用于Over混合，使用线性插值 (Alpha混合)
 *
 * 这个函数处理：
 * - 深度积分（用于Z-buffer或距离估计）
 * - 法线积分（用于光照计算）
 * - 透射率更新（控制后续粒子的影响）
 *
 * @param alpha 当前粒子的不透明度 [0.0, 1.0]
 * @param transmittance 输入输出：累积透射率 [0.0, 1.0]
 * @param depth 当前粒子的深度（击中距离）
 * @param integratedDepth 输入输出：累积深度（Z-buffer）
 * @param enableNormal 是否需要处理法线计算
 * @param normal 当前粒子的表面法线向量
 * @param integratedNormal 输入输出：累积法线向量
 * @return 当前粒子的混合权重（用于后续颜色混合）
 */
// Slang	<let T : type>	func<let N : int>()
// C++ template <> template <int N> func()
[BackwardDifferentiable][ForceInline]
float integrateHit<let backToFront : bool>(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    no_diff bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
   // ========== 步骤1：混合权重计算 ==========
   // 关键：根据渲染模式选择不同的权重计算方式
   // - K-Buffer使用前向后模式：weight = alpha * transmittance （物理正确）
   // - 后向前模式直接使用alpha值：weight = alpha
   const float weight = backToFront ? alpha : alpha * transmittance;
   
   // ========== 步骤2：深度和法线积分 ==========
   if (backToFront)
   {
       // 后向前模式：Over混合，使用线性插值
       // 公式：result = lerp(old, new, alpha) = old * (1-alpha) + new * alpha
       integratedDepth = lerp(integratedDepth, depth, alpha);
       if (enableNormal)
       {
           integratedNormal = lerp(integratedNormal, normal, alpha);
       }
   }
   else 
   {
       // 前向后模式：正向穿越，直接累加权重值
       // 公式：result += value * weight (加权平均)
       integratedDepth += depth * weight;
       if (enableNormal) 
       {
            integratedNormal += normal * weight;
       }
   }

   // ========== 步骤3：透射率更新 ==========
   // 关键：每个粒子都会减少后续光线的透射率
   // 公式：T_new = T_old * (1 - alpha) (指数衰减模型)
   // 物理意义：光线被粒子部分吸收/散射，剩余能量减少
   transmittance *= (1 - alpha);
   
   // ========== 步骤4：返回权重用于颜色混合 ==========
   // 返回的权重将由featureIntegrateFwd使用，实现颜色的Alpha混合
   return weight;  // 当前粒子的实际混合权重
}

/**
 * 📋 DENSITY反向传播调用链（第4层/4层）- 高斯几何相交的数学实现
 * 
 * 🔄 调用路径（到达最终数学层）：
 * 1. 上3层：gutKBufferRenderer.cuh:205 → shRadiativeGaussianParticles.cuh:437 → gaussianParticles.slang:1121
 * 2. 【当前层】gaussianParticles.slang:673 → hit() + integrateHit<backToFront>()
 * 
 * 🎯 本层职责：高斯椭球几何计算的完整实现
 * - 射线-椭球相交测试：完整的3D几何数学计算
 * - 坐标变换链：世界坐标→粒子坐标→标准化坐标
 * - 高斯核函数：计算粒子对光线的响应强度
 * - 体积积分：透射率和深度的数值积分
 * 
 * 🧮 几何数学核心算法：
 * 1. 坐标变换：P_norm = Scale^(-1) × Rotation^(-1) × (P_world - Position)
 * 2. 距离计算：d = |cross(ray_dir, ray_pos)| （点到直线距离）
 * 3. 高斯响应：α = min(0.99, exp(-0.5×d²) × density)
 * 4. 体积积分：T_new = T_old × (1-α), depth_new = depth_old + α×hit_depth
 * 
 * 🔍 反向传播中的梯度计算：
 * ∂L/∂α → ∂α/∂(位置,旋转,缩放,密度) 的完整链式法则应用
 * - ∂α/∂位置：通过距离计算的坐标变换梯度  
 * - ∂α/∂旋转：通过旋转矩阵对四元数的导数
 * - ∂α/∂缩放：通过椭球变换的缩放导数
 * - ∂α/∂密度：高斯响应函数的直接导数
 * 
 * 🚀 Slang自动微分特性：
 * - [BackwardDifferentiable] 支持编译器自动生成反向版本
 * - let模板参数：编译时优化，生成专用版本（前向后/后向前）
 * - 自动梯度管理：无需手动实现复杂的几何导数
 * 
 * ⚡ 渲染模式支持：
 * - backToFront=false: 前向后混合 (标准K-Buffer)
 * - backToFront=true:  后向前混合 (深度排序)
 * - 编译时分支：零运行时开销的模式切换
 *
 * 【函数工作流程】：
 * Step 1: 从缓冲区获取粒子参数 (fetchParameters)
 * Step 2: 执行射线-椭球相交测试 (hit)
 * Step 3: 如果相交，计算alpha混合贡献 (integrateHit)
 * Step 4: 返回权重值用于后续处理
 * 
 * @param rayOrigin         射线起点 (世界坐标，不参与微分)
 * @param rayDirection      射线方向 (世界坐标，不参与微分)
 * @param particleIdx       粒子索引 (不参与微分)
 * @param parametersBuffer  参数缓冲区 (不参与微分)
 * @param transmittance     输入输出：射线透射率 (参与微分)
 * @param integratedDepth   输入输出：积分深度 (参与微分)
 * @param enableNormal      是否计算法线 (不参与微分)
 * @param integratedNormal  输入输出：积分法线 (参与微分)
 * @return 返回混合权重值，0.0表示无相交
 */
[BackwardDifferentiable][ForceInline]
float processHitFromBuffer<let backToFront : bool>(
    no_diff float3 rayOrigin,
    no_diff float3 rayDirection,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    inout float transmittance,
    inout float integratedDepth,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    // ========== 步骤1：初始化输出变量 ==========
    float alpha = 0.0f;    // 粒子不透明度 (输出)
    float depth;           // 相交距离 (输出)  
    float3 normal;         // 表面法线 (输出)
    
    // ========== 步骤2：射线-高斯椭球相交测试 ==========
    // 🎯 关键函数：hit() 包含完整的高斯几何数学计算
    // 内部算法：
    // 1. fetchParameters()：从缓冲区获取粒子的几何参数（位置,旋转,缩放,密度）
    // 2. 坐标变换链：世界空间 → 粒子局部空间 → 标准化椭球空间
    // 3. 点到直线距离：d = |cross(ray_dir, ray_pos_relative)|
    // 4. 高斯核函数：response = exp(-0.5×d²)
    // 5. alpha计算：α = min(0.99, response × density)
    // 6. 早期剔除：如果α太小则直接返回false
    if (hit(rayOrigin,                                          // 射线起点（世界坐标）
            rayDirection,                                       // 射线方向（世界坐标）
            fetchParameters(particleIdx, parametersBuffer),    // 粒子几何参数结构
            alpha,                                             // [输出] 计算得到的不透明度
            depth,                                             // [输出] 击中距离
            enableNormal,                                      // 是否需要计算法线
            normal))                                           // [输出] 表面法线
    {
        // ========== 步骤3：体积积分处理 ==========
        // ✅ 相交成功！调用积分函数处理透射率和深度累积
        // 🚀 integrateHit<backToFront>() 实现体积渲染的数学积分
        // 核心公式：
        // - 透射率：T_new = T_old × (1-α) 
        // - 深度：depth_new = depth_old + α × hit_depth
        // - 法线：normal_new = normal_old + α × hit_normal
        // - 返回权重：weight = α × T_old （用于颜色混合）
        return integrateHit<backToFront>(alpha,           // 当前粒子的不透明度
                                         transmittance,   // [输入输出] 射线剩余透射率
                                         depth,           // 击中距离，用于深度积分
                                         integratedDepth, // [输入输出] 累积深度值
                                         enableNormal,    // 是否处理法线
                                         normal,          // 当前表面法线
                                         integratedNormal); // [输入输出] 累积法线
    }
    
    // ========== 步骤4：无相交的默认情况 ==========
    // ❌ 射线未与椭球相交，返回零权重（粒子无贡献）
    return 0.0f;
}

/**
 * 计算高斯粒子的协方差矩阵 - 椭球几何的数学核心
 * 
 * 【数学原理】：
 * 协方差矩阵 Σ 完全描述了高斯分布的形状和朝向：
 * Σ = R * S * S^T * R^T
 * 
 * 其中：
 * - R: 旋转矩阵 (3×3)，控制椭球的朝向
 * - S: 对角缩放矩阵 diag(scale_x, scale_y, scale_z)，控制椭球各轴长
 * 
 * @param parameters 粒子参数，包含位置、缩放和旋转信息
 * @return 3×3 协方差矩阵，描述高斯椭球的完整几何形状
 */
[BackwardDifferentiable][ForceInline] float3x3 computeCovariance(
    in Parameters parameters) {
    // 🧮 数学公式：Σ = R * S * S^T * R^T = (R*S) * (R*S)^T
    // 步骤1: 计算 StRt = R * S (旋转矩阵右乘缩放)
    // 这里利用了 parameters.rotationT 已经是转置，所以实际计算的是 S * R^T
    const float3x3 StRt = float3x3(
        parameters.scale.x * parameters.rotationT[0],  // 第一列：X轴缩放 * 旋转第一行
        parameters.scale.y * parameters.rotationT[1],  // 第二列：Y轴缩放 * 旋转第二行  
        parameters.scale.z * parameters.rotationT[2]   // 第三列：Z轴缩放 * 旋转第三行
    );
    
    // 步骤2: 计算最终协方差矩阵 Σ = StRt^T * StRt
    // transpose(StRt) 得到 (S*R^T)^T = R*S^T，然后右乘 StRt = S*R^T
    // 结果：(R*S^T) * (S*R^T) = R * S^T * S * R^T = R * S * S^T * R^T = Σ
    return mul(transpose(StRt), StRt);
}

/**
 * 计算从光源到粒子的入射方向向量 - 光照计算的基础
 * 【数学计算】：
 * 入射方向 = normalize(粒子位置 - 光源位置)
 * @param parameters    粒子参数，包含位置信息
 * @param sourcePosition 光源位置 (不参与微分，通常是相机位置)
 * @return 归一化的入射方向向量 (从光源指向粒子)
 */
[BackwardDifferentiable] [ForceInline]
float3 incidentDirectionFromParameters(
    Parameters parameters,
    no_diff float3 sourcePosition
)
{
    // 🌟 计算从光源到粒子的方向向量并归一化
    // 向量减法：粒子位置 - 光源位置 = 从光源指向粒子的向量
    // normalize: 转换为单位向量，只保留方向信息
    return normalize(parameters.position - sourcePosition);
}

/**
 * 从缓冲区计算入射方向 - incidentDirectionFromParameters 的缓冲区版本
 * 
 * 【功能】：
 * 这是 incidentDirectionFromParameters 的便利包装函数，
 * 自动从缓冲区获取粒子参数后计算入射方向。
 * 
 * 【使用场景】：
 * 在渲染循环中，通常只有粒子索引而不是直接的参数对象，
 * 这个函数避免了手动调用 fetchParameters 的步骤。
 * 
 * 【返回值特性】：
 * 返回值标记为 no_diff，意味着这个方向向量本身不参与梯度计算，
 * 但通过 fetchParameters 获取的粒子位置仍然可以接收梯度。
 * 
 * 【调用链】：
 * incidentDirectionFromBuffer → fetchParameters → incidentDirectionFromParameters
 * 
 * @param particleIdx       粒子索引 (不参与微分)
 * @param parametersBuffer  参数缓冲区 (不参与微分)  
 * @param sourcePosition    光源位置 (不参与微分)
 * @return 归一化的入射方向向量 (标记为不参与微分)
 */
[BackwardDifferentiable][ForceInline]
no_diff float3 incidentDirectionFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float3 sourcePosition
)
{
    // 🔄 两步式调用：先获取参数，再计算方向
    // 1. fetchParameters: 从缓冲区获取粒子的完整参数
    // 2. incidentDirectionFromParameters: 使用参数计算入射方向
    return incidentDirectionFromParameters(
        fetchParameters(particleIdx, parametersBuffer),
        sourcePosition
    );
}
} // namespace gaussianParticle

// ------------------------------------------------------------------------------------------------------------------
// 导出函数入口点 (CUDA Device Export Functions)
//
// 以下函数是为CUDA设备端导出的入口点，可以被C++/CUDA代码调用。
// 这些函数封装了核心的高斯粒子计算功能，提供统一的接口。

/**
 * 获取粒子密度计算参数
 * 
 * 从缓冲区获取指定粒子的计算优化参数，用于密度计算。
 * 返回的参数包含预计算的旋转矩阵转置，提高计算效率。
 * 
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数配置
 * @return 优化后的粒子参数结构
 */
[CudaDeviceExport] 
inline gaussianParticle.Parameters particleDensityParameters(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters) 
{
    return gaussianParticle.fetchParameters(
        particleIdx,
        commonParameters.parametersBuffer);
}

// ========== densityHit调用链 - 第3层：Slang导出接口 ==========
//
// 调用链结构：
// 1. K-Buffer (gutKBufferRenderer.cuh:326) → particles.densityHit()
// 2. C++包装 (shRadiativeGaussianParticles.cuh:115) → particleDensityHit()
// 3. 【当前层】Slang导出 (gaussianParticles.slang:568) → gaussianParticle.hit()
// 4. 核心实现 (gaussianParticles.slang:357) → 实际计算hitT
//
// 【本层作用】：跨语言边界接口
// - [CudaDeviceExport] 使得Slang函数可以被CUDA C++代码调用
// - 提供稳定的ABI边界，隔离Slang内部实现细节
// - 简单的函数调用转发，不进行任何计算

/**
 * 粒子密度相交测试 - Slang导出接口
 * 
 * 功能：为CUDA C++代码提供调用Slang实现的densityHit功能的接口
 * 作用：这是一个纯转发函数，将调用直接传递给内部的gaussianParticle.hit()
 * 
 * 重要：本函数不做任何计算，只是语言边界的桥梁
 * - CUDA C++ → [CudaDeviceExport] → Slang内部实现
 * - 确保参数类型兼容性和调用约定正确性
 * 
 * @param rayOrigin 射线起点（世界空间坐标）
 * @param rayDirection 射线方向（世界空间单位向量）
 * @param parameters 粒子参数（位置、旋转、缩放、密度等）
 * @param alpha 输出：不透明度值 [0,1]
 * @param depth 输出：相交深度（这就是K-Buffer需要的hitT！）
 * @param enableNormal 是否计算表面法线
 * @param normal 输出：表面法线向量
 * @return 是否发生有效相交（通过阈值检测）
 */
[CudaDeviceExport]
inline bool particleDensityHit(
    float3 rayOrigin,                           // 输入：光线起点
    float3 rayDirection,                        // 输入：光线方向
    gaussianParticle.Parameters parameters,     // 输入：粒子参数结构
    out float alpha,                           // 输出：不透明度
    out float depth,                           // 输出：hitT距离（关键！）
    bool enableNormal,                         // 输入：是否需要法线
    out float3 normal)                         // 输出：表面法线
{
    // ========== 完整渲染碰撞检测：直接转发到核心实现 ==========
    //
    // 【函数职责】：
    // 这是渲染管线中最常用的碰撞检测接口，提供完整的相交信息：
    // 1. ✅ Alpha值计算：用于透明度混合
    // 2. ✅ 精确深度：用于深度缓冲和排序  
    // 3. ✅ 表面法线：用于光照计算
    // 4. ✅ 自动微分：支持梯度反向传播
    //
    // 【与其他hit函数的对比】：
    // - 本函数：完整功能，适合最终渲染
    // - HitCustom：只几何测试，适合预筛选
    // - HitInstance：预变换优化，适合批量处理
    //
    // 【关键输出说明】：
    // depth参数返回的是精确的hitT值，这对K-Buffer排序至关重要：
    // - globalDepth：粒子中心到相机的距离（用于粗排序）
    // - hitT (depth)：光线实际击中粒子表面的参数化距离（用于精排序）
    //
    // 【调用链路径】：
    // C++ → [CudaDeviceExport] → gaussianParticle.hit() → 完整数学计算
    //
    return gaussianParticle.hit(rayOrigin,      // 转发：光线起点
                                rayDirection,    // 转发：光线方向
                                parameters,      // 转发：粒子参数
                                alpha,          // 转发：不透明度输出 (关键渲染信息!)
                                depth,          // 转发：hitT输出 (精确深度!)
                                enableNormal,   // 转发：法线计算标志
                                normal);        // 转发：表面法线输出
}

// ========== particleDensityIntegrateHit调用链 - 第3层：Slang导出接口 ==========
//
// 🔗 【调用链位置】：
// 1. K-Buffer (gutKBufferRenderer.cuh:244) → particles.densityIntegrateHit()
// 2. C++包装 (shRadiativeGaussianParticles.cuh:344) → particleDensityIntegrateHit()
// 3. 【当前层】Slang导出 (gaussianParticles.slang:873) → gaussianParticle.integrateHit<false>()
// 4. 核心实现 (gaussianParticles.slang:551) → 实际Alpha混合计算
//
// 【本层作用】：跨语言边界的稳定导出接口
// - [CudaDeviceExport] 使得Slang函数可以被CUDA C++代码调用
// - 提供稳定的ABI边界，隔离Slang内部实现细节
// - 纯转发函数，不进行任何计算，只做语言边界的桥梁
// - 确保参数类型兼容性和调用约定正确性
//
// 【渲染模式】：
// - 使用前向后遍历模式 (backToFront=false)
// - 适用于正向穿越的渲染算法 (front-to-back ray marching)
// - 直接累积模式：integratedDepth += depth * weight
//
// 📊 【核心公式】（将在第4层实现）：
// const float weight = alpha * transmittance;  // 权重计算
// integratedDepth += depth * weight;           // 深度累积
// transmittance *= (1 - alpha);               // 透射率更新
// return weight;                               // 返回权重用于颜色混合
//
/**
 * 粒子密度相交积分（前向模式）- Slang导出接口
 * 
 * 功能：为CUDA C++代码提供调用Slang实现的densityIntegrateHit功能的接口
 * 作用：这是一个纯转发函数，将调用直接传递给内部的gaussianParticle.integrateHit()
 * 
 * 重要：本函数不做任何计算，只是语言边界的桥梁
 * - CUDA C++ → [CudaDeviceExport] → Slang内部实现
 * - 确保参数类型兼容性和调用约定正确性
 * 
 * @param alpha 粒子的不透明度 [0.0, 1.0]
 * @param transmittance 输入输出：累积透射率（会被修改）
 * @param depth 粒子深度（击中距离）
 * @param integratedDepth 输入输出：累积深度（会被修改）
 * @param enableNormal 是否需要处理法线计算
 * @param normal 粒子表面法线向量
 * @param integratedNormal 输入输出：累积法线向量（会被修改）
 * @return 粒子的混合权重（alpha * transmittance）
 */
[CudaDeviceExport] inline float particleDensityIntegrateHit(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    in bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal) 
{
    // ========== 直接转发到核心实现层 ==========
    // 功能：将所有参数原封不动地传递给真正的算法实现
    // 模板参数：<false> 表示使用前向后渲染模式
    // 返回值：直接返回核心算法计算的权重值
    return gaussianParticle.integrateHit<false>(
        alpha,              // 粒子不透明度
        transmittance,      // 当前透射率（输入输出）
        depth,              // 粒子深度
        integratedDepth,    // 积分深度（输入输出）
        enableNormal,       // 法线计算标志
        normal,             // 当前法线
        integratedNormal);  // 积分法线（输入输出）
}

/**
 * 直接从缓冲区处理粒子相交（前向模式）
 * 
 * 高效的一站式函数，组合了参数获取、相交测试和积分操作。
 * 适用于渲染循环中的直接调用，减少函数调用开销。
 * 
 * @param rayOrigin 射线起点
 * @param rayDirection 射线方向
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数
 * @param transmittance 输入输出：透射率
 * @param integratedDepth 输入输出：积分深度
 * @param enableNormal 是否处理法线
 * @param integratedNormal 输入输出：积分法线
 * @return 粒子的混合权重（无相交时为0）
 */
[CudaDeviceExport]
inline float particleDensityProcessHitFwdFromBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    inout float transmittance,
    inout float integratedDepth,
    in bool enableNormal,
    inout float3 integratedNormal) 
{
    return gaussianParticle.processHitFromBuffer<false>(
        rayOrigin,
        rayDirection,
        particleIdx,
        commonParameters.parametersBuffer,
        transmittance,
        integratedDepth,
        enableNormal,
        integratedNormal);
}

/**
 * 📋 DENSITY反向传播调用链（第3层/4层）- Slang自动微分核心
 *
 * 🔄 调用路径：
 * 1. 上层：gutKBufferRenderer.cuh:205 → particles.densityProcessHitBwdToBuffer<false>()
 * 2. 上层：shRadiativeGaussianParticles.cuh:437 → particleDensityProcessHitBwdToBuffer()
 * 3. 【当前层】gaussianParticles.slang:1063 → bwd_diff(gaussianParticle.processHitFromBuffer<false>)()
 * 4. 下层：gaussianParticles.slang:内部实现 → 高斯几何变换的完整梯度计算
 *
 * 🎯 本层职责：体积渲染反向传播的数学核心
 * - 状态恢复：从积分结果逆推积分前状态（逆向工程）
 * - 微分对象创建：使用DifferentialPair封装值和梯度
 * - 自动微分调用：触发Slang编译器生成的反向函数
 * - 梯度自动更新：直接写入全局参数梯度缓冲区
 *
 * 🧮 体积渲染的反向模式数学原理：
 * 前向积分公式：
 *   transmittance_new = transmittance_old × (1 - α)
 *   depth_new = depth_old + α × hit_depth  
 *   normal_new = normal_old + α × hit_normal
 *
 * 反向传播（链式法则 + 状态恢复）：
 *   weight = 1.0 / (1 - α)  // 反向权重
 *   transmittance_old = transmittance_new × weight  // 恢复原始透射率
 *   depth_old = (depth_new - α × hit_depth) × weight  // 恢复原始深度
 *
 * 🔍 梯度计算链条：
 * ∂L/∂(几何参数) = ∂L/∂α × ∂α/∂(几何参数)
 * 其中 ∂α/∂(几何参数) 涉及复杂的高斯核函数导数计算：
 * - ∂α/∂位置：坐标变换链的导数
 * - ∂α/∂旋转：四元数到矩阵的导数
 * - ∂α/∂缩放：椭球变换的导数
 * - ∂α/∂密度：高斯响应函数的导数
 *
 * 🚀 Slang自动微分技术：
 * - DifferentialPair<T>：双数实现，同时跟踪值和梯度
 * - bwd_diff()：编译器自动生成的反向函数
 * - 梯度自动传播：无需手工实现复杂的链式法则计算
 * - 原子更新：多线程安全的梯度累积
 *
 * @param rayOrigin             射线起点 (世界坐标，不参与微分)
 * @param rayDirection          射线方向 (世界坐标，不参与微分)  
 * @param particleIdx           粒子索引 (不参与微分)
 * @param commonParameters      公共参数缓冲区 (不参与微分)
 * @param alpha                 前向：粒子不透明度值 (输入)
 * @param alphaGrad             反向：不透明度梯度 (输入)
 * @param transmittance         前向：透射率 | 反向：恢复后的透射率 (输入输出)
 * @param transmittanceGrad     反向：透射率梯度 (输入输出)
 * @param depth                 前向：击中深度 (输入)
 * @param integratedDepth       前向：积分深度 | 反向：恢复后的积分深度 (输入输出)
 * @param integratedDepthGrad   反向：积分深度梯度 (输入输出)
 * @param enableNormal          是否处理法线 (不参与微分)
 * @param normal                前向：表面法线 (输入)
 * @param integratedNormal      前向：积分法线 | 反向：恢复后的积分法线 (输入输出)
 * @param integratedNormalGrad  反向：积分法线梯度 (输入输出)
 */
[CudaDeviceExport]
void particleDensityProcessHitBwdToBuffer(
    // 射线信息（输入，不参与梯度）
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    // 前向传播的值和对应梯度
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float depth,
    inout float integratedDepth,
    inout float integratedDepthGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad) 
{
    if (alpha > 0.0f)
    {
        // ========== 步骤1：前向状态恢复（逆向工程） ==========
        // 计算反向权重，用于从积分结果恢复积分前状态
        const float weight = 1.0f / (1.0f - alpha);

        // 🔄 透射率状态恢复：T_before = T_after / (1 - α)
        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);
        
        // ========== 步骤2：深度积分状态恢复 ==========
        // 🔄 深度状态恢复：depth_before = (depth_after - α×hit_depth) / (1 - α)
        integratedDepth = (integratedDepth - depth * alpha) * weight;
        DifferentialPair<float> integratedDepthDiff = DifferentialPair<float>(integratedDepth, integratedDepthGrad);

        // ========== 步骤3：法线积分状态恢复（可选） ==========
        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            // 🔄 法线状态恢复：normal_before = (normal_after - α×hit_normal) / (1 - α)
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            // 不处理法线时设置零值微分对
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        // ========== 步骤4：调用Slang自动微分引擎 ==========
        // 🚀 关键：bwd_diff() 自动生成复杂的几何变换梯度计算
        // 功能：∂L/∂α → ∂L/∂(位置,旋转,缩放,密度) 的完整变换
        // 结果：自动更新全局高斯参数缓冲区梯度
        bwd_diff(gaussianParticle.processHitFromBuffer<true>)(
            rayOrigin,                          // 光线起点（世界坐标）
            rayDirection,                       // 光线方向（世界坐标）
            particleIdx,                        // 粒子索引
            commonParameters.parametersBuffer,   // 高斯参数缓冲区（数据+梯度）
            transmittanceDiff,                  // 透射率的微分对
            integratedDepthDiff,                // 深度的微分对
            enableNormal,                       // 法线处理标志
            integratedNormalDiff,               // 法线的微分对
            alphaGrad);                         // alpha梯度输入

        // ========== 步骤5：提取更新后的梯度 ==========
        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedDepthGrad = integratedDepthDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }
    }
}

/**
 * 自定义高斯粒子碰撞检测 - K-Buffer渲染器的专用函数
 *
 * 【核心功能】：
 * 这是为K-Buffer深度排序渲染优化的碰撞检测函数，专门用于快速确定
 * 射线是否与高斯粒子相交，并计算精确的击中距离用于深度排序。
 *
 * 【与标准hit函数的区别】：
 * - 标准hit：计算alpha值，用于颜色混合
 * - 本函数：只计算几何相交，用于深度排序
 * - 性能优化：跳过复杂的核函数计算，只做几何测试
 *
 * 【工作流程】：
 * 1. 🔍 参数获取：从缓冲区获取指定粒子的几何参数
 * 2. 🎯 坐标变换：将世界空间射线转换到粒子标准化空间
 * 3. 📊 核响应计算：计算最大可能的核函数响应值
 * 4. 📏 距离计算：计算射线到粒子中心投影的参数化距离
 * 5. ✅ 多重检测：执行距离范围和几何相交的联合测试
 *
 * 【数学原理】：
 * 击中距离 = canonicalRayDistance() 计算的是射线参数t值：
 *   hit_point = rayOrigin + t * rayDirection
 * 其中t就是hitDistance，代表沿射线方向的参数化距离。
 *
 * 【K-Buffer应用】：
 * K-Buffer需要对每个像素维护K个最近的碰撞点：
 * 1. 快速筛选：使用本函数确定相交性
 * 2. 距离排序：使用hitDistance进行深度排序  
 * 3. 精确渲染：只对排序后的前K个粒子进行完整渲染
 *
 * 【性能优化策略】：
 * - 🚀 几何裁剪：通过距离范围快速剔除不相关粒子
 * - 🎯 精确测试：使用canonicalRayMinSquaredDistance做最终验证
 * - 📈 批量处理：支持并行处理大量射线-粒子对
 *
 * @param rayOrigin                    射线起点 (世界坐标)
 * @param rayDirection                 射线方向 (世界坐标，单位向量)
 * @param particleIdx                  目标粒子索引
 * @param commonParameters             公共参数缓冲区
 * @param minHitDistance              最小有效击中距离 (近裁剪面)
 * @param maxHitDistance              最大有效击中距离 (远裁剪面)
 * @param maxParticleSquaredDistance  最大粒子平方距离阈值 (几何相交判断)
 * @param hitDistance                 输出：参数化击中距离 (射线参数t值)
 * @return 是否发生有效相交 (通过所有几何和距离测试)
 */
[CudaDeviceExport]
bool particleDensityHitCustom(
    float3 rayOrigin,
    float3 rayDirection,
    int32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    // ========== K-Buffer优化：几何碰撞检测管线 ==========
    //
    // 【设计思想】：
    // 本函数专为K-Buffer深度排序优化，只关心几何相交而非渲染细节。
    // 跳过昂贵的alpha和法线计算，专注于快速的距离测试。
    //
    
    // 步骤1: 🔍 参数获取 - 从GPU缓冲区读取粒子几何参数
    gaussianParticle.Parameters parameters = gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer);

    // 步骤2: 🎯 坐标空间变换 - 将世界空间转换为粒子标准化空间
    // 这是高斯相交的核心：将椭球标准化为单位球，简化后续计算
    float3 canonicalRayOrigin;      // 变换后的射线起点
    float3 canonicalRayDirection;   // 变换后的射线方向
    gaussianParticle.cannonicalRay(
        rayOrigin,                  // 输入：世界空间射线起点
        rayDirection,               // 输入：世界空间射线方向
        parameters,                 // 输入：粒子变换参数(位置、旋转、缩放)
        canonicalRayOrigin,         // 输出：标准化空间射线起点
        canonicalRayDirection);     // 输出：标准化空间射线方向

    // 步骤3: 📊 核函数响应计算 - 计算射线在高斯核函数中的最大响应值
    // 注意：这里计算了maxResponse但实际上在后续判断中并未使用
    // 可能是为了保持与完整hit函数的接口一致性或未来扩展预留
    const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse<gaussianParticle.KernelDegree>(
        canonicalRayOrigin,         // 标准化射线起点
        canonicalRayDirection);     // 标准化射线方向

    // 步骤4: 📏 几何距离计算 - 计算射线参数t值 (关键!)
    // 这是K-Buffer排序的核心数据：射线到高斯中心投影的参数化距离
    hitDistance = gaussianParticle.canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);

    // 步骤5: ✅ 三重验证 - 联合几何测试确保有效相交
    // 三个条件必须同时满足才算有效击中：
    return (hitDistance > minHitDistance) &&           // 条件1: 距离范围检查 (近裁剪面)
           (hitDistance < maxHitDistance) &&           // 条件2: 距离范围检查 (远裁剪面)  
           (gaussianParticle.canonicalRayMinSquaredDistance(   // 条件3: 几何相交验证
                canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
    //
    // 【最终判断逻辑】：
    // 只有当射线在有效距离范围内，且确实与高斯椭球几何相交时，
    // 才返回true并输出有效的hitDistance供K-Buffer排序使用。
}

/**
 * 实例化高斯粒子碰撞检测 - 预变换射线的超高速检测
 *
 * 【核心功能】：
 * 这是最高性能的碰撞检测函数，专门处理已经过坐标变换的射线。
 * 跳过复杂的坐标变换步骤，直接在标准化空间中进行几何计算。
 *
 * 【与其他hit函数的性能对比】：
 * - particleDensityHit：完整计算，包含alpha值 (最慢，最完整)
 * - particleDensityHitCustom：几何检测，包含坐标变换 (中等速度)
 * - 【本函数】：预变换射线，纯几何计算 (最快，适合批量处理)
 *
 * 【预处理优势】：
 * 输入的射线已经过 canonicalRay() 变换，意味着：
 * 1. 🎯 椭球 → 单位球：高斯椭球已标准化为单位球
 * 2. 🔧 坐标归一化：所有计算都在标准化空间进行
 * 3. ⚡ 跳过变换：省去耗时的矩阵变换步骤
 *
 * 【数学原理 - 射线到原点最近点】：
 * 在标准化空间中，计算射线到原点(椭球中心)的最近距离参数：
 *
 * 射线方程：P(t) = origin + t * direction
 * 目标：最小化 |P(t)|² = |origin + t * direction|²
 *
 * 求导并令其为0：
 * d/dt[|origin + t * direction|²] = 0
 * 2(origin + t * direction) · direction = 0
 * 2(origin · direction + t * |direction|²) = 0
 * t = -(origin · direction) / |direction|²
 *
 * 【实现细节】：
 * numerator = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection)
 * denominator = rcp(dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection))
 * hitDistance = numerator * denominator
 *
 * 【应用场景】：
 * - 🚀 实例化渲染：大量相同几何的高效处理
 * - 📊 批量测试：预先变换射线，然后并行测试多个粒子
 * - 🎮 实时渲染：性能要求极高的交互式应用
 * - 🔄 LOD系统：远距离粒子的快速裁剪
 *
 * 【性能特点】：
 * - ✅ 无矩阵运算：跳过坐标变换
 * - ✅ 纯向量操作：GPU向量单元友好
 * - ✅ 分支最少：减少GPU线程分化
 * - ✅ 内存高效：复用预计算结果
 *
 * @param canonicalRayOrigin              标准化空间中的射线起点
 * @param canonicalUnormalizedRayDirection 标准化空间中的射线方向 (未归一化)
 * @param minHitDistance                  最小有效击中距离
 * @param maxHitDistance                  最大有效击中距离
 * @param maxParticleSquaredDistance      最大粒子平方距离阈值
 * @param hitDistance                     输出：击中距离参数t值
 * @return 是否通过所有距离和几何测试
 */
[CudaDeviceExport]
bool particleDensityHitInstance(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    // ========== 超高速实例化检测：纯数学几何计算 ==========
    //
    // 【设计哲学】：
    // 这是最精简的碰撞检测实现，专为海量并行处理优化。
    // 所有复杂的坐标变换都已预先完成，这里只做纯数学计算。
    //
    
    // 步骤1: 🧮 射线到原点最近点的参数t计算
    // 数学推导：最小化 |canonicalRayOrigin + t * canonicalUnormalizedRayDirection|²
    // 对t求导并令其为0: d/dt[...] = 0 
    // 解得: t = -(origin · direction) / |direction|²
    //
    const float numerator = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
    //    ^^^^^^^^^^^^^
    //    分子: -(射线起点 · 射线方向) 
    //    物理意义：起点向量在方向向量上的投影(取负号)
    
    const float denominator = rcp(dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection));
    //    ^^^^^^^^^^^^^^^^^^^^^
    //    分母倒数: 1 / |射线方向|²
    //    rcp()是GPU优化的倒数函数，比 (1.0/x) 更快
    
    hitDistance = numerator * denominator;
    //    ^^^^^^^^^^^^
    //    最终结果：射线参数t，表示最近点位置
    //    几何意义：origin + t * direction = 离原点最近的射线上的点

    // 步骤2: ✅ 三重安全验证
    // 确保计算出的击中距离满足所有约束条件
    return (hitDistance > minHitDistance) &&              // 验证1: 近裁剪面检查
           (hitDistance < maxHitDistance) &&              // 验证2: 远裁剪面检查
           (gaussianParticle.canonicalRayMinSquaredDistance(    // 验证3: 几何相交检查
                canonicalRayOrigin,                       //   使用标准化射线起点
                normalize(canonicalUnormalizedRayDirection)    //   归一化射线方向
           ) < maxParticleSquaredDistance);
    //
    // 【性能要点】：
    // 1. 🚀 无矩阵运算：避免了昂贵的坐标变换
    // 2. 🎯 向量计算：充分利用GPU SIMD指令
    // 3. 📐 数学优化：使用rcp()等GPU特化函数
    // 4. 🔧 最小分支：减少GPU线程发散
    //
    // 【使用场景】：
    // - 实例化渲染：相同几何的大规模复制
    // - LOD优化：远距离粒子的快速剔除
    // - 预筛选：在完整渲染前的粗筛选阶段
}

/**
 * 计算粒子的入射光线方向
 * 
 * 根据光源位置和粒子位置，计算入射光线的方向。
 * 用于光照计算和阴影效果。
 * 
 * @param parameters 粒子参数（包含位置信息）
 * @param sourcePosition 光源位置（世界空间）
 * @return 从光源到粒子的单位方向向量
 */
[CudaDeviceExport] float3 particleDensityIncidentDirection(
    in gaussianParticle.Parameters parameters,
    in float3 sourcePosition
)
{
    return gaussianParticle.incidentDirectionFromParameters(parameters, sourcePosition);
}

/**
 * 将入射方向梯度反向传播到缓冲区
 * 
 * 对入射方向计算进行反向微分，将梯度传播到粒子位置参数。
 * 用于训练阶段的参数优化。
 * 
 * @param particleIdx 粒子索引
 * @param commonParameters 通用参数
 * @param sourcePosition 光源位置
 */
[CudaDeviceExport] void particleDensityIncidentDirectionBwdToBuffer(
    in uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float3 sourcePosition
)
{
    bwd_diff(gaussianParticle.incidentDirectionFromBuffer)(
        particleIdx, 
        commonParameters.parametersBuffer, 
        sourcePosition
    );
}
