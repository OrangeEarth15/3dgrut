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
struct RawParameters : IDifferentiable {
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
 * 
 * 包含渲染和计算所需的全局配置参数。
 * 提供统一的接口供外部调用。
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
[BackwardDifferentiable][ForceInline] RawParameters fetchParametersFromBuffer(
    no_diff uint32_t particleIdx,              // 粒子索引（不参与微分）
    no_diff RawParametersBuffer parametersBuffer) {  // 参数缓冲区（不参与微分）
    
    // 从全局缓冲区读取指定粒子的参数
    RawParameters rawParameters = parametersBuffer._dataPtr[particleIdx];
    
    // Surfel模式：将三维高斯椒球压缩为二维面片
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
 * 这样可以用统一的算法处理不同形状和大小的高斯椒球。
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
    
    // 1. 计算缩放因子的倒数（用于将椒球标准化为单位球）
    const float3 giscl  = float3(1.0f) / parameters.scale;
    
    // 2. 平移：射线起点相对于粒子中心的位置
    const float3 gposc  = (rayOrigin - parameters.position);
    
    // 3. 旋转：应用旋转矩阵转置将射线转换到粒子的局部坐标系
    const float3 gposcr = mul(parameters.rotationT, gposc);
    
    // 4. 缩放：按比例缩放，将椒球标准化为单位球
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
 * 这个距离用于判断射线是否与高斯椒球相交。
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

/**
 * 积分单次相交结果（体积渲染混合）
 * 
 * 实现体积渲染中单个粒子的贡献积分。支持两种混合模式：
 * 1. 前向后模式：用于正向穿越，直接累加权重值
 * 2. 后向前模式：用于Over混合，使用线性插值
 * 
 * 这个函数处理：
 * - 深度积分（用于Z-buffer或距离估计）
 * - 法线积分（用于光照计算）
 * - 透射率更新（控制后续粒子的影响）
 * 
 * @param alpha 当前粒子的不透明度
 * @param transmittance 输入输出：累积透射率
 * @param depth 当前粒子的深度
 * @param integratedDepth 输入输出：累积深度
 * @param enableNormal 是否需要处理法线
 * @param normal 当前粒子的法线
 * @param integratedNormal 输入输出：累积法线
 * @return 当前粒子的混合权重
 */
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
   // 计算混合权重：根据混合模式选择不同的权重计算方式
   const float weight = backToFront ? alpha : alpha * transmittance;
   
   if (backToFront)
   {
       // 后向前模式：Over混合，使用线性插值
       integratedDepth = lerp(integratedDepth, depth, alpha);
       if (enableNormal)
       {
           integratedNormal = lerp(integratedNormal, normal, alpha);
       }
   }
   else 
   {
       // 前向后模式：正向穿越，直接累加权重值
       integratedDepth += depth * weight;
       if (enableNormal) 
       {
            integratedNormal += normal * weight;
       }
   }

   // 更新透射率：每个粒子都会减少后续光线的透射率
   transmittance *= (1 - alpha);
   
   return weight;  // 返回当前粒子的实际混合权重
}

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
    float alpha = 0.0f;
    float depth;
    float3 normal;
    if (hit(rayOrigin,
            rayDirection,
            fetchParameters(particleIdx, parametersBuffer),
            alpha,
            depth,
            enableNormal,
            normal))
    {
        return integrateHit<backToFront>(alpha, 
                                         transmittance, 
                                         depth, 
                                         integratedDepth, 
                                         enableNormal, 
                                         normal, 
                                         integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable][ForceInline] float3x3 computeCovariance(
    in Parameters parameters) {
    // Σ = RSS^TR^T
    const float3x3 StRt = float3x3(parameters.scale.x * parameters.rotationT[0],
                                  parameters.scale.y * parameters.rotationT[1],
                                  parameters.scale.z * parameters.rotationT[2]);
    return mul(transpose(StRt), StRt);
}

[BackwardDifferentiable] [ForceInline]
float3 incidentDirectionFromParameters(
    Parameters parameters,
    no_diff float3 sourcePosition
)
{
    return normalize(parameters.position - sourcePosition);
}

[BackwardDifferentiable][ForceInline]
no_diff float3 incidentDirectionFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float3 sourcePosition
)
{
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
    // ========== 直接转发到核心实现 ==========
    // 注意：这里的depth参数将返回精确的hitT值，用于K-Buffer排序
    // 这个hitT与全局排序使用的globalDepth不同：
    // - globalDepth：粒子中心到相机的距离（用于粗排序）
    // - hitT (depth)：光线实际击中粒子表面的参数化距离（用于精排序）
    return gaussianParticle.hit(rayOrigin,      // 转发：光线起点
                                rayDirection,    // 转发：光线方向
                                parameters,      // 转发：粒子参数
                                alpha,          // 转发：不透明度输出
                                depth,          // 转发：hitT输出（核心！）
                                enableNormal,   // 转发：法线计算标志
                                normal);        // 转发：法线输出
}

/**
 * 粒子密度相交积分（前向模式）
 * 
 * 对单个粒子的相交结果进行前向积分，用于体积渲染。
 * 使用前向后遍历模式，适用于正向穿越的渲染算法。
 * 
 * @param alpha 粒子的不透明度
 * @param transmittance 输入输出：累积透射率
 * @param depth 粒子深度
 * @param integratedDepth 输入输出：累积深度
 * @param enableNormal 是否处理法线
 * @param normal 粒子法线
 * @param integratedNormal 输入输出：累积法线
 * @return 粒子的混合权重
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
    return gaussianParticle.integrateHit<false>(
        alpha,
        transmittance,
        depth,
        integratedDepth,
        enableNormal,
        normal,
        integratedNormal);
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

[CudaDeviceExport]
void particleDensityProcessHitBwdToBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
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
        const float weight = 1.0f / (1.0f - alpha);
        
        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);
        
        integratedDepth = (integratedDepth - depth * alpha) * weight;
        DifferentialPair<float> integratedDepthDiff = DifferentialPair<float>(integratedDepth, integratedDepthGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        bwd_diff(gaussianParticle.processHitFromBuffer<true>)(
            rayOrigin,
            rayDirection,
            particleIdx,
            commonParameters.parametersBuffer,
            transmittanceDiff,
            integratedDepthDiff,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedDepthGrad = integratedDepthDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }
    }
}

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
    gaussianParticle.Parameters parameters = gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer);

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    gaussianParticle.cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse<gaussianParticle.KernelDegree>(
        canonicalRayOrigin,
        canonicalRayDirection);

    // distance to the gaussian center projection on the ray
    hitDistance = gaussianParticle.canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);

    return (hitDistance > minHitDistance) &&
           (hitDistance < maxHitDistance) &&
           (gaussianParticle.canonicalRayMinSquaredDistance(
                canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
}

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
    const float numerator = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
    const float denominator = rcp(dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection));
    hitDistance = numerator * denominator;
    return (hitDistance > minHitDistance) &&
           (hitDistance < maxHitDistance) &&
           (gaussianParticle.canonicalRayMinSquaredDistance(
                canonicalRayOrigin,
                normalize(canonicalUnormalizedRayDirection)) < maxParticleSquaredDistance);
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
