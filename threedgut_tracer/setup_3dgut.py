# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, math
import torch

from threedgrut.utils import jit


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf):
    # 获取 / 创建当前项目的编译输出目录，如 build/lib3dgut_cc，后续所有编译结果都写到这里（临时/缓存文件/头文件等）
    build_dir = torch.utils.cpp_extension._get_build_directory("lib3dgut_cc", verbose=True)

    include_paths = []
    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies", "fmt", "include"))
    include_paths.append(build_dir) # build_dir也加入，确保自动生成的头文件可被找到

    # Compiler options.

    def to_cpp_bool(value):
        return "true" if value else "false"

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha*ut_alpha*(ut_d+ut_kappa))

    defines = [
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={to_cpp_bool(conf.render.enable_hitcounts)}",
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        # -- Performance Optimizations --
        f"-DDYNAMIC_LOAD_BALANCING={to_cpp_bool(conf.render.splat.dynamic_load_balancing)}",
        f"-DFINE_GRAINED_LOAD_BALANCING={to_cpp_bool(getattr(conf.render.splat, 'fine_grained_load_balancing', False))}",
        # -- Unscented Transform --
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={conf.render.splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={to_cpp_bool(conf.render.splat.ut_require_all_sigma_points_valid)}",
        # -- Culling --
        f"-DGAUSSIAN_RECT_BOUNDING={to_cpp_bool(conf.render.splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={to_cpp_bool(conf.render.splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={to_cpp_bool(conf.render.splat.tile_based_culling)}",
    ]

    cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    cuda_cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        # 禁用CUDA的某些功能，以确保兼容性
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # 开启CUDA高性能选项
        "-use_fast_math", "-O3",
        # 添加调试信息支持NCU profiling
        "-lineinfo", "-g",
        *defines,
    ]

    # List of sources.
    source_files = [
        "src/splatRaster.cpp",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels
    # TODO: do not overwrite files, use config hash to register the needed version
    import importlib, subprocess

    slang_mod = importlib.import_module("slangtorch") # 动态导入指定名称的模块，比import slangtorch更安全
    slang_dir = os.path.dirname(slang_mod.__file__) # 获取模块安装目录

    slang_build_env = os.environ
    slang_build_env["PATH"] += ";" if os.name == "nt" else ":"
    slang_build_env["PATH"] += os.path.join(slang_dir, "bin")
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include", "3dgut")

    subprocess.check_call( # 执行外部命令并等待完成
        [
            "slangc", "-target", "cuda",
            "-I", os.path.join(os.path.dirname(__file__), "include"),
            "-I", os.path.join(os.path.dirname(__file__), "..", "threedgrt_tracer", "include"),
            "-line-directive-mode", "none",
            "-matrix-layout-row-major", # NB : this is required for cuda target
            "-Wno-41018",
            "-O2",
            *defines,
            f"{os.path.join(slang_build_inc_dir,'threedgut.slang')}",
            "-o", f"{os.path.join(build_dir,'threedgutSlang.cuh')}",
        ],
        env=slang_build_env,
    )

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    jit.load(
        name="lib3dgut_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )
