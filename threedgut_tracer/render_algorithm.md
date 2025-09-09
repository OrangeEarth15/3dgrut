# 3DGUT K-Buffer渲染算法伪代码

## 主渲染流程

```python
def render_3dgut():
    """主渲染函数"""
    # 1. GPU核函数启动
    launch_cuda_kernel(
        grid=(tile_width, tile_height, 1),
        block=(block_x, block_y, 1),
        params=render_params,
        tile_ranges=sorted_tile_ranges,
        particle_indices=sorted_particles,
        ray_data=(origins, directions),
        outputs=(hit_counts, distances, colors)
    )


def render_kernel_per_pixel():
    """每个GPU线程执行的渲染逻辑"""
    # 1. 初始化光线
    ray = initialize_ray()
    
    # 2. K-Buffer渲染
    eval_k_buffer(ray)
    
    # 3. 输出结果
    finalize_ray(ray)
```

## 光线初始化

```python
def initialize_ray():
    """初始化单条光线"""
    # 计算像素坐标
    pixel_x = thread_x + block_x * block_width
    pixel_y = thread_y + block_y * block_height
    
    ray = Ray()
    ray.pixel_id = pixel_x + screen_width * pixel_y
    ray.origin = camera_position
    ray.direction = compute_ray_direction(pixel_x, pixel_y)
    
    # 初始化渲染状态
    ray.color = (0, 0, 0)          # RGB累积颜色
    ray.transmittance = 1.0        # 透射率
    ray.depth = 0.0                # 累积深度
    ray.hit_count = 0              # 击中次数
    
    # 计算有效渲染范围
    ray.t_near, ray.t_far = intersect_scene_bounds(ray)
    ray.active = (ray.t_far > ray.t_near)
    
    return ray
```

## K-Buffer核心算法

```python
def eval_k_buffer(ray):
    """K-Buffer渲染核心逻辑"""
    # 获取当前tile的粒子范围
    tile_id = compute_tile_id()
    particle_start, particle_end = tile_particle_ranges[tile_id]
    
    # 初始化K-Buffer
    k_buffer = KBuffer(capacity=K)
    
    # 分批处理粒子
    for batch_start in range(particle_start, particle_end, BATCH_SIZE):
        # 早停检查
        if ray.transmittance < EARLY_STOP_THRESHOLD:
            break
        
        # 协作加载粒子数据到共享内存
        batch_end = min(batch_start + BATCH_SIZE, particle_end)
        load_particles_to_shared_memory(batch_start, batch_end)
        sync_threads()
        
        # 处理批次中的每个粒子
        for i in range(batch_end - batch_start):
            if not ray.active:
                break
                
            particle = shared_memory[i]
            if particle.id == INVALID_ID:
                break
            
            # 测试光线与粒子相交
            hit = test_intersection(ray, particle)
            if hit.valid:
                # K-Buffer管理
                manage_k_buffer(ray, k_buffer, hit)
    
    # 处理K-Buffer中剩余的击中
    process_remaining_hits(ray, k_buffer)


def manage_k_buffer(ray, k_buffer, new_hit):
    """K-Buffer击中管理"""
    if k_buffer.full():
        # 缓冲区已满，处理最近的击中腾出空间
        closest_hit = k_buffer.pop_closest()
        process_hit(ray, closest_hit)
    
    # 插入新击中，保持距离排序
    k_buffer.insert_sorted(new_hit)


def process_remaining_hits(ray, k_buffer):
    """处理剩余击中"""
    while not k_buffer.empty() and ray.active:
        hit = k_buffer.pop_closest()
        process_hit(ray, hit)
```

## 粒子相交检测

```python
def test_intersection(ray, particle):
    """光线与高斯粒子相交测试"""
    # 坐标变换：世界空间到粒子局部空间
    local_origin, local_direction = transform_to_particle_space(
        ray.origin, ray.direction, particle
    )
    
    # 计算高斯核函数响应
    distance_squared = compute_min_distance_squared(local_origin, local_direction)
    gaussian_response = exp(-0.5 * distance_squared)
    
    # 计算不透明度
    alpha = min(MAX_ALPHA, gaussian_response * particle.density)
    
    # 检查是否足够显著
    if alpha < MIN_ALPHA:
        return Hit(valid=False)
    
    # 计算精确击中距离
    hit_distance = compute_hit_distance(local_origin, local_direction, particle.scale)
    
    # 验证距离范围
    if not (ray.t_near < hit_distance < ray.t_far):
        return Hit(valid=False)
    
    return Hit(
        valid=True,
        particle_id=particle.id,
        distance=hit_distance,
        alpha=alpha
    )


def transform_to_particle_space(ray_origin, ray_direction, particle):
    """射线到粒子局部空间的变换"""
    # 平移
    offset = ray_origin - particle.position
    
    # 旋转
    rotated_offset = particle.rotation_transpose @ offset
    rotated_direction = particle.rotation_transpose @ ray_direction
    
    # 缩放
    scale_inv = 1.0 / particle.scale
    local_origin = scale_inv * rotated_offset
    local_direction = normalize(scale_inv * rotated_direction)
    
    return local_origin, local_direction


def compute_hit_distance(local_origin, local_direction, scale):
    """计算光线参数化击中距离"""
    # 投影参数
    projection = dot(local_direction, -local_origin)
    
    # 加权距离向量
    weighted_vector = scale * local_direction * projection
    
    # 返回距离
    return length(weighted_vector)
```

## 击中处理

```python
def process_hit(ray, hit):
    """处理单个粒子击中"""
    # 计算混合权重
    weight = compute_blend_weight(hit.alpha, ray.transmittance, hit.distance, ray.depth)
    
    # 获取粒子颜色
    if USE_SPHERICAL_HARMONICS:
        color = compute_sh_color(hit.particle_id, ray.direction)
    else:
        color = precomputed_colors[hit.particle_id]
    
    # 颜色混合
    blend_color(ray.color, color, weight)
    
    # 统计
    if weight > 0:
        ray.hit_count += 1
    
    # 早停检查
    if ray.transmittance < MIN_TRANSMITTANCE:
        ray.active = False


def compute_blend_weight(alpha, transmittance, depth, integrated_depth):
    """计算Alpha混合权重"""
    # 权重计算
    weight = alpha * transmittance
    
    # 深度积分
    integrated_depth += depth * weight
    
    # 透射率更新
    transmittance *= (1.0 - alpha)
    
    return weight


def blend_color(ray_color, particle_color, weight):
    """颜色混合"""
    if weight > 0:
        ray_color += particle_color * weight
```

## 数据结构

```python
class Ray:
    pixel_id: int
    origin: Vec3
    direction: Vec3
    t_near: float
    t_far: float
    color: Vec3
    transmittance: float
    depth: float
    hit_count: int
    active: bool


class Hit:
    valid: bool
    particle_id: int
    distance: float
    alpha: float


class KBuffer:
    hits: List[Hit]
    capacity: int
    count: int
    
    def full(self) -> bool:
        return count >= capacity
    
    def empty(self) -> bool:
        return count == 0
    
    def insert_sorted(self, hit: Hit):
        if count < capacity:
            hits.append(hit)
            hits.sort(key=lambda h: h.distance)
            count += 1
        elif hit.distance < hits[-1].distance:
            hits[-1] = hit
            hits.sort(key=lambda h: h.distance)
    
    def pop_closest(self) -> Hit:
        closest = hits[0]
        hits = hits[1:]
        count -= 1
        return closest


class Particle:
    id: int
    position: Vec3
    rotation_transpose: Mat3x3
    scale: Vec3
    density: float
```

## 最终输出

```python
def finalize_ray(ray):
    """输出最终渲染结果"""
    if not ray.active:
        return
    
    # 输出颜色 (RGB + Alpha)
    final_alpha = 1.0 - ray.transmittance
    output_color[ray.pixel_id] = (ray.color.r, ray.color.g, ray.color.b, final_alpha)
    
    # 输出深度
    output_depth[ray.pixel_id] = ray.depth
    
    # 输出命中次数
    if ENABLE_HIT_COUNT:
        output_hit_count[ray.pixel_id] = ray.hit_count
```