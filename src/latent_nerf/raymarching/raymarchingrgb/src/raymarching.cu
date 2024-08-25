#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}


////////////////////////////////////////////////////
/////////////           utils          /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}


// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const float radius,
    const uint32_t N,
    scalar_t * coords
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    coords += n * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // solve t from || o + td || = radius
    const float A = dx * dx + dy * dy + dz * dz;
    const float B = ox * dx + oy * dy + oz * dz; // in fact B / 2
    const float C = ox * ox + oy * oy + oz * oz - radius * radius;

    const float t = (- B + sqrtf(B * B - A * C)) / A; // always use the larger solution (positive)

    // solve theta, phi (assume y is the up axis)
    const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
    const float theta = atan2(sqrtf(x * x + z * z), y); // [0, PI)
    const float phi = atan2(z, x); // [-PI, PI)

    // normalize to [-1, 1]
    coords[0] = 2 * theta * RPI() - 1;
    coords[1] = phi * RPI();
}


void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "sph_from_ray", ([&] {
        kernel_sph_from_ray<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
    }));
}


// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(
    const int * __restrict__ coords,
    const uint32_t N,
    int * indices
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;
    indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D<<<div_round_up(N, N_THREAD), N_THREAD>>>(coords.data_ptr<int>(), N, indices.data_ptr<int>());
}


// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(
    const int * __restrict__ indices,
    const uint32_t N,
    int * coords
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;

    const int ind = indices[n];

    coords[0] = __morton3D_invert(ind >> 0);
    coords[1] = __morton3D_invert(ind >> 1);
    coords[2] = __morton3D_invert(ind >> 2);
}


void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D_invert<<<div_round_up(N, N_THREAD), N_THREAD>>>(indices.data_ptr<int>(), N, coords.data_ptr<int>());
}


// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t * __restrict__ grid,
    const uint32_t N,
    const float density_thresh,
    uint8_t * bitfield
) {
    // parallel per byte
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    grid += n * 8;

    uint8_t bits = 0;

    #pragma unroll
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (grid[i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    }

    bitfield[n] = bits;
}


void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grid.scalar_type(), "packbits", ([&] {
        kernel_packbits<<<div_round_up(N, N_THREAD), N_THREAD>>>(grid.data_ptr<scalar_t>(), N, density_thresh, bitfield.data_ptr<uint8_t>());
    }));
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const uint8_t * __restrict__ grid,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays,
    int * counter,
    const scalar_t* __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;
    
    float t0 = near;
    
    // perturb
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);
    
    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (occ) {
            num_steps++;
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d, near=%f, far=%f, dt=%f, max_steps=%f\n", n, num_steps, near, far, dt_min, (far - near) / dt_min);

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    
    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;

    t = t0;
    uint32_t step = 0;

    float last_t = t;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max); 
            } while (t < tt);
        }
    }
}

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, dt_gamma, max_steps, N, C, H, M, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;

    // accumulate
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        t += deltas[1]; // real delta
        d += weight * t;

        ws += weight;

        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}


void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), M, N, T_thresh, weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) return;

    grad_weights_sum += index;
    grad_image += index * 3;
    weights_sum += index;
    image += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;

    // accumulate
    uint32_t step = 0;

    scalar_t T = 1.0f;
    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], ws_final = weights_sum[0];
    scalar_t r = 0, g = 0, b = 0, ws = 0;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        ws += weight;

        T *= 1.0f - alpha;

        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs[0] = grad_image[0] * weight;
        grad_rgbs[1] = grad_image[1] * weight;
        grad_rgbs[2] = grad_image[2] * weight;

        // write grad_sigmas
        grad_sigmas[0] = deltas[0] * (
            grad_image[0] * (T * rgbs[0] - (r_final - r)) +
            grad_image[1] * (T * rgbs[1] - (g_final - g)) +
            grad_image[2] * (T * rgbs[2] - (b_final - b)) +
            grad_weights_sum[0] * (1 - ws_final)
        );

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        grad_sigmas++;
        grad_rgbs += 3;

        step++;
    }
}


void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(), grad_image.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), M, N, T_thresh, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive,
    const uint32_t n_step,
    const int* __restrict__ rays_alive,
    const scalar_t* __restrict__ rays_t,
    const scalar_t* __restrict__ rays_o,
    const scalar_t* __restrict__ rays_d,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const scalar_t* __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];

    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    float t = rays_t[index]; // current ray's t
    const float near = nears[index], far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), bound);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor near, const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises) {
    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), bound, dt_gamma, max_steps, C, H, grid.data_ptr<uint8_t>(), near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), noises.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive,
    const uint32_t n_step,
    const float T_thresh,
    int* rays_alive,
    scalar_t* rays_t,
    const scalar_t* __restrict__ sigmas,
    const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ deltas,
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id

    // locate
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;

    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;

    scalar_t t = rays_t[0]; // current ray's t

    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    // accumulate
    uint32_t step = 0;
    while (step < n_step) {

        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /*
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        -->
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}



// ------------------------------


template <typename scalar_t>
__global__ void kernel_composite_rays_train_localization_int_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ loc_prob,
    const scalar_t * __restrict__ style_rgbs,
    const scalar_t * __restrict__ beck_rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * depth,
    scalar_t * weights_sum,
    scalar_t * image_origin,
    scalar_t * image_prob,
    scalar_t * image_style_origin,
    scalar_t * image_back_origin,
    scalar_t * image_localization_mix,
    scalar_t * image_style_mix,
    scalar_t * image_back_mix
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image_origin[index * 3] = 0;
        image_origin[index * 3 + 1] = 0;
        image_origin[index * 3 + 2] = 0;
        
        image_prob[index] = 0;

        image_style_origin[index * 3] = 0;
        image_style_origin[index * 3 + 1] = 0;
        image_style_origin[index * 3 + 2] = 0;

        image_back_origin[index * 3] = 0;
        image_back_origin[index * 3 + 1] = 0;
        image_back_origin[index * 3 + 2] = 0;

        image_localization_mix[index * 3] = 0;
        image_localization_mix[index * 3 + 1] = 0;
        image_localization_mix[index * 3 + 2] = 0;

        image_style_mix[index * 3] = 0;
        image_style_mix[index * 3 + 1] = 0;
        image_style_mix[index * 3 + 2] = 0;

        image_back_mix[index * 3] = 0;
        image_back_mix[index * 3 + 1] = 0;
        image_back_mix[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;

    loc_prob += offset;
    style_rgbs += offset * 3;
    beck_rgbs += offset * 3;

    // accumulate
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;
    scalar_t  localization_prob = 0;
    scalar_t r_stylization = 0, g_stylization = 0, b_stylization = 0;
    scalar_t r_background = 0, g_background = 0, b_background = 0;
    scalar_t r_yellow = 0.8, g_yellow = 1.0, b_yellow = 0.0;
    scalar_t r_gray = 0.71, g_gray = 0.71, b_gray = 0.71;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        localization_prob += weight * loc_prob[0];

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        r_stylization += weight * style_rgbs[0];
        g_stylization += weight * style_rgbs[1];
        b_stylization += weight * style_rgbs[2];

        r_background += weight * beck_rgbs[0];
        g_background += weight * beck_rgbs[1];
        b_background += weight * beck_rgbs[2];

        t += deltas[1]; // real delta
        d += weight * t;

        ws += weight;

        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;

        loc_prob++;
        style_rgbs += 3;
        beck_rgbs += 3;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image_origin[index * 3] = r;
    image_origin[index * 3 + 1] = g;
    image_origin[index * 3 + 2] = b;

    image_prob[index] = localization_prob;

    image_style_origin[index * 3 ] = r_stylization;
    image_style_origin[index * 3  + 1] = g_stylization;
    image_style_origin[index * 3  + 2] = b_stylization;

    image_back_origin[index * 3 ] = r_background;
    image_back_origin[index * 3  + 1] = g_background;
    image_back_origin[index * 3  + 2] = b_background;

    image_localization_mix[index * 3 ] = localization_prob * r_yellow + (1 - localization_prob) * r_gray;
    image_localization_mix[index * 3  + 1] = localization_prob * g_yellow + (1 - localization_prob) * g_gray;
    image_localization_mix[index * 3  + 2] = localization_prob * b_yellow + (1 - localization_prob) * b_gray;
    
    image_style_mix[index * 3 ] = localization_prob * r_stylization + (1 - localization_prob) * r_gray;
    image_style_mix[index * 3  + 1] = localization_prob * g_stylization + (1 - localization_prob) * g_gray;
    image_style_mix[index * 3  + 2] = localization_prob * b_stylization + (1 - localization_prob) * b_gray;

    image_back_mix[index * 3 ] = localization_prob * r_yellow + (1 - localization_prob) * r_background;
    image_back_mix[index * 3  + 1] = localization_prob * g_yellow + (1 - localization_prob) * g_background;
    image_back_mix[index * 3  + 2] = localization_prob * b_yellow + (1 - localization_prob) * b_background;
}

void composite_rays_train_localization_int_forward(const at::Tensor sigmas, const at::Tensor rgbs, 
                                  const at::Tensor loc_prob, const at::Tensor style_rgbs, const at::Tensor beck_rgbs,
                                  const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N,
                                  const float T_thresh, 
                                  at::Tensor weights_sum, at::Tensor depth, at::Tensor image_origin,
                                  at::Tensor image_prob, at::Tensor image_style_origin, at::Tensor image_back_origin,
                                  at::Tensor image_localization_mix, at::Tensor image_style_mix,
                                  at::Tensor image_back_mix) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_localization_int_forward", ([&] {
        kernel_composite_rays_train_localization_int_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(),
                                                                                     rgbs.data_ptr<scalar_t>(),
                                                                                     loc_prob.data_ptr<scalar_t>(),
                                                                                     style_rgbs.data_ptr<scalar_t>(),
                                                                                     beck_rgbs.data_ptr<scalar_t>(),
                                                                                     deltas.data_ptr<scalar_t>(),
                                                                                     rays.data_ptr<int>(),
                                                                                     M, N, T_thresh,
                                                                                     weights_sum.data_ptr<scalar_t>(),
                                                                                     depth.data_ptr<scalar_t>(),
                                                                                     image_origin.data_ptr<scalar_t>(),
                                                                                     image_prob.data_ptr<scalar_t>(),
                                                                                     image_style_origin.data_ptr<scalar_t>(),
                                                                                     image_back_origin.data_ptr<scalar_t>(),
                                                                                     image_localization_mix.data_ptr<scalar_t>(),
                                                                                     image_style_mix.data_ptr<scalar_t>(),
                                                                                     image_back_mix.data_ptr<scalar_t>());

        
    }));

}  


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_localization_int_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ grad_image_style,
    const scalar_t * __restrict__ grad_image_back,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ loc_prob,
    const scalar_t * __restrict__ style_rgbs,
    const scalar_t * __restrict__ beck_rgbs,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image_origin,
    const scalar_t * __restrict__ image_prob,
    const scalar_t * __restrict__ image_style_origin,
    const scalar_t * __restrict__ image_back_origin,
    const scalar_t * __restrict__ image_localization_mix,
    const scalar_t * __restrict__ image_style_mix,
    const scalar_t * __restrict__ image_back_mix,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_loc_prob,
    scalar_t * grad_rgbs_style,
    scalar_t * grad_rgbs_back
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) return;

    grad_weights_sum += index;
    grad_image += index * 3;
    weights_sum += index;

    image_origin += index * 3;

    image_prob += index;

    image_style_origin += index * 3;

    image_back_origin += index * 3;

    image_localization_mix += index * 3;

    image_style_mix += index * 3;

    image_back_mix += index * 3;

    sigmas += offset;
    rgbs += offset * 3;
    loc_prob += offset;
    style_rgbs += offset * 3;
    beck_rgbs += offset * 3;
    deltas += offset * 2;
    grad_loc_prob += offset;
    grad_rgbs_style += offset * 3;
    grad_rgbs_back += offset * 3;

    // accumulate
    uint32_t step = 0;

    scalar_t T = 1.0f;
    // const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], ws_final = weights_sum[0];
    // scalar_t r = 0, g = 0, b = 0, ws = 0;
    scalar_t r_yellow = 0.8, g_yellow = 1.0, b_yellow = 0.0;
    scalar_t r_gray = 0.71, g_gray = 0.71, b_gray = 0.71;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        T *= 1.0f - alpha;

        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs_style[0] = grad_image_style[0] * weight * image_prob[0];
        grad_rgbs_style[1] = grad_image_style[1] * weight * image_prob[0];
        grad_rgbs_style[2] = grad_image_style[2] * weight * image_prob[0];

        grad_rgbs_back[0] = grad_image_back[0] * weight * ( 1- image_prob[0]);
        grad_rgbs_back[1] = grad_image_back[1] * weight * ( 1- image_prob[0]);
        grad_rgbs_back[2] = grad_image_back[2] * weight * ( 1- image_prob[0]);

        // write grad_sigmas

        grad_loc_prob[0] = grad_image[0] * weight * (r_yellow - r_gray) +
                           grad_image[1] * weight * (g_yellow - g_gray) +
                           grad_image[2] * weight * (b_yellow - b_gray) +
                        //    grad_image[3] * (T * rgbs[3] - (newvar_final - newvar)) TODO - add newvar gradient
                           grad_image_style[0] * weight * (image_style_origin[0] - r_gray) +
                           grad_image_style[1] * weight * (image_style_origin[1] - g_gray) +
                           grad_image_style[2] * weight * (image_style_origin[2] - b_gray) +
                           // grad_image_style[3] * weight * (style_rgbs[3] - rgbs[3]) + TODO - add newvar gradient
                        //    grad_image_back[0] * weight * (rgbs[0] - beck_rgbs[0]) +
                        //    grad_image_back[1] * weight * (rgbs[1] - beck_rgbs[1]) +
                        //    grad_image_back[2] * weight * (rgbs[2] - beck_rgbs[2])+
                        //    grad_image_back[3] * weight * (rgbs[3] - beck_rgbs[3]);
                            grad_image_back[0] * weight * (r_yellow - image_back_origin[0]) +
                            grad_image_back[1] * weight * (g_yellow - image_back_origin[1]) +
                            grad_image_back[2] * weight * (b_yellow - image_back_origin[2]);
                           // grad_image_back[3] * weight * (rgbs[3] - beck_rgbs[3]); TODO - add newvar gradient

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;

        // locate
        sigmas++;
        deltas += 2;
        grad_loc_prob++;
        grad_rgbs_style += 3;
        grad_rgbs_back += 3;

        step++;
    }
}



void composite_rays_train_localization_int_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image,
                                const at::Tensor grad_image_style, const at::Tensor grad_image_back,
                                const at::Tensor sigmas, const at::Tensor rgbs, 
                                const at::Tensor loc_prob, const at::Tensor style_rgbs, const at::Tensor beck_rgbs, 
                                const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, 
                                const at::Tensor image_origin, const at::Tensor image_prob,
                                const at::Tensor image_style_origin,
                                const at::Tensor image_back_origin, const at::Tensor image_localization_mix,
                                const at::Tensor image_style_mix,
                                const at::Tensor image_back_mix,
                                const uint32_t M, const uint32_t N, const float T_thresh,
                                at::Tensor grad_loc_prob, at::Tensor grad_rgbs_style, at::Tensor grad_rgbs_back
                                ) {





    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_localization_int_backward", ([&] {
        kernel_composite_rays_train_localization_int_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(),
                                                                                    grad_image.data_ptr<scalar_t>(),
                                                                                    grad_image_style.data_ptr<scalar_t>(),
                                                                                    grad_image_back.data_ptr<scalar_t>(),
                                                                                    sigmas.data_ptr<scalar_t>(), 
                                                                                    rgbs.data_ptr<scalar_t>(),
                                                                                    loc_prob.data_ptr<scalar_t>(),
                                                                                    style_rgbs.data_ptr<scalar_t>(),
                                                                                    beck_rgbs.data_ptr<scalar_t>(),
                                                                                    deltas.data_ptr<scalar_t>(), 
                                                                                    rays.data_ptr<int>(), 
                                                                                    weights_sum.data_ptr<scalar_t>(),
                                                                                    image_origin.data_ptr<scalar_t>(),
                                                                                    image_prob.data_ptr<scalar_t>(),
                                                                                    image_style_origin.data_ptr<scalar_t>(),
                                                                                    image_back_origin.data_ptr<scalar_t>(),
                                                                                    image_localization_mix.data_ptr<scalar_t>(),
                                                                                    image_style_mix.data_ptr<scalar_t>(),
                                                                                    image_back_mix.data_ptr<scalar_t>(),
                                                                                    M, N, T_thresh,    
                                                                                    grad_loc_prob.data_ptr<scalar_t>(),
                                                                                    grad_rgbs_style.data_ptr<scalar_t>(),
                                                                                    grad_rgbs_back.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays_localization_int(
    const uint32_t n_alive,
    const uint32_t n_step,
    const float T_thresh,
    int* rays_alive,
    scalar_t* rays_t,
    const scalar_t* __restrict__ sigmas,
    const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ loc_prob,
    const scalar_t* __restrict__ style_rgbs,
    const scalar_t* __restrict__ beck_rgbs,
    const scalar_t* __restrict__ deltas,
    scalar_t* weights_sum, scalar_t* depth,
    scalar_t* image_origin,
    scalar_t* image_prob,
    scalar_t* image_style_origin,
    scalar_t* image_back_origin,
    scalar_t* image_localization_mix,
    scalar_t* image_style_mix,
    scalar_t* image_back_mix
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id

    // locate
    sigmas += n * n_step;
    loc_prob += n * n_step;
    rgbs += n * n_step * 3;
    style_rgbs += n * n_step * 3;
    beck_rgbs += n * n_step * 3;
    deltas += n * n_step * 2;

    rays_t += index;
    weights_sum += index;
    depth += index;
    image_origin += index * 3;
    image_prob += index;
    image_style_origin += index * 3;
    image_back_origin += index * 3;
    image_localization_mix += index * 3;
    image_style_mix += index * 3;
    image_back_mix += index * 3;

    scalar_t t = rays_t[0]; // current ray's t

    //TODO
    scalar_t r_yellow = 0.8, g_yellow = 1.0, b_yellow = 0.0;
    scalar_t r_gray = 0.71, g_gray = 0.71, b_gray = 0.71;

    //






    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t localization_prob = image_prob[0];
    scalar_t r = image_origin[0];
    scalar_t g = image_origin[1];
    scalar_t b = image_origin[2];
    
    scalar_t r_stylization = image_style_origin[0];
    scalar_t g_stylization = image_style_origin[1];
    scalar_t b_stylization = image_style_origin[2];

    scalar_t r_background = image_back_origin[0];
    scalar_t g_background = image_back_origin[1];
    scalar_t b_background = image_back_origin[2];

    // accumulate
    uint32_t step = 0;
    while (step < n_step) {

        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /*
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        -->
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;


        r += weight * rgbs[0]; // TODO - ADD YELLOW COLOR red chanel
        g += weight * rgbs[1]; // TODO - ADD YELLOW COLOR green chanel
        b += weight * rgbs[2]; // TODO - ADD YELLOW COLOR blue chanel

        localization_prob += weight * loc_prob[0];

        r_stylization += weight * style_rgbs[0];
        g_stylization += weight * style_rgbs[1];
        b_stylization += weight * style_rgbs[2];


        r_background += weight * beck_rgbs[0];
        g_background += weight * beck_rgbs[1];
        b_background += weight * beck_rgbs[2];

        
        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        loc_prob++;
        style_rgbs += 3;
        beck_rgbs += 3;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;

    image_origin[0] = r;
    image_origin[1] = g;
    image_origin[2] = b;

    image_prob[0] = localization_prob;

    image_style_origin[0] = r_stylization;
    image_style_origin[1] = g_stylization;
    image_style_origin[2] = b_stylization;

    image_back_origin[0] = r_background;
    image_back_origin[1] = g_background;
    image_back_origin[2] = b_background;

    image_localization_mix[0] = localization_prob * r_yellow + (1 - localization_prob) * r_gray;
    image_localization_mix[1] = localization_prob * g_yellow + (1 - localization_prob) * g_gray;
    image_localization_mix[2] = localization_prob * b_yellow + (1 - localization_prob) * b_gray;
    
    image_style_mix[0] = localization_prob * r_stylization + (1 - localization_prob) * r_gray;
    image_style_mix[1] = localization_prob * g_stylization + (1 - localization_prob) * g_gray;
    image_style_mix[2] = localization_prob * b_stylization + (1 - localization_prob) * b_gray;

    image_back_mix[0] = localization_prob * r_yellow + (1 - localization_prob) * r_background;
    image_back_mix[1] = localization_prob * g_yellow + (1 - localization_prob) * g_background;
    image_back_mix[2] = localization_prob * b_yellow + (1 - localization_prob) * b_background;
}


void composite_rays_localization_int(const uint32_t n_alive, const uint32_t n_step, const float T_thresh,
                                 at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs,
                                 at::Tensor loc_prob, at::Tensor style_rgbs, at::Tensor beck_rgbs,
                                 at::Tensor deltas, at::Tensor weights, at::Tensor depth,
                                 at::Tensor image_origin, at::Tensor image_prob,
                                 at::Tensor image_style_origin,
                                 at::Tensor image_back_origin,
                                 at::Tensor image_localization_mix,
                                 at::Tensor image_style_mix,
                                 at::Tensor image_back_mix) {

    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image_origin.scalar_type(), "composite_rays_localization_int", ([&] {
        kernel_composite_rays_localization_int<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh,
                                                                                          rays_alive.data_ptr<int>(),
                                                                                          rays_t.data_ptr<scalar_t>(),
                                                                                          sigmas.data_ptr<scalar_t>(),
                                                                                          rgbs.data_ptr<scalar_t>(),
                                                                                          loc_prob.data_ptr<scalar_t>(),
                                                                                          style_rgbs.data_ptr<scalar_t>(),
                                                                                          beck_rgbs.data_ptr<scalar_t>(),
                                                                                          deltas.data_ptr<scalar_t>(),
                                                                                          weights.data_ptr<scalar_t>(),
                                                                                          depth.data_ptr<scalar_t>(),
                                                                                          image_origin.data_ptr<scalar_t>(),
                                                                                          image_prob.data_ptr<scalar_t>(),
                                                                                          image_style_origin.data_ptr<scalar_t>(),
                                                                                          image_back_origin.data_ptr<scalar_t>(),
                                                                                          image_localization_mix.data_ptr<scalar_t>(),
                                                                                          image_style_mix.data_ptr<scalar_t>(),
                                                                                          image_back_mix.data_ptr<scalar_t>());
    }));
}
