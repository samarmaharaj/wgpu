from math import ceil

import numpy as np
import wgpu


WORKGROUP_SIZE = 64
MAX_DISPATCH_X = 65535

# 1 GPU thread = 1 brain voxel.
#
# Architecture advantages over the matmul benchmark:
#   - W_inv is tiny (7×90 = 630 floats, ~2.5 KB) → fits entirely in GPU L1/shared
#     cache, so every thread reads it for near-zero cost.
#   - Each thread independently computes its 7 tensor coefficients with zero
#     inter-thread communication → no barriers, no divergence.
#   - At millions of voxels the GPU launches millions of threads simultaneously,
#     completely hiding memory latency through massive parallelism.
SHADER_SOURCE = '''
struct Params {
    num_voxels: u32,
    num_dirs:   u32,
    num_coeffs: u32,
    _pad:       u32,
};

@group(0) @binding(0) var<uniform>       params:      Params;
@group(0) @binding(1) var<storage, read> w_inv:       array<f32>; // (num_coeffs x num_dirs)
@group(0) @binding(2) var<storage, read> signal_data: array<f32>; // (num_voxels x num_dirs)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // (num_voxels x num_coeffs)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Support 2-D dispatch for very large voxel counts (> 65535 * 64 ≈ 4.2M)
    let voxel = id.x + id.y * 65535u * 64u;
    if (voxel >= params.num_voxels) { return; }

    // Compute all num_coeffs tensor elements for this single voxel.
    // w_inv is tiny and will be cached in L1 after the first warp reads it.
    for (var i = 0u; i < params.num_coeffs; i++) {
        var sum: f32 = 0.0;
        for (var j = 0u; j < params.num_dirs; j++) {
            // log() = natural log in WGSL — matches the Stejskal-Tanner log-linearization
            sum += w_inv[i * params.num_dirs + j]
                 * log(signal_data[voxel * params.num_dirs + j]);
        }
        output[voxel * params.num_coeffs + i] = sum;
    }
}
'''


class GpuDTI:

    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No compatible GPU adapter found")
        self.device = adapter.request_device_sync(required_limits={})
        shader = self.device.create_shader_module(code=SHADER_SOURCE)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )

    def preload(self, w_inv, signal):
        """Upload w_inv and signal to GPU-resident storage buffers (done once)."""
        w_inv = np.ascontiguousarray(w_inv, dtype=np.float32)
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        num_voxels, num_dirs = signal.shape
        num_coeffs = w_inv.shape[0]
        buf_w_inv = self.device.create_buffer_with_data(data=w_inv, usage=wgpu.BufferUsage.STORAGE)
        buf_signal = self.device.create_buffer_with_data(data=signal, usage=wgpu.BufferUsage.STORAGE)
        return buf_w_inv, buf_signal, num_voxels, num_dirs, num_coeffs

    def dispatch(self, num_voxels, num_dirs, num_coeffs, buf_w_inv, buf_signal):
        """Run OLS fit on pre-resident GPU buffers. Only compute + readback timed."""
        params_data = np.array([num_voxels, num_dirs, num_coeffs, 0], dtype=np.uint32)
        out_size = num_voxels * num_coeffs * 4

        buf_params = self.device.create_buffer_with_data(
            data=params_data, usage=wgpu.BufferUsage.UNIFORM,
        )
        buf_out = self.device.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        staging = self.device.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_params,  "offset": 0, "size": params_data.nbytes}},
                {"binding": 1, "resource": {"buffer": buf_w_inv,   "offset": 0, "size": buf_w_inv.size}},
                {"binding": 2, "resource": {"buffer": buf_signal,  "offset": 0, "size": buf_signal.size}},
                {"binding": 3, "resource": {"buffer": buf_out,     "offset": 0, "size": out_size}},
            ],
        )

        total_workgroups = ceil(num_voxels / WORKGROUP_SIZE)
        dispatch_x = min(total_workgroups, MAX_DISPATCH_X)
        dispatch_y = ceil(total_workgroups / MAX_DISPATCH_X)

        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1)
        compute_pass.end()
        command_encoder.copy_buffer_to_buffer(buf_out, 0, staging, 0, out_size)
        self.device.queue.submit([command_encoder.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(num_voxels, num_coeffs)
        staging.unmap()
        return result

    def fit(self, w_inv, signal):
        """Full round-trip: upload inputs, compute, readback result."""
        buf_w_inv, buf_signal, V, D, C = self.preload(w_inv, signal)
        return self.dispatch(V, D, C, buf_w_inv, buf_signal)


def run_gpu(num_voxels=1_000_000):
    from cpu_dti import make_w_inv, NUM_DIRS
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    w_inv = make_w_inv()
    return GpuDTI().fit(w_inv, signal)
