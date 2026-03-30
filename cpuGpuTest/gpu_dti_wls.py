from math import ceil

import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WORKGROUP_SIZE = 64
MAX_DISPATCH_X = 65535
MAX_C = 7
MAX_D = 90


SHADER_SOURCE = """
override NUM_DIRS: u32;
override NUM_COEFFS: u32;

struct Params {
    num_voxels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> design: array<f32>;  // (D x C)
@group(0) @binding(2) var<storage, read> signal_data: array<f32>; // (V x D)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // (V x C)

fn design_idx(d: u32, c: u32) -> u32 { return d * NUM_COEFFS + c; }
fn signal_idx(v: u32, d: u32) -> u32 { return v * NUM_DIRS + d; }
fn out_idx(v: u32, c: u32) -> u32 { return v * NUM_COEFFS + c; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let voxel = id.x + id.y * 65535u * 64u;
    if (voxel >= params.num_voxels) { return; }

    var A: array<array<f32, 8>, 7>;

    // Build normal equations: Xt W X and Xt W y
    for (var i = 0u; i < NUM_COEFFS; i = i + 1u) {
        for (var j = 0u; j < NUM_COEFFS; j = j + 1u) {
            var sum_x: f32 = 0.0;
            for (var d = 0u; d < NUM_DIRS; d = d + 1u) {
                let s = max(signal_data[signal_idx(voxel, d)], 1e-6);
                let w = s * s;
                let xi = design[design_idx(d, i)];
                let xj = design[design_idx(d, j)];
                sum_x = sum_x + w * xi * xj;
            }
            if (i == j) {
                A[i][j] = sum_x + 1e-6;
            } else {
                A[i][j] = sum_x;
            }
        }

        var sum_y: f32 = 0.0;
        for (var d = 0u; d < NUM_DIRS; d = d + 1u) {
            let s = max(signal_data[signal_idx(voxel, d)], 1e-6);
            let w = s * s;
            let y = log(s);
            let xi = design[design_idx(d, i)];
            sum_y = sum_y + w * xi * y;
        }
        A[i][NUM_COEFFS] = sum_y;
    }

    // Gauss-Jordan elimination on 7x7 system
    for (var k = 0u; k < NUM_COEFFS; k = k + 1u) {
        let pivot = max(abs(A[k][k]), 1e-6);
        for (var j = k; j < NUM_COEFFS + 1u; j = j + 1u) {
            A[k][j] = A[k][j] / pivot;
        }

        for (var i = 0u; i < NUM_COEFFS; i = i + 1u) {
            if (i == k) { continue; }
            let f = A[i][k];
            for (var j = k; j < NUM_COEFFS + 1u; j = j + 1u) {
                A[i][j] = A[i][j] - f * A[k][j];
            }
        }
    }

    for (var c = 0u; c < NUM_COEFFS; c = c + 1u) {
        output[out_idx(voxel, c)] = A[c][NUM_COEFFS];
    }
}
"""


class GpuDTIWLS:
    def __init__(self):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No compatible GPU adapter found")
        self.device = adapter.request_device_sync(required_limits={})

        self.shader = self.device.create_shader_module(code=SHADER_SOURCE)
        self._pipelines = {}

    def _get_pipeline(self, num_dirs, num_coeffs):
        key = (int(num_dirs), int(num_coeffs))
        if key not in self._pipelines:
            self._pipelines[key] = self.device.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": self.shader,
                    "entry_point": "main",
                    "constants": {
                        "NUM_DIRS": int(num_dirs),
                        "NUM_COEFFS": int(num_coeffs),
                    },
                },
            )
        return self._pipelines[key]

    def preload(self, design, signal):
        design = np.ascontiguousarray(design, dtype=np.float32)
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        V, D = signal.shape
        if design.shape[0] != D:
            raise ValueError("design rows must equal signal num_dirs")
        C = design.shape[1]

        buf_design = self.device.create_buffer_with_data(data=design, usage=wgpu.BufferUsage.STORAGE)
        buf_signal = self.device.create_buffer_with_data(data=signal, usage=wgpu.BufferUsage.STORAGE)
        return buf_design, buf_signal, V, D, C

    def dispatch(self, num_voxels, num_dirs, num_coeffs, buf_design, buf_signal):
        params_data = np.array([num_voxels, 0, 0, 0], dtype=np.uint32)
        out_size = num_voxels * num_coeffs * 4

        buf_params = self.device.create_buffer_with_data(data=params_data, usage=wgpu.BufferUsage.UNIFORM)
        buf_out = self.device.create_buffer(size=out_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        staging = self.device.create_buffer(size=out_size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

        pipeline = self._get_pipeline(num_dirs, num_coeffs)

        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_params, "offset": 0, "size": params_data.nbytes}},
                {"binding": 1, "resource": {"buffer": buf_design, "offset": 0, "size": buf_design.size}},
                {"binding": 2, "resource": {"buffer": buf_signal, "offset": 0, "size": buf_signal.size}},
                {"binding": 3, "resource": {"buffer": buf_out, "offset": 0, "size": out_size}},
            ],
        )

        total_wg = ceil(num_voxels / WORKGROUP_SIZE)
        dispatch_x = min(total_wg, MAX_DISPATCH_X)
        dispatch_y = ceil(total_wg / MAX_DISPATCH_X)

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(dispatch_x, dispatch_y, 1)
        cp.end()
        enc.copy_buffer_to_buffer(buf_out, 0, staging, 0, out_size)
        self.device.queue.submit([enc.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(num_voxels, num_coeffs)
        staging.unmap()
        return result

    def fit(self, design, signal):
        buf_design, buf_signal, V, D, C = self.preload(design, signal)
        return self.dispatch(V, D, C, buf_design, buf_signal)


def run_gpu(num_voxels=100_000):
    from cpu_dti_wls import make_dti_design, NUM_DIRS

    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    design = make_dti_design()
    return GpuDTIWLS().fit(design, signal)
