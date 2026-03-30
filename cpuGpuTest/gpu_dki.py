from math import ceil

import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WORKGROUP_SIZE = 64
MAX_DISPATCH_X = 65535


SHADER_SOURCE = """
override NUM_DIRS: u32;
override NUM_PARAMS: u32;

struct Params {
    num_voxels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pinv: array<f32>;        // (P x D)
@group(0) @binding(2) var<storage, read> signal_data: array<f32>; // (V x D)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // (V x P)

fn pinv_idx(p: u32, d: u32) -> u32 { return p * NUM_DIRS + d; }
fn signal_idx(v: u32, d: u32) -> u32 { return v * NUM_DIRS + d; }
fn out_idx(v: u32, p: u32) -> u32 { return v * NUM_PARAMS + p; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let voxel = id.x + id.y * 65535u * 64u;
    if (voxel >= params.num_voxels) { return; }

    for (var p = 0u; p < NUM_PARAMS; p = p + 1u) {
        var sum: f32 = 0.0;
        for (var d = 0u; d < NUM_DIRS; d = d + 1u) {
            let s = max(signal_data[signal_idx(voxel, d)], 1e-6);
            sum = sum + pinv[pinv_idx(p, d)] * log(s);
        }
        output[out_idx(voxel, p)] = sum;
    }
}
"""


class GpuDKI:
    def __init__(self):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No compatible GPU adapter found")
        self.device = adapter.request_device_sync(required_limits={})

        self.shader = self.device.create_shader_module(code=SHADER_SOURCE)
        self._pipelines = {}

    def _get_pipeline(self, num_dirs, num_params):
        key = (int(num_dirs), int(num_params))
        if key not in self._pipelines:
            self._pipelines[key] = self.device.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": self.shader,
                    "entry_point": "main",
                    "constants": {
                        "NUM_DIRS": int(num_dirs),
                        "NUM_PARAMS": int(num_params),
                    },
                },
            )
        return self._pipelines[key]

    def preload(self, pinv, signal):
        pinv = np.ascontiguousarray(pinv, dtype=np.float32)
        signal = np.ascontiguousarray(signal, dtype=np.float32)
        V, D = signal.shape
        P = pinv.shape[0]
        if pinv.shape[1] != D:
            raise ValueError("pinv second dimension must equal signal num_dirs")

        buf_pinv = self.device.create_buffer_with_data(data=pinv, usage=wgpu.BufferUsage.STORAGE)
        buf_signal = self.device.create_buffer_with_data(data=signal, usage=wgpu.BufferUsage.STORAGE)
        return buf_pinv, buf_signal, V, D, P

    def dispatch(self, num_voxels, num_dirs, num_params, buf_pinv, buf_signal):
        params_data = np.array([num_voxels, 0, 0, 0], dtype=np.uint32)
        out_size = num_voxels * num_params * 4

        buf_params = self.device.create_buffer_with_data(data=params_data, usage=wgpu.BufferUsage.UNIFORM)
        buf_out = self.device.create_buffer(size=out_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        staging = self.device.create_buffer(size=out_size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

        pipeline = self._get_pipeline(num_dirs, num_params)

        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_params, "offset": 0, "size": params_data.nbytes}},
                {"binding": 1, "resource": {"buffer": buf_pinv, "offset": 0, "size": buf_pinv.size}},
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
        result = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(num_voxels, num_params)
        staging.unmap()
        return result

    def fit(self, pinv, signal):
        buf_pinv, buf_signal, V, D, P = self.preload(pinv, signal)
        return self.dispatch(V, D, P, buf_pinv, buf_signal)


def run_gpu(num_voxels=100_000):
    from cpu_dki import make_dki_design, NUM_DIRS

    rng = np.random.default_rng(0)
    signal = rng.uniform(0.01, 1.0, (num_voxels, NUM_DIRS)).astype(np.float32)
    design = make_dki_design()
    pinv = np.linalg.pinv(design).astype(np.float32)
    return GpuDKI().fit(pinv, signal)
