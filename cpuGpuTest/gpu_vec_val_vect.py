from math import ceil

import numpy as np
import wgpu


WORKGROUP_SIZE = 64
MAX_DISPATCH_X = 65535


SHADER_SOURCE = """
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> evecs: array<f32>; // (n, 3, 3)
@group(0) @binding(2) var<storage, read> evals: array<f32>; // (n, 3)
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // (n, 3, 3)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x + id.y * 65535u * 64u;
    let total = params.n * 9u;
    if (tid >= total) { return; }

    let sample = tid / 9u;
    let rem = tid % 9u;
    let row = rem / 3u;
    let col = rem % 3u;

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < 3u; k++) {
        let v1 = evecs[sample * 9u + row * 3u + k];
        let d = evals[sample * 3u + k];
        let v2 = evecs[sample * 9u + col * 3u + k];
        acc += v1 * d * v2;
    }

    out[tid] = acc;
}
"""


class GpuVecValVect:
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

    def preload(self, evecs, evals):
        evecs = np.ascontiguousarray(evecs, dtype=np.float32)
        evals = np.ascontiguousarray(evals, dtype=np.float32)

        n = int(np.prod(evecs.shape[:-2]))
        evecs_2d = evecs.reshape(n, 3, 3)
        evals_2d = evals.reshape(n, 3)

        buf_evecs = self.device.create_buffer_with_data(data=evecs_2d, usage=wgpu.BufferUsage.STORAGE)
        buf_evals = self.device.create_buffer_with_data(data=evals_2d, usage=wgpu.BufferUsage.STORAGE)
        return buf_evecs, buf_evals, n

    def dispatch(self, buf_evecs, buf_evals, n):
        params = np.array([n, 0, 0, 0], dtype=np.uint32)
        out_size = n * 9 * 4

        buf_params = self.device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)
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
                {"binding": 0, "resource": {"buffer": buf_params, "offset": 0, "size": params.nbytes}},
                {"binding": 1, "resource": {"buffer": buf_evecs, "offset": 0, "size": buf_evecs.size}},
                {"binding": 2, "resource": {"buffer": buf_evals, "offset": 0, "size": buf_evals.size}},
                {"binding": 3, "resource": {"buffer": buf_out, "offset": 0, "size": out_size}},
            ],
        )

        total_threads = n * 9
        total_workgroups = ceil(total_threads / WORKGROUP_SIZE)
        dispatch_x = min(total_workgroups, MAX_DISPATCH_X)
        dispatch_y = ceil(total_workgroups / MAX_DISPATCH_X)

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(buf_out, 0, staging, 0, out_size)
        self.device.queue.submit([encoder.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(n, 3, 3)
        staging.unmap()
        return out

    def fit(self, evecs, evals):
        buf_evecs, buf_evals, n = self.preload(evecs, evals)
        return self.dispatch(buf_evecs, buf_evals, n)
