from math import ceil

import numpy as np
import wgpu


WORKGROUP_SIZE = 64
MAX_DISPATCH_X = 65535


def _prepare_lengths(points):
    diffs = points[:, 1:, :] - points[:, :-1, :]
    seg_lens = np.linalg.norm(diffs, axis=-1)
    return np.concatenate(
        [np.zeros((points.shape[0], 1), dtype=np.float32), np.cumsum(seg_lens, axis=1, dtype=np.float32)],
        axis=1,
    )


SHADER_SOURCE = """
struct Params {
    n_streamlines: u32,
    n_in: u32,
    n_out: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<f32>; // (S, n_in, 3)
@group(0) @binding(2) var<storage, read> cumlen: array<f32>; // (S, n_in)
@group(0) @binding(3) var<storage, read> tnorm: array<f32>;  // (n_out,)
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // (S, n_out, 3)

fn pidx(s: u32, i: u32, c: u32) -> u32 {
    return ((s * params.n_in + i) * 3u) + c;
}

fn oidx(s: u32, j: u32, c: u32) -> u32 {
    return ((s * params.n_out + j) * 3u) + c;
}

fn cidx(s: u32, i: u32) -> u32 {
    return s * params.n_in + i;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x + id.y * 65535u * 64u;
    let total = params.n_streamlines * params.n_out;
    if (tid >= total) { return; }

    let s = tid / params.n_out;
    let j = tid % params.n_out;

    let total_len = cumlen[cidx(s, params.n_in - 1u)];
    if (total_len <= 1e-12) {
        out[oidx(s, j, 0u)] = points[pidx(s, 0u, 0u)];
        out[oidx(s, j, 1u)] = points[pidx(s, 0u, 1u)];
        out[oidx(s, j, 2u)] = points[pidx(s, 0u, 2u)];
        return;
    }

    let t_query = tnorm[j] * total_len;

    var seg: u32 = 0u;
    if (params.n_in > 1u) {
        for (var i: u32 = 0u; i < params.n_in - 1u; i++) {
            let l0 = cumlen[cidx(s, i)];
            let l1 = cumlen[cidx(s, i + 1u)];
            if (t_query >= l0 && t_query <= l1) {
                seg = i;
                break;
            }
            if (t_query > l1) {
                seg = i + 1u;
            }
        }
        if (seg >= params.n_in - 1u) {
            seg = params.n_in - 2u;
        }
    }

    let l0 = cumlen[cidx(s, seg)];
    let l1 = cumlen[cidx(s, seg + 1u)];
    let denom = l1 - l0;
    let alpha = select((t_query - l0) / max(denom, 1e-12), 0.0, denom <= 1e-12);

    for (var c: u32 = 0u; c < 3u; c++) {
        let v0 = points[pidx(s, seg, c)];
        let v1 = points[pidx(s, seg + 1u, c)];
        out[oidx(s, j, c)] = (1.0 - alpha) * v0 + alpha * v1;
    }
}
"""


class GpuSetNumberOfPoints:
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

    def preload(self, streamlines, nb_points=50):
        streamlines = np.ascontiguousarray(streamlines, dtype=np.float32)
        n_streamlines, n_in, _ = streamlines.shape
        cumlen = _prepare_lengths(streamlines)
        t_norm = np.linspace(0.0, 1.0, nb_points, dtype=np.float32)

        buf_points = self.device.create_buffer_with_data(data=streamlines, usage=wgpu.BufferUsage.STORAGE)
        buf_cumlen = self.device.create_buffer_with_data(data=cumlen, usage=wgpu.BufferUsage.STORAGE)
        buf_tnorm = self.device.create_buffer_with_data(data=t_norm, usage=wgpu.BufferUsage.STORAGE)
        return buf_points, buf_cumlen, buf_tnorm, n_streamlines, n_in, nb_points

    def dispatch(self, buf_points, buf_cumlen, buf_tnorm, n_streamlines, n_in, n_out):
        params = np.array([n_streamlines, n_in, n_out, 0], dtype=np.uint32)
        out_size = n_streamlines * n_out * 3 * 4

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
                {"binding": 1, "resource": {"buffer": buf_points, "offset": 0, "size": buf_points.size}},
                {"binding": 2, "resource": {"buffer": buf_cumlen, "offset": 0, "size": buf_cumlen.size}},
                {"binding": 3, "resource": {"buffer": buf_tnorm, "offset": 0, "size": buf_tnorm.size}},
                {"binding": 4, "resource": {"buffer": buf_out, "offset": 0, "size": out_size}},
            ],
        )

        total_threads = n_streamlines * n_out
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
        out = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(n_streamlines, n_out, 3)
        staging.unmap()
        return out

    def fit(self, streamlines, nb_points=50):
        state = self.preload(streamlines, nb_points=nb_points)
        return self.dispatch(*state)
