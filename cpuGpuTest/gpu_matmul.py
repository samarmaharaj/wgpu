from math import ceil

import numpy as np
import wgpu


TILE_SIZE = 16

# Tiled matrix multiply using workgroup shared memory.
# Each workgroup computes a TILE_SIZE x TILE_SIZE output tile.
# Inner loop loads TILE_SIZE-wide strips of A and B into shared memory,
# reducing global memory traffic from O(M*N*K) to O(M*N*K/TILE_SIZE).
SHADER_SOURCE = '''
const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tileA: array<f32, 256>;  // TILE * TILE
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let lr  = local_id.y;
    let lc  = local_id.x;

    var sum = 0.0f;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // Collaboratively load one TILE x TILE strip of A and B
        let a_col = t * TILE + lc;
        tileA[lr * TILE + lc] = select(0.0, A[row * dims.K + a_col],
                                       row < dims.M && a_col < dims.K);

        let b_row = t * TILE + lr;
        tileB[lr * TILE + lc] = select(0.0, B[b_row * dims.N + col],
                                       b_row < dims.K && col < dims.N);

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
'''


class GpuMatMul:

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

    def preload(self, a, b):
        """Upload a and b to GPU-resident storage buffers. Returns (buf_a, buf_b, M, K, N)."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, "Inner dimensions must match"
        buf_a = self.device.create_buffer_with_data(data=a, usage=wgpu.BufferUsage.STORAGE)
        buf_b = self.device.create_buffer_with_data(data=b, usage=wgpu.BufferUsage.STORAGE)
        return buf_a, buf_b, M, K, N

    def dispatch(self, M, K, N, buf_a, buf_b):
        """Run matmul on pre-resident GPU buffers. Only times compute + readback."""
        dims_data = np.array([M, K, N, 0], dtype=np.uint32)
        out_size = M * N * 4

        buf_dims = self.device.create_buffer_with_data(
            data=dims_data,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        buf_c = self.device.create_buffer(
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
                {"binding": 0, "resource": {"buffer": buf_dims, "offset": 0, "size": dims_data.nbytes}},
                {"binding": 1, "resource": {"buffer": buf_a, "offset": 0, "size": buf_a.size}},
                {"binding": 2, "resource": {"buffer": buf_b, "offset": 0, "size": buf_b.size}},
                {"binding": 3, "resource": {"buffer": buf_c, "offset": 0, "size": out_size}},
            ],
        )

        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(
            ceil(N / TILE_SIZE),
            ceil(M / TILE_SIZE),
            1,
        )
        compute_pass.end()
        command_encoder.copy_buffer_to_buffer(buf_c, 0, staging, 0, out_size)
        self.device.queue.submit([command_encoder.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy().reshape(M, N)
        staging.unmap()
        return result

    def multiply(self, a, b):
        """Full round-trip: upload inputs, compute, readback result."""
        buf_a, buf_b, M, K, N = self.preload(a, b)
        return self.dispatch(M, K, N, buf_a, buf_b)


def run_gpu(n=1024):
    rng = np.random.default_rng(0)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    gpu = GpuMatMul()
    return gpu.multiply(a, b)
