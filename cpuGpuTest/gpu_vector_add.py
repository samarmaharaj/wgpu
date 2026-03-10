from math import ceil

import numpy as np
import wgpu


WORKGROUP_SIZE = 64

SHADER_SOURCE = '''
@group(0) @binding(0)
var<storage, read> a: array<f32>;

@group(0) @binding(1)
var<storage, read> b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&a)) {
        return;
    }

    c[i] = a[i] + b[i];
}
'''


class GpuVectorAdder:

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

    def add(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Input arrays must have the same shape")

        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        size = a.nbytes

        buf_a = self.device.create_buffer_with_data(data=a, usage=wgpu.BufferUsage.STORAGE)
        buf_b = self.device.create_buffer_with_data(data=b, usage=wgpu.BufferUsage.STORAGE)
        buf_c = self.device.create_buffer(
            size=size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        staging = self.device.create_buffer(
            size=size,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": size}},
                {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": size}},
                {"binding": 2, "resource": {"buffer": buf_c, "offset": 0, "size": size}},
            ],
        )

        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(ceil(a.size / WORKGROUP_SIZE))
        compute_pass.end()
        command_encoder.copy_buffer_to_buffer(buf_c, 0, staging, 0, size)
        self.device.queue.submit([command_encoder.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy()
        staging.unmap()
        return result


_GPU_VECTOR_ADDER = None


def get_gpu_vector_adder():
    global _GPU_VECTOR_ADDER
    if _GPU_VECTOR_ADDER is None:
        _GPU_VECTOR_ADDER = GpuVectorAdder()
    return _GPU_VECTOR_ADDER


def gpu_vector_add(a, b, adder=None):
    if adder is None:
        adder = get_gpu_vector_adder()
    return adder.add(a, b)


def run_gpu(n=10_000_000):
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)
    return gpu_vector_add(a, b)