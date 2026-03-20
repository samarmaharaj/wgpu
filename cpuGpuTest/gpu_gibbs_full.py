import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WGSL_GIBBS_FULL_PROXY = r"""
override X : u32;
override Y : u32;
override Z : u32;
override G : u32;
override ALPHA : f32;
override EPS : f32;

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

fn reflect_idx(i: i32, n: i32) -> u32 {
    var v = i;
    if (v < 0) { v = -v - 1; }
    if (v >= n) { v = 2 * n - v - 1; }
    if (v < 0) { v = 0; }
    if (v >= n) { v = n - 1; }
    return u32(v);
}

fn idx4(x: u32, y: u32, z: u32, g: u32) -> u32 {
    return (((x * Y + y) * Z + z) * G) + g;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    if (x >= X || y >= Y || z >= Z) { return; }

    for (var gg = 0u; gg < G; gg = gg + 1u) {
        let xm = reflect_idx(i32(x) - 1, i32(X));
        let xp = reflect_idx(i32(x) + 1, i32(X));
        let ym = reflect_idx(i32(y) - 1, i32(Y));
        let yp = reflect_idx(i32(y) + 1, i32(Y));
        let zm = reflect_idx(i32(z) - 1, i32(Z));
        let zp = reflect_idx(i32(z) + 1, i32(Z));

        let c = vol[idx4(x, y, z, gg)];
        let vxm = vol[idx4(xm, y, z, gg)];
        let vxp = vol[idx4(xp, y, z, gg)];
        let vym = vol[idx4(x, ym, z, gg)];
        let vyp = vol[idx4(x, yp, z, gg)];
        let vzm = vol[idx4(x, y, zm, gg)];
        let vzp = vol[idx4(x, y, zp, gg)];

        let lap = (vxm + vxp + vym + vyp + vzm + vzp) - 6.0 * c;
        let ratio = abs(lap) / (abs(c) + EPS);
        let w = clamp(ALPHA * ratio, 0.0, 1.0);
        let smoothed = (vxm + vxp + vym + vyp + vzm + vzp) / 6.0;
        out[idx4(x, y, z, gg)] = (1.0 - w) * c + w * smoothed;
    }
}
"""


class GpuGibbsFull:
    """wgpu implementation scaffold for full Gibbs workflow APIs.

    Notes
    -----
    - API follows requested interface: __init__, fit, preload, fit_preloaded.
    - Uses cached pipelines.
    - Current compute kernel is a GPU Gibbs-suppression proxy; for exact DIPY
      parity the CPU reference path is available through `gibbs_cpu`.
    """

    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()

        self.device = device
        self._shader = self.device.create_shader_module(code=WGSL_GIBBS_FULL_PROXY)
        self._bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ]
        )
        self._layout = self.device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipelines = {}

    def _get_pipeline(self, x, y, z, g, alpha, eps):
        key = (int(x), int(y), int(z), int(g), float(alpha), float(eps))
        if key not in self._pipelines:
            self._pipelines[key] = self.device.create_compute_pipeline(
                layout=self._layout,
                compute={
                    "module": self._shader,
                    "entry_point": "main",
                    "constants": {
                        "X": int(x),
                        "Y": int(y),
                        "Z": int(z),
                        "G": int(g),
                        "ALPHA": float(alpha),
                        "EPS": float(eps),
                    },
                },
            )
        return self._pipelines[key]

    def preload(self, vol):
        vol = np.ascontiguousarray(vol, dtype=np.float32)
        if vol.ndim != 4:
            raise ValueError("vol must be 4D with shape (X, Y, Z, N_gradients)")
        shape = vol.shape
        buf = self.device.create_buffer_with_data(
            data=vol.reshape(-1).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        return buf, shape

    def fit_preloaded(self, buf, shape, alpha=0.8, eps=1e-6):
        x, y, z, g = shape
        n = x * y * z * g

        buf_out = self.device.create_buffer(
            size=n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        pipeline = self._get_pipeline(x, y, z, g, alpha, eps)

        bind_group = self.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf}},
                {"binding": 1, "resource": {"buffer": buf_out}},
            ],
        )

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups((x + 3) // 4, (y + 3) // 4, (z + 3) // 4)
        cp.end()
        self.device.queue.submit([enc.finish()])

        read_buf = self.device.create_buffer(
            size=n * 4,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc2 = self.device.create_command_encoder()
        enc2.copy_buffer_to_buffer(buf_out, 0, read_buf, 0, n * 4)
        self.device.queue.submit([enc2.finish()])

        read_buf.map_sync(wgpu.MapMode.READ)
        arr = np.frombuffer(read_buf.read_mapped(), dtype=np.float32).copy()
        read_buf.unmap()
        return arr.reshape(shape)

    def fit(self, vol, alpha=0.8, eps=1e-6):
        buf, shape = self.preload(vol)
        return self.fit_preloaded(buf, shape, alpha=alpha, eps=eps)


if __name__ == "__main__":
    import time

    from cpu_gibbs_full import gibbs_cpu

    rng = np.random.default_rng(42)
    vol = (rng.standard_normal((32, 32, 32, 16)) * 100 + 500).astype(np.float32)

    gpu = GpuGibbsFull()

    t0 = time.perf_counter()
    cpu_out = gibbs_cpu(vol)
    cpu_ms = (time.perf_counter() - t0) * 1000

    gpu.fit(vol)
    t0 = time.perf_counter()
    gpu_out = gpu.fit(vol)
    gpu_ms = (time.perf_counter() - t0) * 1000

    diff = np.abs(cpu_out - gpu_out)
    print(f"CPU (ms): {cpu_ms:.1f}")
    print(f"GPU (ms): {gpu_ms:.1f}")
    print(f"Max diff: {float(diff.max()):.4f}")

