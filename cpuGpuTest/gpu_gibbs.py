import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WGSL_GIBBS = r"""
override X : u32;
override Y : u32;
override Z : u32;
override ALPHA : f32;
override EPS : f32;

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

fn reflect_idx(i: i32, n: i32) -> u32 {
    var v = i;
    if (v < 0) {
        v = -v - 1;
    }
    if (v >= n) {
        v = 2 * n - v - 1;
    }
    if (v < 0) {
        v = 0;
    }
    if (v >= n) {
        v = n - 1;
    }
    return u32(v);
}

fn idx3(x: u32, y: u32, z: u32) -> u32 {
    return (x * Y + y) * Z + z;
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    if (x >= X || y >= Y || z >= Z) { return; }

    let xm = reflect_idx(i32(x) - 1, i32(X));
    let xp = reflect_idx(i32(x) + 1, i32(X));
    let ym = reflect_idx(i32(y) - 1, i32(Y));
    let yp = reflect_idx(i32(y) + 1, i32(Y));
    let zm = reflect_idx(i32(z) - 1, i32(Z));
    let zp = reflect_idx(i32(z) + 1, i32(Z));

    let c = vol[idx3(x, y, z)];
    let vxm = vol[idx3(xm, y, z)];
    let vxp = vol[idx3(xp, y, z)];
    let vym = vol[idx3(x, ym, z)];
    let vyp = vol[idx3(x, yp, z)];
    let vzm = vol[idx3(x, y, zm)];
    let vzp = vol[idx3(x, y, zp)];

    let lap = (vxm + vxp + vym + vyp + vzm + vzp) - 6.0 * c;
    let ratio = abs(lap) / (abs(c) + EPS);
    let w = clamp(ALPHA * ratio, 0.0, 1.0);
    let smoothed = (vxm + vxp + vym + vyp + vzm + vzp) / 6.0;

    out[idx3(x, y, z)] = (1.0 - w) * c + w * smoothed;
}
"""


class GpuGibbsSuppress:
    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device
        self._shader = self.device.create_shader_module(code=WGSL_GIBBS)
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

    def _get_pipeline(self, x, y, z, alpha, eps):
        key = (int(x), int(y), int(z), float(alpha), float(eps))
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            pipeline = self.device.create_compute_pipeline(
                layout=self._layout,
                compute={
                    "module": self._shader,
                    "entry_point": "main",
                    "constants": {
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "ALPHA": float(alpha),
                        "EPS": float(eps),
                    },
                },
            )
            self._pipelines[key] = pipeline
        return pipeline

    def preload(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("data must be 3D (X, Y, Z)")
        x, y, z = data.shape

        buf_vol = self.device.create_buffer_with_data(
            data=data.flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        return buf_vol, x, y, z

    def dispatch(self, buf_vol, x, y, z, alpha=0.8, eps=1e-6):
        out_n = x * y * z
        buf_out = self.device.create_buffer(
            size=out_n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        pipeline = self._get_pipeline(x, y, z, alpha, eps)

        bind_group = self.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": buf_out}},
            ],
        )

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((x + 7) // 8, (y + 3) // 4, (z + 3) // 4)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        buf_read = self.device.create_buffer(
            size=out_n * 4,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc2 = self.device.create_command_encoder()
        enc2.copy_buffer_to_buffer(buf_out, 0, buf_read, 0, out_n * 4)
        self.device.queue.submit([enc2.finish()])

        buf_read.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(buf_read.read_mapped(), dtype=np.float32).reshape(x, y, z).copy()
        buf_read.unmap()
        return result

    def fit(self, data, alpha=0.8, eps=1e-6):
        buf_vol, x, y, z = self.preload(data)
        return self.dispatch(buf_vol, x, y, z, alpha=alpha, eps=eps)


def gibbs_suppress_gpu(data, alpha=0.8, eps=1e-6, device=None):
    gpu = GpuGibbsSuppress(device=device)
    return gpu.fit(data, alpha=alpha, eps=eps)