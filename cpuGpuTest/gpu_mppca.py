import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WGSL_MPPCA = r"""
override X : u32;
override Y : u32;
override Z : u32;
override C : u32;
override P : i32;
override TAU : f32;
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

fn idx4(x: u32, y: u32, z: u32, c: u32) -> u32 {
    return (((x * Y + y) * Z + z) * C + c);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    if (x >= X || y >= Y || z >= Z) { return; }

    var means : array<f32, 64>;
    var vars : array<f32, 64>;

    var c : u32 = 0u;
    var sigma2 : f32 = 0.0;
    let side = 2 * P + 1;
    let patch_count = f32(side * side * side);

    loop {
        if (c >= C) { break; }

        var s1 : f32 = 0.0;
        var dx : i32 = -P;
        loop {
            if (dx > P) { break; }
            var dy : i32 = -P;
            loop {
                if (dy > P) { break; }
                var dz : i32 = -P;
                loop {
                    if (dz > P) { break; }
                    let xx = reflect_idx(i32(x) + dx, i32(X));
                    let yy = reflect_idx(i32(y) + dy, i32(Y));
                    let zz = reflect_idx(i32(z) + dz, i32(Z));
                    s1 += vol[idx4(xx, yy, zz, c)];
                    dz += 1;
                }
                dy += 1;
            }
            dx += 1;
        }

        let m = s1 / patch_count;
        means[c] = m;

        var s2 : f32 = 0.0;
        var dx2 : i32 = -P;
        loop {
            if (dx2 > P) { break; }
            var dy2 : i32 = -P;
            loop {
                if (dy2 > P) { break; }
                var dz2 : i32 = -P;
                loop {
                    if (dz2 > P) { break; }
                    let xx2 = reflect_idx(i32(x) + dx2, i32(X));
                    let yy2 = reflect_idx(i32(y) + dy2, i32(Y));
                    let zz2 = reflect_idx(i32(z) + dz2, i32(Z));
                    let d = vol[idx4(xx2, yy2, zz2, c)] - m;
                    s2 += d * d;
                    dz2 += 1;
                }
                dy2 += 1;
            }
            dx2 += 1;
        }

        let v = s2 / patch_count;
        vars[c] = v;
        sigma2 += v;
        c += 1u;
    }

    sigma2 = sigma2 / f32(C);

    var c2 : u32 = 0u;
    loop {
        if (c2 >= C) { break; }
        let center = vol[idx4(x, y, z, c2)];
        let denom = vars[c2] + EPS;
        let shrink = max(0.0, 1.0 - (TAU * sigma2) / denom);
        out[idx4(x, y, z, c2)] = means[c2] + shrink * (center - means[c2]);
        c2 += 1u;
    }
}
"""


class GpuMPPCAProxy:
    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device
        self._shader = self.device.create_shader_module(code=WGSL_MPPCA)
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

    def _get_pipeline(self, x, y, z, channels, patch_radius, tau, eps):
        key = (int(x), int(y), int(z), int(channels), int(patch_radius), float(tau), float(eps))
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
                        "C": channels,
                        "P": patch_radius,
                        "TAU": float(tau),
                        "EPS": float(eps),
                    },
                },
            )
            self._pipelines[key] = pipeline
        return pipeline

    def preload(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError("data must be 4D (X, Y, Z, C)")
        if data.shape[3] > 64:
            raise ValueError("channels > 64 are not supported in this prototype")

        x, y, z, channels = data.shape
        buf_vol = self.device.create_buffer_with_data(
            data=data.flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        return buf_vol, x, y, z, channels

    def dispatch(self, buf_vol, x, y, z, channels, patch_radius=1, tau=1.2, eps=1e-6):
        out_n = x * y * z * channels
        buf_out = self.device.create_buffer(
            size=out_n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        pipeline = self._get_pipeline(x, y, z, channels, patch_radius, tau, eps)

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
        compute_pass.dispatch_workgroups((x + 3) // 4, (y + 3) // 4, (z + 3) // 4)
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
        result = np.frombuffer(buf_read.read_mapped(), dtype=np.float32).reshape(x, y, z, channels).copy()
        buf_read.unmap()
        return result

    def fit(self, data, patch_radius=1, tau=1.2, eps=1e-6):
        buf_vol, x, y, z, channels = self.preload(data)
        return self.dispatch(
            buf_vol,
            x,
            y,
            z,
            channels,
            patch_radius=patch_radius,
            tau=tau,
            eps=eps,
        )


def mppca_proxy_gpu(data, patch_radius=1, tau=1.2, eps=1e-6, device=None):
    gpu = GpuMPPCAProxy(device=device)
    return gpu.fit(data, patch_radius=patch_radius, tau=tau, eps=eps)