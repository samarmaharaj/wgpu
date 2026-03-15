import numpy as np

try:
    import wgpu
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WGSL_NLM = r"""
override X : u32;
override Y : u32;
override Z : u32;
override P : u32;
override B : u32;
override H2 : f32;
override BLOCK_VOL : u32;

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

var<workgroup> w_weights : array<f32, 2048>;
var<workgroup> w_values  : array<f32, 2048>;

fn pad_stride_z() -> u32 { return Z + 2u * (P + B); }
fn pad_stride_y() -> u32 { return (Y + 2u * (P + B)) * pad_stride_z(); }

fn pad_idx(px: u32, py: u32, pz: u32) -> u32 {
    return px * pad_stride_y() + py * pad_stride_z() + pz;
}

fn patch_dist2(ax: u32, ay: u32, az: u32,
               bx: u32, by: u32, bz: u32) -> f32 {
    var s : f32 = 0.0;
    var ix : u32 = 0u;
    loop {
        if ix > 2u * P { break; }
        var iy : u32 = 0u;
        loop {
            if iy > 2u * P { break; }
            var iz : u32 = 0u;
            loop {
                if iz > 2u * P { break; }
                let va = vol[pad_idx(ax + ix, ay + iy, az + iz)];
                let vb = vol[pad_idx(bx + ix, by + iy, bz + iz)];
                let d = va - vb;
                s += d * d;
                iz++;
            }
            iy++;
        }
        ix++;
    }
    return s;
}

@compute @workgroup_size(BLOCK_VOL, 1, 1)
fn main(
    @builtin(workgroup_id) wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let out_x = wid.x;
    let out_y = wid.y;
    let out_z = wid.z;

    let cx = out_x + P + B;
    let cy = out_y + P + B;
    let cz = out_z + P + B;

    let BW = 2u * B + 1u;
    let tid = lid.x;
    let dx = tid / (BW * BW);
    let dy = (tid / BW) % BW;
    let dz = tid % BW;

    let nx = cx + dx - B;
    let ny = cy + dy - B;
    let nz = cz + dz - B;

    let dist2 = patch_dist2(cx - P, cy - P, cz - P, nx - P, ny - P, nz - P);
    let w = exp(-dist2 / H2);

    w_weights[tid] = w;
    w_values[tid] = w * vol[pad_idx(nx, ny, nz)];

    workgroupBarrier();

    if (tid == 0u) {
        var wsum : f32 = 0.0;
        var vsum : f32 = 0.0;
        for (var k : u32 = 0u; k < BLOCK_VOL; k++) {
            wsum += w_weights[k];
            vsum += w_values[k];
        }
        let flat = out_x * Y * Z + out_y * Z + out_z;
        out[flat] = vsum / wsum;
    }
}
"""


class GpuNLMeans:
    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()

        self.device = device

    def preload(self, data, patch_radius=1, block_radius=3):
        X, Y, Z = data.shape
        pad = patch_radius + block_radius
        padded = np.pad(data, pad, mode="reflect").astype(np.float32)

        buf_vol = self.device.create_buffer_with_data(
            data=padded.flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        return buf_vol, X, Y, Z

    def dispatch(self, buf_vol, X, Y, Z, patch_radius=1, block_radius=3, sigma=1.0):
        P = patch_radius
        B = block_radius
        patch_size = (2 * P + 1) ** 3
        block_vol = (2 * B + 1) ** 3
        h2 = 2.0 * (sigma ** 2) * patch_size

        if block_vol > 2048:
            raise ValueError(
                f"block_vol={block_vol} exceeds shared memory limit (2048)."
            )

        out_n = X * Y * Z
        buf_out = self.device.create_buffer(
            size=out_n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        shader = self.device.create_shader_module(code=WGSL_NLM)

        bgl = self.device.create_bind_group_layout(entries=[
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
        ])

        pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[bgl]),
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "X": X,
                    "Y": Y,
                    "Z": Z,
                    "P": P,
                    "B": B,
                    "H2": float(h2),
                    "BLOCK_VOL": block_vol,
                },
            },
        )

        bind_group = self.device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": buf_out}},
            ],
        )

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(X, Y, Z)
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
        result = np.frombuffer(buf_read.read_mapped(), dtype=np.float32).reshape(X, Y, Z).copy()
        buf_read.unmap()
        return result

    def fit(self, data, patch_radius=1, block_radius=3, sigma=1.0):
        buf_vol, X, Y, Z = self.preload(data, patch_radius=patch_radius, block_radius=block_radius)
        return self.dispatch(
            buf_vol,
            X,
            Y,
            Z,
            patch_radius=patch_radius,
            block_radius=block_radius,
            sigma=sigma,
        )


def nlmeans_patch_weights_gpu(data, patch_radius=1, block_radius=3, sigma=1.0, device=None):
    gpu = GpuNLMeans(device=device)
    return gpu.fit(data, patch_radius=patch_radius, block_radius=block_radius, sigma=sigma)
