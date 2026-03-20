import numpy as np

try:
    import wgpu

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


WGSL_MPPCA_HYBRID = r"""
override X : u32;
override Y : u32;
override Z : u32;
override C : u32;
override P : i32;

struct DispatchInfo {
    offset : u32,
}

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read_write> means : array<f32>;
@group(0) @binding(2) var<storage, read_write> covs : array<f32>;
@group(0) @binding(3) var<storage, read> proj : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;
@group(0) @binding(5) var<uniform> dispatch_info : DispatchInfo;

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

fn voxel_id(x: u32, y: u32, z: u32) -> u32 {
    return (x * Y + y) * Z + z;
}

fn idx4(x: u32, y: u32, z: u32, c: u32) -> u32 {
    return voxel_id(x, y, z) * C + c;
}

fn mean_idx(v: u32, c: u32) -> u32 {
    return v * C + c;
}

fn cov_idx(v: u32, c1: u32, c2: u32) -> u32 {
    return v * C * C + c1 * C + c2;
}

@compute @workgroup_size(64)
fn mean_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = gid.x + dispatch_info.offset;
    let total = X * Y * Z * C;
    if (tid >= total) { return; }

    let v = tid / C;
    let c = tid % C;

    let x = v / (Y * Z);
    let rem = v % (Y * Z);
    let y = rem / Z;
    let z = rem % Z;

    let side = 2 * P + 1;
    let patch_count = f32(side * side * side);

    var s : f32 = 0.0;
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
                s += vol[idx4(xx, yy, zz, c)];
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }

    means[mean_idx(v, c)] = s / patch_count;
}

@compute @workgroup_size(64)
fn cov_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = gid.x + dispatch_info.offset;
    let total = X * Y * Z * C * C;
    if (tid >= total) { return; }

    let cc = C * C;
    let v = tid / cc;
    let rem = tid % cc;
    let c1 = rem / C;
    let c2 = rem % C;

    let x = v / (Y * Z);
    let remv = v % (Y * Z);
    let y = remv / Z;
    let z = remv % Z;

    let m1 = means[mean_idx(v, c1)];
    let m2 = means[mean_idx(v, c2)];

    let side = 2 * P + 1;
    let patch_count = f32(side * side * side);

    var s : f32 = 0.0;
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
                let v1 = vol[idx4(xx, yy, zz, c1)] - m1;
                let v2 = vol[idx4(xx, yy, zz, c2)] - m2;
                s += v1 * v2;
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }

    covs[cov_idx(v, c1, c2)] = s / patch_count;
}

@compute @workgroup_size(64)
fn recon_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = gid.x + dispatch_info.offset;
    let total = X * Y * Z * C;
    if (tid >= total) { return; }

    let v = tid / C;
    let c_out = tid % C;

    let ox = v / (Y * Z);
    let rem = v % (Y * Z);
    let oy = rem / Z;
    let oz = rem % Z;

    var accum : f32 = 0.0;
    var count : u32 = 0u;

    var dcx : i32 = -P;
    loop {
        if (dcx > P) { break; }
        var dcy : i32 = -P;
        loop {
            if (dcy > P) { break; }
            var dcz : i32 = -P;
            loop {
                if (dcz > P) { break; }

                let cx = reflect_idx(i32(ox) + dcx, i32(X));
                let cy = reflect_idx(i32(oy) + dcy, i32(Y));
                let cz = reflect_idx(i32(oz) + dcz, i32(Z));
                let cv = voxel_id(cx, cy, cz);

                var dotv : f32 = 0.0;
                var k : u32 = 0u;
                loop {
                    if (k >= C) { break; }
                    let xk = vol[idx4(ox, oy, oz, k)] - means[mean_idx(cv, k)];
                    let bkc = proj[cov_idx(cv, k, c_out)];
                    dotv += xk * bkc;
                    k += 1u;
                }

                let xest = dotv + means[mean_idx(cv, c_out)];
                accum += xest;
                count += 1u;
                dcz += 1;
            }
            dcy += 1;
        }
        dcx += 1;
    }

    out[mean_idx(v, c_out)] = accum / f32(count);
}
"""


def _pca_classifier(evals_asc, nvoxels):
    vals = evals_asc
    if vals.size > nvoxels - 1:
        vals = vals[-(nvoxels - 1):]

    var = np.mean(vals)
    c = vals.size - 1
    r = vals[c] - vals[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var
    while r > 0 and c > 0:
        var = np.mean(vals[:c])
        c -= 1
        r = vals[c] - vals[0] - 4.0 * np.sqrt((c + 1.0) / nvoxels) * var
    return var, c + 1


class GpuMPPCAHybrid:
    _WG_SIZE = 64
    _MAX_GROUPS_X = 65535

    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device
        self._shader = self.device.create_shader_module(code=WGSL_MPPCA_HYBRID)
        self._pipelines = {}
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
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        self._layout = self.device.create_pipeline_layout(bind_group_layouts=[self._bgl])

    def _dispatch_1d_chunked(self, compute_pass, total_items, offset_buf):
        max_items = self._WG_SIZE * self._MAX_GROUPS_X
        offset = 0
        while offset < total_items:
            remaining = total_items - offset
            dispatch_items = min(remaining, max_items)
            groups_x = (dispatch_items + self._WG_SIZE - 1) // self._WG_SIZE
            self.device.queue.write_buffer(
                offset_buf,
                0,
                np.array([offset], dtype=np.uint32).tobytes(),
            )
            compute_pass.dispatch_workgroups(groups_x, 1, 1)
            offset += groups_x * self._WG_SIZE

    def _get_pipelines(self, x, y, z, channels, patch_radius):
        key = (int(x), int(y), int(z), int(channels), int(patch_radius))
        pipes = self._pipelines.get(key)
        if pipes is not None:
            return pipes

        constants = {
            "X": x,
            "Y": y,
            "Z": z,
            "C": channels,
            "P": patch_radius,
        }
        pipes = {
            "mean": self.device.create_compute_pipeline(
                layout=self._layout,
                compute={
                    "module": self._shader,
                    "entry_point": "mean_main",
                    "constants": constants,
                },
            ),
            "cov": self.device.create_compute_pipeline(
                layout=self._layout,
                compute={
                    "module": self._shader,
                    "entry_point": "cov_main",
                    "constants": constants,
                },
            ),
            "recon": self.device.create_compute_pipeline(
                layout=self._layout,
                compute={
                    "module": self._shader,
                    "entry_point": "recon_main",
                    "constants": constants,
                },
            ),
        }
        self._pipelines[key] = pipes
        return pipes

    def preload(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError("data must be 4D (X, Y, Z, C)")
        if data.shape[3] > 64:
            raise ValueError("channels > 64 not supported")

        x, y, z, channels = data.shape
        buf_vol = self.device.create_buffer_with_data(
            data=data.flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        return buf_vol, x, y, z, channels

    def _readback(self, src_buf, nbytes, dtype, shape):
        staging = self.device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(src_buf, 0, staging, 0, nbytes)
        self.device.queue.submit([enc.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        arr = np.frombuffer(staging.read_mapped(), dtype=dtype).copy().reshape(shape)
        staging.unmap()
        return arr

    def _compute_means_covs(self, buf_vol, x, y, z, channels, patch_radius):
        nvox = x * y * z
        means_n = nvox * channels
        cov_n = nvox * channels * channels

        buf_means = self.device.create_buffer(
            size=means_n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buf_cov = self.device.create_buffer(
            size=cov_n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        dummy_proj = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)
        dummy_out = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)
        offset_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        pipes = self._get_pipelines(x, y, z, channels, patch_radius)
        bind_group = self.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": buf_means}},
                {"binding": 2, "resource": {"buffer": buf_cov}},
                {"binding": 3, "resource": {"buffer": dummy_proj}},
                {"binding": 4, "resource": {"buffer": dummy_out}},
                {"binding": 5, "resource": {"buffer": offset_buf}},
            ],
        )

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipes["mean"])
        cp.set_bind_group(0, bind_group)
        self._dispatch_1d_chunked(cp, means_n, offset_buf)
        cp.set_pipeline(pipes["cov"])
        cp.set_bind_group(0, bind_group)
        self._dispatch_1d_chunked(cp, cov_n, offset_buf)
        cp.end()
        self.device.queue.submit([enc.finish()])

        means = self._readback(buf_means, means_n * 4, np.float32, (nvox, channels))
        covs = self._readback(buf_cov, cov_n * 4, np.float32, (nvox, channels, channels))
        return means, covs

    def _cpu_projectors(self, covs, patch_voxels):
        nvox, channels, _ = covs.shape
        proj = np.zeros((nvox, channels, channels), dtype=np.float32)
        for i in range(nvox):
            evals, evecs = np.linalg.eigh(covs[i])
            _, ncomps = _pca_classifier(evals, patch_voxels)
            evecs[:, :ncomps] = 0.0
            proj[i] = (evecs @ evecs.T).astype(np.float32)
        return proj

    def _reconstruct(self, buf_vol, x, y, z, channels, patch_radius, means, proj):
        nvox = x * y * z
        means_buf = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(means, dtype=np.float32).flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        proj_buf = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(proj, dtype=np.float32).flatten().tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        out_buf = self.device.create_buffer(
            size=nvox * channels * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        dummy_cov = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)
        offset_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        pipes = self._get_pipelines(x, y, z, channels, patch_radius)
        bind_group = self.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": means_buf}},
                {"binding": 2, "resource": {"buffer": dummy_cov}},
                {"binding": 3, "resource": {"buffer": proj_buf}},
                {"binding": 4, "resource": {"buffer": out_buf}},
                {"binding": 5, "resource": {"buffer": offset_buf}},
            ],
        )

        out_n = nvox * channels
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipes["recon"])
        cp.set_bind_group(0, bind_group)
        self._dispatch_1d_chunked(cp, out_n, offset_buf)
        cp.end()
        self.device.queue.submit([enc.finish()])

        out = self._readback(out_buf, out_n * 4, np.float32, (x, y, z, channels))
        return out

    def fit_preloaded(self, buf_vol, x, y, z, channels, patch_radius=1):
        means, covs = self._compute_means_covs(buf_vol, x, y, z, channels, patch_radius)
        patch_voxels = (2 * patch_radius + 1) ** 3
        proj = self._cpu_projectors(covs, patch_voxels)
        out = self._reconstruct(buf_vol, x, y, z, channels, patch_radius, means, proj)
        return out

    def fit(self, data, patch_radius=1):
        buf_vol, x, y, z, channels = self.preload(data)
        return self.fit_preloaded(buf_vol, x, y, z, channels, patch_radius=patch_radius)


def mppca_hybrid_gpu(data, patch_radius=1, device=None):
    gpu = GpuMPPCAHybrid(device=device)
    return gpu.fit(data, patch_radius=patch_radius)