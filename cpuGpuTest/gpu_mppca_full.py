"""
GPU MPPCA (Marchenko-Pastur PCA) implementation using wgpu-py.

Full GPU-resident computation with four compute shader stages:
  Stage A: patch extraction + mean + covariance (C, M, X_centered)
  Stage B: Jacobi eigendecomposition (d, W)
  Stage C: MP threshold + reconstruction (thetax, theta)
  Stage D: normalization (denoised)

Only final denoised output is read back to CPU.
"""

import numpy as np

try:
    import wgpu
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


MAX_DIM = 64
WORKGROUP_SIZE = 256
MAX_DISPATCH_X = 65535


WGSL_COMMON = r"""
struct DispatchParams {
    offset: u32,
    color_x: u32,
    color_y: u32,
    color_z: u32,
};
"""


WGSL_STAGE_A = WGSL_COMMON + r"""
override X : u32;
override Y : u32;
override Z : u32;
override DIM : u32;
override PATCH_RADIUS : i32;

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read_write> means : array<f32>;
@group(0) @binding(2) var<storage, read_write> covs : array<f32>;
@group(0) @binding(3) var<storage, read_write> x_centered : array<f32>;
@group(0) @binding(4) var<uniform> params : DispatchParams;

fn reflect_idx(i: i32, n: i32) -> u32 {
    var v = i;
    if v < 0 { v = -v - 1; }
    if v >= n { v = 2 * n - v - 1; }
    if v < 0 { v = 0; }
    if v >= n { v = n - 1; }
    return u32(v);
}

fn vol_idx(x: u32, y: u32, z: u32, c: u32) -> u32 {
    return ((x * Y + y) * Z + z) * DIM + c;
}

fn means_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

fn cov_idx(vox: u32, c1: u32, c2: u32) -> u32 {
    return vox * DIM * DIM + c1 * DIM + c2;
}

fn x_cent_idx(vox: u32, samp: u32, c: u32) -> u32 {
    let patch_side = 2u * u32(PATCH_RADIUS) + 1u;
    let num_samples = patch_side * patch_side * patch_side;
    return vox * num_samples * DIM + samp * DIM + c;
}

@compute @workgroup_size(256)
fn stage_a_means(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    let ix = tid / (Y * Z);
    let rem = tid % (Y * Z);
    let iy = rem / Z;
    let iz = rem % Z;

    let patch_side = 2u * u32(PATCH_RADIUS) + 1u;
    let num_samples = patch_side * patch_side * patch_side;
    let inv_num_samples = 1.0 / f32(num_samples);

    for (var dx = i32(0) - PATCH_RADIUS; dx <= PATCH_RADIUS; dx = dx + 1) {
        let px = reflect_idx(i32(ix) + dx, i32(X));
        for (var dy = i32(0) - PATCH_RADIUS; dy <= PATCH_RADIUS; dy = dy + 1) {
            let py = reflect_idx(i32(iy) + dy, i32(Y));
            for (var dz = i32(0) - PATCH_RADIUS; dz <= PATCH_RADIUS; dz = dz + 1) {
                let pz = reflect_idx(i32(iz) + dz, i32(Z));
                for (var c = 0u; c < DIM; c = c + 1u) {
                    means[means_idx(tid, c)] = means[means_idx(tid, c)] +
                        vol[vol_idx(px, py, pz, c)] * inv_num_samples;
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn stage_a_centered(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    let ix = tid / (Y * Z);
    let rem = tid % (Y * Z);
    let iy = rem / Z;
    let iz = rem % Z;

    var patch_samp = 0u;
    for (var dx = i32(0) - PATCH_RADIUS; dx <= PATCH_RADIUS; dx = dx + 1) {
        let px = reflect_idx(i32(ix) + dx, i32(X));
        for (var dy = i32(0) - PATCH_RADIUS; dy <= PATCH_RADIUS; dy = dy + 1) {
            let py = reflect_idx(i32(iy) + dy, i32(Y));
            for (var dz = i32(0) - PATCH_RADIUS; dz <= PATCH_RADIUS; dz = dz + 1) {
                let pz = reflect_idx(i32(iz) + dz, i32(Z));
                for (var c = 0u; c < DIM; c = c + 1u) {
                    x_centered[x_cent_idx(tid, patch_samp, c)] =
                        vol[vol_idx(px, py, pz, c)] - means[means_idx(tid, c)];
                }
                patch_samp = patch_samp + 1u;
            }
        }
    }
}

@compute @workgroup_size(256)
fn stage_a_cov(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    let patch_side = 2u * u32(PATCH_RADIUS) + 1u;
    let num_samples = patch_side * patch_side * patch_side;
    let inv_num_samples = 1.0 / f32(num_samples);

    for (var c1 = 0u; c1 < DIM; c1 = c1 + 1u) {
        for (var c2 = 0u; c2 < DIM; c2 = c2 + 1u) {
            var sum = 0.0;
            for (var s = 0u; s < num_samples; s = s + 1u) {
                sum = sum +
                    x_centered[x_cent_idx(tid, s, c1)] *
                    x_centered[x_cent_idx(tid, s, c2)];
            }
            covs[cov_idx(tid, c1, c2)] = sum * inv_num_samples;
        }
    }
}
"""


WGSL_STAGE_B = WGSL_COMMON + r"""
override X : u32;
override Y : u32;
override Z : u32;
override DIM : u32;
override MAX_SWEEPS : u32;

@group(0) @binding(0) var<storage, read> covs : array<f32>;
@group(0) @binding(1) var<storage, read_write> evals : array<f32>;
@group(0) @binding(2) var<storage, read_write> evecs : array<f32>;
@group(0) @binding(3) var<uniform> params : DispatchParams;

fn cov_idx(vox: u32, c1: u32, c2: u32) -> u32 {
    return vox * DIM * DIM + c1 * DIM + c2;
}

fn eval_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

fn evec_idx(vox: u32, c1: u32, c2: u32) -> u32 {
    return vox * DIM * DIM + c1 * DIM + c2;
}

@compute @workgroup_size(256)
fn stage_b_jacobi(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    var C: array<array<f32, 64>, 64>;
    var W: array<array<f32, 64>, 64>;

    for (var i = 0u; i < DIM; i = i + 1u) {
        for (var j = 0u; j < DIM; j = j + 1u) {
            C[i][j] = covs[cov_idx(tid, i, j)];
            W[i][j] = select(0.0, 1.0, i == j);
        }
    }

    for (var sweep = 0u; sweep < MAX_SWEEPS; sweep = sweep + 1u) {
        for (var p = 0u; p < DIM; p = p + 1u) {
            for (var q = p + 1u; q < DIM; q = q + 1u) {
                let Cpp = C[p][p];
                let Cqq = C[q][q];
                let Cpq = C[p][q];

                var angle = 0.0;
                if Cqq - Cpp == 0.0 {
                    angle = 0.78539816339;
                } else {
                    angle = 0.5 * atan2(2.0 * Cpq, Cqq - Cpp);
                }

                let c = cos(angle);
                let s = sin(angle);

                let Cpp_new = c * c * Cpp - 2.0 * s * c * Cpq + s * s * Cqq;
                let Cqq_new = s * s * Cpp + 2.0 * s * c * Cpq + c * c * Cqq;

                C[p][p] = Cpp_new;
                C[q][q] = Cqq_new;
                C[p][q] = 0.0;
                C[q][p] = 0.0;

                for (var r = 0u; r < DIM; r = r + 1u) {
                    if r != p && r != q {
                        let Crp = C[r][p];
                        let Crq = C[r][q];
                        C[r][p] = c * Crp - s * Crq;
                        C[p][r] = C[r][p];
                        C[r][q] = s * Crp + c * Crq;
                        C[q][r] = C[r][q];
                    }

                    let Wrp = W[r][p];
                    let Wrq = W[r][q];
                    W[r][p] = c * Wrp - s * Wrq;
                    W[r][q] = s * Wrp + c * Wrq;
                }
            }
        }
    }

    for (var i = 0u; i < DIM; i = i + 1u) {
        for (var j = 0u; j < DIM; j = j + 1u) {
            evecs[evec_idx(tid, i, j)] = W[i][j];
        }
        evals[eval_idx(tid, i)] = C[i][i];
    }

    for (var i = 0u; i < DIM; i = i + 1u) {
        for (var j = 0u; j + 1u < DIM - i; j = j + 1u) {
            let a = evals[eval_idx(tid, j)];
            let b = evals[eval_idx(tid, j + 1u)];
            if a > b {
                evals[eval_idx(tid, j)] = b;
                evals[eval_idx(tid, j + 1u)] = a;
                for (var r = 0u; r < DIM; r = r + 1u) {
                    let t = evecs[evec_idx(tid, r, j)];
                    evecs[evec_idx(tid, r, j)] = evecs[evec_idx(tid, r, j + 1u)];
                    evecs[evec_idx(tid, r, j + 1u)] = t;
                }
            }
        }
    }
}
"""


WGSL_STAGE_C = WGSL_COMMON + r"""
override X : u32;
override Y : u32;
override Z : u32;
override DIM : u32;
override PATCH_RADIUS : i32;
override TAU_FACTOR : f32;
override STRIDE : u32;

@group(0) @binding(0) var<storage, read> x_centered : array<f32>;
@group(0) @binding(1) var<storage, read> means : array<f32>;
@group(0) @binding(2) var<storage, read> evals : array<f32>;
@group(0) @binding(3) var<storage, read> evecs : array<f32>;
@group(0) @binding(4) var<storage, read_write> thetax : array<f32>;
@group(0) @binding(5) var<storage, read_write> theta : array<f32>;
@group(0) @binding(6) var<uniform> params : DispatchParams;

fn eval_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

fn evec_idx(vox: u32, c1: u32, c2: u32) -> u32 {
    return vox * DIM * DIM + c1 * DIM + c2;
}

fn x_cent_idx(vox: u32, samp: u32, c: u32) -> u32 {
    let patch_side = 2u * u32(PATCH_RADIUS) + 1u;
    let num_samples = patch_side * patch_side * patch_side;
    return vox * num_samples * DIM + samp * DIM + c;
}

fn means_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

fn out_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

fn theta_idx(x: u32, y: u32, z: u32) -> u32 {
    return (x * Y + y) * Z + z;
}

fn thetax_idx(x: u32, y: u32, z: u32, c: u32) -> u32 {
    return ((x * Y + y) * Z + z) * DIM + c;
}

fn reflect_idx(i: i32, n: i32) -> u32 {
    var v = i;
    if v < 0 { v = -v - 1; }
    if v >= n { v = 2 * n - v - 1; }
    if v < 0 { v = 0; }
    if v >= n { v = n - 1; }
    return u32(v);
}

fn pca_classifier_ncomps(evals_ptr: ptr<function, array<f32, 64>>, dim: u32, num_samples: u32) -> u32 {
    var start_idx : u32 = 0u;
    var eff_dim : u32 = dim;
    if dim > num_samples - 1u {
        eff_dim = num_samples - 1u;
        start_idx = dim - eff_dim;
    }

    var c : i32 = i32(eff_dim) - 1;
    var sum_all : f32 = 0.0;
    for (var i = 0u; i < eff_dim; i = i + 1u) {
        sum_all = sum_all + (*evals_ptr)[start_idx + i];
    }
    var var_est : f32 = sum_all / f32(eff_dim);

    for (var iter = 0u; iter < eff_dim; iter = iter + 1u) {
        let r = (*evals_ptr)[start_idx + u32(c)] - (*evals_ptr)[start_idx] -
                4.0 * sqrt(f32(c + 1) / f32(num_samples)) * var_est;
        if (r <= 0.0) {
            break;
        }
        if (c <= 0) {
            c = 0;
            break;
        }
        c = c - 1;

        var sum_prefix : f32 = 0.0;
        for (var j = 0u; j <= u32(c); j = j + 1u) {
            sum_prefix = sum_prefix + (*evals_ptr)[start_idx + j];
        }
        var_est = sum_prefix / f32(c + 1);
    }
    return u32(c + 1);
}

@compute @workgroup_size(256)
fn stage_c_reconstruct(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    let patch_side = 2u * u32(PATCH_RADIUS) + 1u;
    let num_samples = patch_side * patch_side * patch_side;

    let ix = tid / (Y * Z);
    let rem = tid % (Y * Z);
    let iy = rem / Z;
    let iz = rem % Z;

    if (ix % STRIDE) != params.color_x { return; }
    if (iy % STRIDE) != params.color_y { return; }
    if (iz % STRIDE) != params.color_z { return; }

    var d_local: array<f32, 64>;
    for (var i = 0u; i < DIM; i = i + 1u) {
        d_local[i] = evals[eval_idx(tid, i)];
    }

    let ncomps = pca_classifier_ncomps(&d_local, DIM, num_samples);

    var var_est : f32 = 0.0;
    for (var i = 0u; i < ncomps; i = i + 1u) {
        var_est = var_est + d_local[i];
    }
    if ncomps > 0u {
        var_est = var_est / f32(ncomps);
    }
    let _tau = TAU_FACTOR * TAU_FACTOR * var_est;

    var xproj: array<f32, 64>;
    var xrec: array<f32, 64>;
    let this_theta = 1.0 / (1.0 + f32(DIM) - f32(ncomps));

    for (var s = 0u; s < num_samples; s = s + 1u) {
        for (var j = 0u; j < DIM; j = j + 1u) {
            var sum = 0.0;
            for (var i = 0u; i < DIM; i = i + 1u) {
                let w = select(0.0, evecs[evec_idx(tid, i, j)], j >= ncomps);
                sum = sum + w * x_centered[x_cent_idx(tid, s, i)];
            }
            xproj[j] = sum;
        }

        for (var i = 0u; i < DIM; i = i + 1u) {
            var sum = 0.0;
            for (var j = 0u; j < DIM; j = j + 1u) {
                let w = select(0.0, evecs[evec_idx(tid, i, j)], j >= ncomps);
                sum = sum + w * xproj[j];
            }
            xrec[i] = sum + means[means_idx(tid, i)];
        }

        var patch_samp = 0u;
        for (var dx = i32(0) - PATCH_RADIUS; dx <= PATCH_RADIUS; dx = dx + 1) {
            let px = reflect_idx(i32(ix) + dx, i32(X));
            for (var dy = i32(0) - PATCH_RADIUS; dy <= PATCH_RADIUS; dy = dy + 1) {
                let py = reflect_idx(i32(iy) + dy, i32(Y));
                for (var dz = i32(0) - PATCH_RADIUS; dz <= PATCH_RADIUS; dz = dz + 1) {
                    let pz = reflect_idx(i32(iz) + dz, i32(Z));
                    if patch_samp == s {
                        for (var c = 0u; c < DIM; c = c + 1u) {
                            thetax[thetax_idx(px, py, pz, c)] =
                                thetax[thetax_idx(px, py, pz, c)] + xrec[c] * this_theta;
                        }
                        theta[theta_idx(px, py, pz)] = theta[theta_idx(px, py, pz)] + this_theta;
                    }
                    patch_samp = patch_samp + 1u;
                }
            }
        }
    }
}
"""


WGSL_STAGE_D = WGSL_COMMON + r"""
override X : u32;
override Y : u32;
override Z : u32;
override DIM : u32;

@group(0) @binding(0) var<storage, read> vol : array<f32>;
@group(0) @binding(1) var<storage, read> thetax : array<f32>;
@group(0) @binding(2) var<storage, read> theta : array<f32>;
@group(0) @binding(3) var<storage, read_write> denoised : array<f32>;
@group(0) @binding(4) var<uniform> params : DispatchParams;

fn out_idx(vox: u32, c: u32) -> u32 {
    return vox * DIM + c;
}

@compute @workgroup_size(256)
fn stage_d_normalize(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = params.offset + gid.x;
    let num_voxels = X * Y * Z;
    if tid >= num_voxels { return; }

    let t = theta[tid];
    for (var c = 0u; c < DIM; c = c + 1u) {
        if t > 0.0 {
            denoised[out_idx(tid, c)] = thetax[out_idx(tid, c)] / t;
        } else {
            denoised[out_idx(tid, c)] = vol[out_idx(tid, c)];
        }
    }
}
"""


class GpuMPPCAFull:
    """GPU MPPCA implementation using wgpu-py."""

    def __init__(self, device=None):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed")

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()

        self.device = device
        self._shader_modules = {}
        self._bind_group_layouts = {}
        self._pipeline_layouts = {}
        self._pipelines = {}
        self._chunk_debug_seen = set()
        self._offset_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _get_or_create_shader_module(self, name, code):
        if name not in self._shader_modules:
            self._shader_modules[name] = self.device.create_shader_module(code=code)
        return self._shader_modules[name]

    def _get_or_create_bgl(self, name, entries):
        if name not in self._bind_group_layouts:
            self._bind_group_layouts[name] = self.device.create_bind_group_layout(entries=entries)
        return self._bind_group_layouts[name]

    def _get_or_create_pipeline_layout(self, name, bgl):
        if name not in self._pipeline_layouts:
            self._pipeline_layouts[name] = self.device.create_pipeline_layout(bind_group_layouts=[bgl])
        return self._pipeline_layouts[name]

    def _get_or_create_pipeline(self, stage_name, shader_code, entry_point, constants, bgl_entries):
        key = (stage_name, entry_point, tuple(sorted(constants.items())))
        if key not in self._pipelines:
            shader = self._get_or_create_shader_module(stage_name, shader_code)
            bgl = self._get_or_create_bgl(stage_name, bgl_entries)
            layout = self._get_or_create_pipeline_layout(stage_name, bgl)
            self._pipelines[key] = self.device.create_compute_pipeline(
                layout=layout,
                compute={"module": shader, "entry_point": entry_point, "constants": constants},
            )
        return self._pipelines[key]

    def _dispatch_chunked(self, pipeline, bind_group, num_voxels, color=(0, 0, 0), workgroup_size=WORKGROUP_SIZE):
        total_workgroups = (num_voxels + workgroup_size - 1) // workgroup_size
        chunk_size = MAX_DISPATCH_X
        n_chunks = (total_workgroups + chunk_size - 1) // chunk_size
        debug_key = (num_voxels, workgroup_size)
        if debug_key not in self._chunk_debug_seen:
            print(f"n_voxels={num_voxels}, chunk_size={chunk_size}, n_chunks={n_chunks}")
            self._chunk_debug_seen.add(debug_key)

        cx, cy, cz = color
        for chunk_idx in range(n_chunks):
            workgroup_offset = chunk_idx * chunk_size
            workgroups = min(chunk_size, total_workgroups - workgroup_offset)
            offset = workgroup_offset * workgroup_size

            offset_bytes = np.array([offset, cx, cy, cz], dtype=np.uint32).tobytes()
            self.device.queue.write_buffer(self._offset_buf, 0, offset_bytes)

            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bind_group)
            cp.dispatch_workgroups(workgroups, 1, 1)
            cp.end()
            self.device.queue.submit([enc.finish()])

    def preload(self, arr, patch_radius=2):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError("arr must be 4D")

        buf_vol = self.device.create_buffer_with_data(
            data=arr.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        return buf_vol, arr.shape

    def fit(self, arr, patch_radius=2, tau_factor=None):
        buf_vol, shape = self.preload(arr, patch_radius)
        return self.fit_preloaded(
            buf_vol,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            patch_radius=patch_radius,
            tau_factor=tau_factor,
        )

    def fit_preloaded(self, buf_vol, X, Y, Z, dim, patch_radius=2, tau_factor=None):
        if dim > MAX_DIM:
            raise ValueError(f"DIM={dim} exceeds MAX_DIM={MAX_DIM}")

        if tau_factor is None:
            patch_side = 2 * patch_radius + 1
            num_samples = patch_side ** 3
            tau_factor = float(1.0 + np.sqrt(dim / num_samples))

        num_voxels = X * Y * Z
        num_samples = (2 * patch_radius + 1) ** 3

        def make_storage_buffer(num_f32):
            return self.device.create_buffer(
                size=int(num_f32) * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )

        means_buf = make_storage_buffer(num_voxels * dim)
        covs_buf = make_storage_buffer(num_voxels * dim * dim)
        x_cent_buf = make_storage_buffer(num_voxels * num_samples * dim)
        evals_buf = make_storage_buffer(num_voxels * dim)
        evecs_buf = make_storage_buffer(num_voxels * dim * dim)
        thetax_buf = make_storage_buffer(num_voxels * dim)
        theta_buf = make_storage_buffer(num_voxels)
        denoised_buf = make_storage_buffer(num_voxels * dim)

        self._run_stage_a(buf_vol, means_buf, covs_buf, x_cent_buf, X, Y, Z, dim, patch_radius, num_voxels)
        self._run_stage_b(covs_buf, evals_buf, evecs_buf, X, Y, Z, dim, num_voxels)
        self._run_stage_c(x_cent_buf, means_buf, evals_buf, evecs_buf, thetax_buf, theta_buf,
                          X, Y, Z, dim, patch_radius, tau_factor, num_voxels)
        self._run_stage_d(buf_vol, thetax_buf, theta_buf, denoised_buf, X, Y, Z, dim, num_voxels)

        return self._readback(denoised_buf, num_voxels * dim * 4, (X, Y, Z, dim))

    def _run_stage_a(self, buf_vol, means_buf, covs_buf, x_cent_buf,
                     X, Y, Z, dim, patch_radius, num_voxels):
        constants = {"X": X, "Y": Y, "Z": Z, "DIM": dim, "PATCH_RADIUS": patch_radius}
        bgl_entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]

        bg = self.device.create_bind_group(
            layout=self._get_or_create_bgl("stage_a", bgl_entries),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": means_buf}},
                {"binding": 2, "resource": {"buffer": covs_buf}},
                {"binding": 3, "resource": {"buffer": x_cent_buf}},
                {"binding": 4, "resource": {"buffer": self._offset_buf}},
            ],
        )

        p_means = self._get_or_create_pipeline("stage_a", WGSL_STAGE_A, "stage_a_means", constants, bgl_entries)
        p_centered = self._get_or_create_pipeline("stage_a", WGSL_STAGE_A, "stage_a_centered", constants, bgl_entries)
        p_cov = self._get_or_create_pipeline("stage_a", WGSL_STAGE_A, "stage_a_cov", constants, bgl_entries)

        self._dispatch_chunked(p_means, bg, num_voxels)
        self._dispatch_chunked(p_centered, bg, num_voxels)
        self._dispatch_chunked(p_cov, bg, num_voxels)

    def _run_stage_b(self, covs_buf, evals_buf, evecs_buf, X, Y, Z, dim, num_voxels):
        constants = {"X": X, "Y": Y, "Z": Z, "DIM": dim, "MAX_SWEEPS": 20}
        bgl_entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]

        bg = self.device.create_bind_group(
            layout=self._get_or_create_bgl("stage_b", bgl_entries),
            entries=[
                {"binding": 0, "resource": {"buffer": covs_buf}},
                {"binding": 1, "resource": {"buffer": evals_buf}},
                {"binding": 2, "resource": {"buffer": evecs_buf}},
                {"binding": 3, "resource": {"buffer": self._offset_buf}},
            ],
        )

        p_jacobi = self._get_or_create_pipeline("stage_b", WGSL_STAGE_B, "stage_b_jacobi", constants, bgl_entries)
        self._dispatch_chunked(p_jacobi, bg, num_voxels)

    def _run_stage_c(self, x_cent_buf, means_buf, evals_buf, evecs_buf, thetax_buf, theta_buf,
                     X, Y, Z, dim, patch_radius, tau_factor, num_voxels):
        constants = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "DIM": dim,
            "PATCH_RADIUS": patch_radius,
            "TAU_FACTOR": float(tau_factor),
            "STRIDE": 2 * patch_radius + 1,
        }
        bgl_entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]

        bg = self.device.create_bind_group(
            layout=self._get_or_create_bgl("stage_c", bgl_entries),
            entries=[
                {"binding": 0, "resource": {"buffer": x_cent_buf}},
                {"binding": 1, "resource": {"buffer": means_buf}},
                {"binding": 2, "resource": {"buffer": evals_buf}},
                {"binding": 3, "resource": {"buffer": evecs_buf}},
                {"binding": 4, "resource": {"buffer": thetax_buf}},
                {"binding": 5, "resource": {"buffer": theta_buf}},
                {"binding": 6, "resource": {"buffer": self._offset_buf}},
            ],
        )

        p_rec = self._get_or_create_pipeline(
            "stage_c",
            WGSL_STAGE_C,
            "stage_c_reconstruct",
            constants,
            bgl_entries,
        )

        stride = 2 * patch_radius + 1
        for cx in range(stride):
            for cy in range(stride):
                for cz in range(stride):
                    self._dispatch_chunked(p_rec, bg, num_voxels, color=(cx, cy, cz))

    def _run_stage_d(self, buf_vol, thetax_buf, theta_buf, denoised_buf,
                     X, Y, Z, dim, num_voxels):
        constants = {"X": X, "Y": Y, "Z": Z, "DIM": dim}
        bgl_entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]

        bg = self.device.create_bind_group(
            layout=self._get_or_create_bgl("stage_d", bgl_entries),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_vol}},
                {"binding": 1, "resource": {"buffer": thetax_buf}},
                {"binding": 2, "resource": {"buffer": theta_buf}},
                {"binding": 3, "resource": {"buffer": denoised_buf}},
                {"binding": 4, "resource": {"buffer": self._offset_buf}},
            ],
        )

        p_norm = self._get_or_create_pipeline("stage_d", WGSL_STAGE_D, "stage_d_normalize", constants, bgl_entries)
        self._dispatch_chunked(p_norm, bg, num_voxels)

    def _readback(self, src_buf, nbytes, shape):
        staging = self.device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(src_buf, 0, staging, 0, nbytes)
        self.device.queue.submit([enc.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(staging.read_mapped(), dtype=np.float32).copy()
        staging.unmap()
        return data.reshape(shape).astype(np.float32)


if __name__ == "__main__":
    from time import perf_counter
    from cpu_mppca_full import mppca_cpu

    if not HAS_WGPU:
        print("wgpu not installed; skipping GPU tests")
    else:
        rng = np.random.default_rng(42)
        data = (rng.standard_normal((8, 8, 8, 8)) * 50 + 300).astype(np.float32)

        cpu_t0 = perf_counter()
        cpu_result = mppca_cpu(data, patch_radius=1)
        cpu_ms = (perf_counter() - cpu_t0) * 1000

        gpu = GpuMPPCAFull()
        gpu_t0 = perf_counter()
        gpu_result = gpu.fit(data, patch_radius=1)
        gpu_ms = (perf_counter() - gpu_t0) * 1000

        diff = np.abs(cpu_result - gpu_result)
        print(f"CPU (ms): {cpu_ms:.2f}")
        print(f"GPU (ms): {gpu_ms:.2f}")
        print(f"Max diff: {diff.max():.4f}")
        print(f"Mean diff: {diff.mean():.4f}")
