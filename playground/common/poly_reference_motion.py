import jax.numpy as jp
from jax import vmap
import numpy as np
import pickle


# dimensions_names = [
#     0  "pos left_hip_yaw",
#     1  "pos left_hip_roll",
#     2  "pos left_hip_pitch",
#     3  "pos left_knee",
#     4  "pos left_ankle",
#     5  "pos neck_pitch",
#     6  "pos head_pitch",
#     7  "pos head_yaw",
#     8  "pos head_roll",
#     9  "pos left_antenna",
#     10 "pos right_antenna",
#     11 "pos right_hip_yaw",
#     12 "pos right_hip_roll",
#     13 "pos right_hip_pitch",
#     14 "pos right_knee",
#     15 "pos right_ankle",

#     16 "vel left_hip_yaw",
#     17 "vel left_hip_roll",
#     18 "vel left_hip_pitch",
#     19 "vel left_knee",
#     20 "vel left_ankle",
#     21 "vel neck_pitch",
#     22 "vel head_pitch",
#     23 "vel head_yaw",
#     24 "vel head_roll",
#     25 "vel left_antenna",
#     26 "vel right_antenna",
#     27 "vel right_hip_yaw",
#     28 "vel right_hip_roll",
#     29 "vel right_hip_pitch",
#     30 "vel right_knee",
#     31 "vel right_ankle",

#     32 "foot_contacts left",
#     33 "foot_contacts right",

#     34 "base_linear_vel x",
#     35 "base_linear_vel y",
#     36 "base_linear_vel z",

#     37 "base_angular_vel x",
#     38 "base_angular_vel y",
#     39 "base_angular_vel z",
# ]


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        # data = json.load(open(polynomial_coefficients))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]
            _data[dx][dy][dtheta] = [
                np.asarray(v, dtype=np.float64) for v in _coeffs.values()
            ]

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        coeffs_asc = np.asarray(
            [
                [[_data[dx][dy][dtheta] for dtheta in self.dthetas] for dy in self.dys]
                for dx in self.dxs
            ],
            dtype=np.float64,
        )
        # Keep descending-order polynomial coefficients for sample_polynomial backwards compatibility.
        self.data_array = jp.array(coeffs_asc[..., ::-1].astype(np.float32))
        self._dxs_jp = jp.array(self.dxs)
        self._dys_jp = jp.array(self.dys)
        self._dthetas_jp = jp.array(self.dthetas)

        # Precompute the discrete phase lookup table once instead of evaluating 40 deg-15 polynomials per step.
        ts = np.clip(
            np.arange(self.nb_steps_in_period, dtype=np.float64)
            / self.nb_steps_in_period,
            0.0,
            1.0,
        )
        powers = ts[:, None] ** np.arange(
            coeffs_asc.shape[-1], dtype=np.float64
        )[None, :]
        precomputed_np = np.einsum("xyzdk,tk->xyztd", coeffs_asc, powers).astype(
            np.float32
        )
        self.precomputed = jp.array(precomputed_np)

        # L/R mirror permutation and sign flips for the 40-dim reference layout:
        # [0:5] L leg pos, [5:11] head/antenna pos, [11:16] R leg pos,
        # [16:21] L leg vel, [21:27] head/antenna vel, [27:32] R leg vel,
        # [32:34] foot contacts (L, R), [34:37] base lin vel, [37:40] base ang vel.
        self._mirror_perm = jp.array(
            [
                11, 12, 13, 14, 15,
                5, 6, 7, 8, 10, 9,
                0, 1, 2, 3, 4,
                27, 28, 29, 30, 31,
                21, 22, 23, 24, 26, 25,
                16, 17, 18, 19, 20,
                33, 32,
                34, 35, 36,
                37, 38, 39,
            ],
            dtype=jp.int32,
        )
        self._mirror_signs = jp.array(
            [
                -1.0, -1.0, -1.0, 1.0, 1.0,
                1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
                -1.0, -1.0, -1.0, 1.0, 1.0,
                -1.0, -1.0, -1.0, 1.0, 1.0,
                1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
                -1.0, -1.0, -1.0, 1.0, 1.0,
                1.0, 1.0,
                1.0, -1.0, 1.0,
                -1.0, 1.0, -1.0,
            ],
            dtype=self.precomputed.dtype,
        )

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):

        dx = jp.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = jp.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = jp.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = jp.argmin(jp.abs(self._dxs_jp - dx))
        iy = jp.argmin(jp.abs(self._dys_jp - dy))
        itheta = jp.argmin(jp.abs(self._dthetas_jp - dtheta))

        return ix, iy, itheta

    def _bracket_and_weight(self, val, grid):
        """Return lower/upper grid indices and interpolation weight in [0, 1]."""
        val = jp.clip(val, grid[0], grid[-1])
        i1 = jp.clip(jp.searchsorted(grid, val, side="right"), 0, grid.shape[0] - 1)
        i0 = jp.maximum(i1 - 1, 0)
        x0 = grid[i0]
        x1 = grid[i1]
        denom = jp.maximum(x1 - x0, 1e-8)
        t = jp.clip((val - x0) / denom, 0.0, 1.0).astype(self.precomputed.dtype)
        return i0, i1, t

    def sample_polynomial(self, t, coeffs):
        return vmap(lambda c: jp.polyval(c, t))(coeffs)

    def _trilinear_raw(self, dx, dy, dtheta, step_idx):
        """Trilinearly interpolate precomputed reference motion over (dx, dy, dtheta)."""
        i0, i1, tx = self._bracket_and_weight(dx, self._dxs_jp)
        j0, j1, ty = self._bracket_and_weight(dy, self._dys_jp)
        k0, k1, tz = self._bracket_and_weight(dtheta, self._dthetas_jp)
        step_idx = jp.asarray(step_idx, dtype=jp.int32) % self.nb_steps_in_period

        c000 = self.precomputed[i0, j0, k0, step_idx]
        c001 = self.precomputed[i0, j0, k1, step_idx]
        c010 = self.precomputed[i0, j1, k0, step_idx]
        c011 = self.precomputed[i0, j1, k1, step_idx]
        c100 = self.precomputed[i1, j0, k0, step_idx]
        c101 = self.precomputed[i1, j0, k1, step_idx]
        c110 = self.precomputed[i1, j1, k0, step_idx]
        c111 = self.precomputed[i1, j1, k1, step_idx]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx

        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty

        return c0 * (1.0 - tz) + c1 * tz

    def get_reference_motion(self, dx, dy, dtheta, i):
        """Return L/R mirror-symmetrized reference motion at command (dx, dy, dtheta) and step i."""
        n = self.nb_steps_in_period
        step_idx = jp.asarray(i, dtype=jp.int32) % n
        ref_a = self._trilinear_raw(dx, dy, dtheta, step_idx)

        half_lo = (step_idx + n // 2) % n
        half_hi = (step_idx + (n + 1) // 2) % n
        ref_b = 0.5 * (
            self._trilinear_raw(dx, -dy, -dtheta, half_lo)
            + self._trilinear_raw(dx, -dy, -dtheta, half_hi)
        )
        ref_b_mirrored = ref_b[self._mirror_perm] * self._mirror_signs

        ref = 0.5 * (ref_a + ref_b_mirrored)
        contacts = (ref[32:34] >= 0.5).astype(ref.dtype)
        return ref.at[32:34].set(contacts)


if __name__ == "__main__":

    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()
