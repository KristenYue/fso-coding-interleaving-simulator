"""
fso_lib.py -- Core simulation library for Task 10 (turbulence sensitivity).

Reproduces the channel / codec / interleaver conventions documented in
"FSO硕士毕设零基础完整说明书" (Tasks 01-09) so that Task 10 results are
directly comparable with the Moderate-turbulence results already in the
dissertation.

Conventions
-----------
  y = h*x + n,   h = h_a * (h_p / A0),   x in {0,1}
  h_a  : Gamma-Gamma, unit mean, product of two unit-mean Gamma RVs
  h_p  : zero-boresight Farid-Hranilovic pointing error, normalised by A0
         so that h_p/A0 = U^(1/xi^2),  U ~ Uniform(0,1)
  n    : N(0, sigma^2),  sigma^2 = 1 / (4 * R * Eb/N0)   [information-bit Eb/N0]
  Block fading: h held constant over Lc consecutive TRANSMITTED bits.
  Perfect CSI at the receiver.
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. Channel
# ----------------------------------------------------------------------------

TURBULENCE = {
    "Weak":     0.20,
    "Moderate": 0.75,
    "Strong":   2.00,
}

# Pointing-error severity -> xi = w_eq / (2 sigma_s).  Severe is the condition
# used for all coded experiments in Tasks 04C/04D/07B/08/09.
POINTING = {
    "Near-ideal": 6.08,
    "Moderate":   3.04,
    "Severe":     1.52,
}


def gg_params(rytov_var: float):
    """Al-Habash/Andrews/Phillips alpha,beta from the Rytov variance."""
    sR2 = float(rytov_var)
    sR = np.sqrt(sR2)
    a = 1.0 / (np.exp(0.49 * sR2 / (1.0 + 1.11 * sR ** (12 / 5)) ** (7 / 6)) - 1.0)
    b = 1.0 / (np.exp(0.51 * sR2 / (1.0 + 0.69 * sR ** (12 / 5)) ** (5 / 6)) - 1.0)
    return float(a), float(b)


def scintillation_index(a: float, b: float) -> float:
    return 1.0 / a + 1.0 / b + 1.0 / (a * b)


def draw_h(n_blocks: int, alpha: float, beta: float, xi: float, rng) -> np.ndarray:
    """Draw n_blocks i.i.d. composite channel gains h = h_a * (h_p/A0)."""
    x = rng.gamma(shape=alpha, scale=1.0 / alpha, size=n_blocks)
    y = rng.gamma(shape=beta, scale=1.0 / beta, size=n_blocks)
    u = rng.random(n_blocks)
    hp = u ** (1.0 / xi ** 2)
    return x * y * hp


def expand_blocks(h_blocks: np.ndarray, Lc: int, n_bits: int) -> np.ndarray:
    """Repeat each block gain Lc times and trim to n_bits."""
    return np.repeat(h_blocks, Lc)[:n_bits]


def sigma_from_ebn0(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return np.sqrt(1.0 / (4.0 * rate * ebn0))


# ----------------------------------------------------------------------------
# 2. Block interleaver (column-write / row-read), depth D
# ----------------------------------------------------------------------------

def interleave(sym_matrix: np.ndarray) -> np.ndarray:
    """sym_matrix: (D, n) -> transmitted stream of length D*n (column-wise)."""
    return sym_matrix.T.reshape(-1)


def deinterleave(stream: np.ndarray, D: int, n: int) -> np.ndarray:
    return stream.reshape(n, D).T


# ----------------------------------------------------------------------------
# 3. RS(255,223) over GF(256) -- bounded-distance decoding model
# ----------------------------------------------------------------------------
# A bounded-distance RS decoder corrects the codeword iff the number of symbol
# errors is <= t = 16.  If it fails, the (uncorrected) received information
# symbols are passed through.  This is the standard model and avoids per-frame
# Python decoding; verified against the `reedsolo` library in selftest().

RS_N, RS_K = 255, 223
RS_T = (RS_N - RS_K) // 2          # 16
RS_M = 8                            # bits per GF(256) symbol
RS_RATE = RS_K / RS_N               # 0.874510
RS_NBITS = RS_N * RS_M              # 2040 coded bits
RS_KBITS = RS_K * RS_M              # 1784 information bits


def rs_frame_errors(bit_err: np.ndarray):
    """
    bit_err : (D, RS_NBITS) boolean array of raw coded-bit errors, already
              de-interleaved back into codeword order.
    Returns  (info_bit_errors_per_frame, frame_failed_flags)
    """
    D = bit_err.shape[0]
    sym_err = bit_err.reshape(D, RS_N, RS_M).any(axis=2)      # (D, 255)
    n_sym_err = sym_err.sum(axis=1)
    failed = n_sym_err > RS_T
    # Successful frames -> 0 information bit errors.
    # Failed frames -> uncorrected information bits are delivered.
    info_bit_err = np.where(
        failed, bit_err[:, :RS_KBITS].sum(axis=1), 0
    )
    return info_bit_err, failed


# ----------------------------------------------------------------------------
# 4. LDPC(512,449) -- regular Gallager construction, batched min-sum decoding
# ----------------------------------------------------------------------------

LDPC_N, LDPC_DV, LDPC_DC = 512, 2, 16
LDPC_SEED = 7


def build_regular_h(n=LDPC_N, dv=LDPC_DV, dc=LDPC_DC, seed=LDPC_SEED):
    """Gallager regular (dv, dc) parity-check matrix."""
    rng = np.random.default_rng(seed)
    rows_per_block = n // dc                 # 32
    m = rows_per_block * dv                  # 64
    H = np.zeros((m, n), dtype=np.uint8)
    base = np.arange(n)
    for b in range(dv):
        perm = base if b == 0 else rng.permutation(n)
        for r in range(rows_per_block):
            H[b * rows_per_block + r, perm[r * dc:(r + 1) * dc]] = 1
    return H


def gf2_rref(H):
    """Return (rref matrix, pivot columns) over GF(2)."""
    A = H.copy().astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if A[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        mask = A[:, c].astype(bool).copy()
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A[:r], pivots


def build_generator(H):
    """
    Systematic-ish generator: returns (G, info_cols, k) with H @ G.T = 0 (mod 2)
    and G[:, info_cols] = I_k, so information bits are read straight back off
    the codeword at info_cols.
    """
    n = H.shape[1]
    R, pivots = gf2_rref(H)
    r = R.shape[0]
    free = [c for c in range(n) if c not in set(pivots)]
    k = n - r
    G = np.zeros((k, n), dtype=np.uint8)
    for i, fc in enumerate(free):
        G[i, fc] = 1
        # pivot column p_j gets R[j, fc]
        for j, pc in enumerate(pivots):
            G[i, pc] = R[j, fc]
    return G, np.array(free), k


class LDPCCode:
    def __init__(self, seed=LDPC_SEED):
        self.H = build_regular_h(seed=seed)
        self.G, self.info_cols, self.k = build_generator(self.H)
        self.n = self.H.shape[1]
        self.m = self.H.shape[0]
        self.rate = self.k / self.n
        # Edge lists for fast batched min-sum
        self.chk_idx, self.var_idx = np.nonzero(self.H)   # length = n_edges
        self.n_edges = self.chk_idx.size
        # order edges by check for segment ops
        order = np.argsort(self.chk_idx, kind="stable")
        self.chk_idx = self.chk_idx[order]
        self.var_idx = self.var_idx[order]
        self.dc_actual = np.bincount(self.chk_idx, minlength=self.m)
        assert np.all(self.dc_actual == self.dc_actual[0]), "irregular rows"
        self.dc = int(self.dc_actual[0])
        self.edge_by_chk = self.var_idx.reshape(self.m, self.dc)
        # For each variable node, the flat edge indices touching it.
        # Regular dv means this is a dense (n, dv) table -> fast gather-sum.
        flat_edge = np.arange(self.n_edges)
        order_v = np.argsort(self.var_idx, kind="stable")
        self.var_edges = flat_edge[order_v].reshape(self.n, LDPC_DV)

    def encode(self, info_bits):
        """info_bits: (F, k) uint8 -> codewords (F, n) uint8"""
        return (info_bits @ self.G) % 2

    def decode(self, llr, max_iter=100):
        """
        Batched normalised min-sum decoding.
        llr : (F, n) float, L = log P(x=0)/P(x=1)   (positive -> bit 0)
        returns hard decisions (F, n) uint8
        """
        F, n = llr.shape
        m, dc = self.m, self.dc
        ebc = self.edge_by_chk                        # (m, dc)
        ve = self.var_edges                           # (n, dv)
        L = llr.astype(np.float32)
        alpha = np.float32(0.75)                      # normalised min-sum scale
        out = (L < 0).astype(np.uint8)
        idx = np.arange(F)
        M = np.zeros((F, m, dc), dtype=np.float32)    # check->var messages
        Ls = L
        for _ in range(max_iter):
            if idx.size == 0:
                break
            total = Ls + M.reshape(M.shape[0], -1)[:, ve].sum(axis=2)
            V = total[:, ebc] - M                     # var->chk messages
            sgn = np.where(V >= 0, np.float32(1), np.float32(-1))
            prod_sgn = np.prod(sgn, axis=2, keepdims=True) * sgn
            absV = np.abs(V)
            part = np.partition(absV, 1, axis=2)
            minex = np.where(absV <= part[:, :, 0:1], part[:, :, 1:2], part[:, :, 0:1])
            M = alpha * prod_sgn * minex
            post = Ls + M.reshape(M.shape[0], -1)[:, ve].sum(axis=2)
            hd = (post < 0).astype(np.uint8)
            out[idx] = hd
            done = ~_syndrome(hd, ebc).any(axis=1)
            if done.any():
                keep = ~done
                idx = idx[keep]
                Ls = Ls[keep]
                M = M[keep]
        return out

    def extract_info(self, codewords):
        return codewords[:, self.info_cols]


def _syndrome(hd, ebc):
    """hd:(F,n) uint8, ebc:(m,dc) -> (F,m) uint8 parity results"""
    return hd[:, ebc].sum(axis=2) % 2


# ----------------------------------------------------------------------------
# 5. Self-tests
# ----------------------------------------------------------------------------

def selftest(verbose=True):
    ok = True
    rng = np.random.default_rng(0)

    # (a) Gamma-Gamma parameters against published values
    a, b = gg_params(0.75)
    if verbose:
        print(f"  GG(0.75): alpha={a:.4f} beta={b:.4f} (paper: 4.87 / 3.16)")
    ok &= abs(a - 4.87) < 0.02 and abs(b - 3.16) < 0.02

    # (b) unit mean of h_a and scintillation index
    x = rng.gamma(a, 1 / a, 400000) * rng.gamma(b, 1 / b, 400000)
    si_emp = x.var() / x.mean() ** 2
    si_th = scintillation_index(a, b)
    if verbose:
        print(f"  h_a mean={x.mean():.4f} (1.0)  SI emp={si_emp:.4f} th={si_th:.4f}")
    ok &= abs(x.mean() - 1) < 0.01 and abs(si_emp / si_th - 1) < 0.03

    # (c) pointing error CDF
    xi = POINTING["Severe"]
    u = rng.random(400000) ** (1 / xi ** 2)
    emp = np.mean(u <= 0.5)
    th = 0.5 ** (xi ** 2)
    if verbose:
        print(f"  pointing CDF(0.5): emp={emp:.4f} th={th:.4f}")
    ok &= abs(emp - th) < 0.005

    # (d) RS bounded-distance model vs reedsolo
    try:
        from reedsolo import RSCodec, ReedSolomonError
        rsc = RSCodec(RS_N - RS_K)
        for n_err in (16, 17):
            msg = bytearray(rng.integers(0, 256, RS_K, dtype=np.uint8).tolist())
            cw = bytearray(rsc.encode(msg))
            pos = rng.choice(RS_N, n_err, replace=False)
            for p in pos:
                cw[p] ^= 0xFF
            try:
                dec = rsc.decode(bytes(cw))[0]
                success = bytes(dec) == bytes(msg)
            except ReedSolomonError:
                success = False
            expect = n_err <= RS_T
            if verbose:
                print(f"  RS {n_err} symbol errors -> success={success} (model {expect})")
            ok &= success == expect
    except ImportError:
        if verbose:
            print("  [skip] reedsolo not installed")

    # (e) LDPC code parameters and noiseless round trip
    code = LDPCCode()
    if verbose:
        print(f"  LDPC n={code.n} m={code.m} k={code.k} R={code.rate:.6f} "
              f"(paper: 512/449/0.876953)")
    ok &= code.k == 449
    info = rng.integers(0, 2, (8, code.k)).astype(np.uint8)
    cw = code.encode(info)
    ok &= not _syndrome(cw, code.edge_by_chk).any()
    llr = np.where(cw == 1, -20.0, 20.0)
    dec = code.decode(llr, max_iter=10)
    ok &= np.array_equal(code.extract_info(dec), info)
    if verbose:
        print(f"  LDPC H*G^T=0 and noiseless round-trip: "
              f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("fso_lib self-test")
    print("PASS" if selftest() else "FAIL")
