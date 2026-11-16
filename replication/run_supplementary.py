"""
run_supplementary.py -- two supplementary experiments for the replication
campaign.

EXPERIMENT S1 -- column-weight sensitivity of the LDPC code
    The primary campaign uses a regular (d_v, d_c) = (2, 16) LDPC code, the
    highest-rate regular construction the library produces at n = 512.  Codes
    with d_v = 2 have minimum distance growing only logarithmically with block
    length.  A rate-matched (d_v, d_c) = (3, 24) code has the same n = 512,
    m = 64 and k = 449, hence the same rate 0.876953, but a stronger distance
    spectrum.  Running it answers whether the conclusions depend on the weak
    d_v = 2 construction.

EXPERIMENT S2 -- common random numbers (paired comparison)
    RS and LDPC are driven by the SAME block-fading realisations, so the
    difference between their BERs is estimated as a paired quantity.  The
    variance of a paired difference is much smaller than that of the difference
    of two independent estimates, which tightens the confidence interval on the
    RS-versus-LDPC gap without any additional simulation effort.

Both experiments: Moderate turbulence, Lc = 64 bits, Severe pointing,
G >= 40 independent interleaver groups.  Resumable via CSV.
"""

import csv
import os
import sys
import time
import zlib

import numpy as np

import fso_lib as F
from qfunc import Q

LC = 64
XI = F.POINTING["Severe"]
SIGMA_R2 = 0.75
DEPTHS = [1, 8, 32, 128]
MIN_GROUPS = 40
MIN_FRAMES = 2000
STOP_BER = 2e-4
SNR_START, SNR_MAX, SNR_STEP = 15.0, 40.0, 1.0

CSV_S1 = "s1_ldpc_dv3.csv"
CSV_S2 = "s2_paired_crn.csv"


def seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------- S1
def regular_random_h(n=512, m=64, dv=3, seed=7):
    """Configuration-model regular bipartite graph with duplicate-edge repair.

    Handles any (n, m, dv) with n*dv divisible by m; the resulting row weight
    is dc = n*dv/m.  Used for the d_v = 3 comparison code, whose row weight
    dc = 24 does not divide n and therefore cannot be produced by the
    block-permutation Gallager construction of fso_lib.
    """
    rng = np.random.default_rng(seed)
    dc = n * dv // m
    v = np.repeat(np.arange(n), dv)
    c = np.repeat(np.arange(m), dc)
    rng.shuffle(c)
    for _ in range(20000):
        pairs = c.astype(np.int64) * n + v
        order = np.argsort(pairs, kind="stable")
        sp = pairs[order]
        dupmask = np.zeros(len(sp), dtype=bool)
        dupmask[1:] = sp[1:] == sp[:-1]
        if not dupmask.any():
            break
        j = order[np.flatnonzero(dupmask)[0]]          # a duplicate edge slot
        k = int(rng.integers(0, len(c)))               # random partner slot
        c[j], c[k] = c[k], c[j]
    else:
        raise RuntimeError("could not build a simple regular graph")
    H = np.zeros((m, n), dtype=np.uint8)
    H[c, v] = 1
    assert H.sum() == n * dv
    return H


def build_dv3():
    """Rate-matched (d_v, d_c) = (3, 24) code: n = 512, m = 64, k = 449."""
    code = F.LDPCCode.__new__(F.LDPCCode)
    code.H = regular_random_h(n=512, m=64, dv=3, seed=7)
    code.G, code.info_cols, code.k = F.build_generator(code.H)
    code.n, code.m = code.H.shape[1], code.H.shape[0]
    code.rate = code.k / code.n
    code.chk_idx, code.var_idx = np.nonzero(code.H)
    order = np.argsort(code.chk_idx, kind="stable")
    code.chk_idx, code.var_idx = code.chk_idx[order], code.var_idx[order]
    code.n_edges = code.chk_idx.size
    dc = np.bincount(code.chk_idx).max()
    code.dc = int(dc)
    code.edge_by_chk = code.var_idx.reshape(code.m, code.dc)
    flat = np.arange(code.n_edges)
    ov = np.argsort(code.var_idx, kind="stable")
    code.var_edges = flat[ov].reshape(code.n, 3)
    assert np.ptp(code.H.sum(axis=1)) == 0 and np.ptp(code.H.sum(axis=0)) == 0
    return code


def ldpc_point(code, alpha, beta, D, snr, rng, dv):
    groups = max(MIN_GROUPS, max(MIN_GROUPS * D, MIN_FRAMES) // D)
    n = code.n
    nbits = D * n
    nblocks = nbits // LC
    sigma = F.sigma_from_ebn0(snr, code.rate)
    inv = 1.0 / (2.0 * sigma ** 2)
    per_group, fails, tot = [], 0, 0
    chunk = max(1, min(groups, int(1.5e6 // nbits) or 1))
    done = 0
    while done < groups:
        g = min(chunk, groups - done)
        info = rng.integers(0, 2, (g * D, code.k)).astype(np.uint8)
        cw = code.encode(info).reshape(g, D, n)
        tx = cw.transpose(0, 2, 1).reshape(g, nbits)
        h = F.draw_h(g * nblocks, alpha, beta, XI, rng).reshape(g, nblocks)
        hb = np.repeat(h, LC, axis=1)
        y = hb * tx + rng.normal(0.0, sigma, (g, nbits))
        llr = (-(2.0 * hb * y - hb ** 2) * inv).astype(np.float32)
        llr = llr.reshape(g, n, D).transpose(0, 2, 1).reshape(g * D, n)
        dec = code.decode(llr, max_iter=100)
        be = (code.extract_info(dec) != info).sum(axis=1).reshape(g, D)
        per_group.extend((be.sum(axis=1) / (D * code.k)).tolist())
        fails += int((be > 0).sum())
        tot += g * D
        done += g
    a = np.array(per_group)
    return (float(a.mean()),
            1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)),
            fails / tot, groups, tot, tot * code.k)


# ---------------------------------------------------------------- S2
def paired_point(code, alpha, beta, D, snr, rng):
    """RS and LDPC on the SAME block-fading realisations."""
    groups = max(MIN_GROUPS, max(MIN_GROUPS * D, MIN_FRAMES) // D)
    rs_bits = D * F.RS_NBITS
    ld_bits = D * code.n
    nb_rs = int(np.ceil(rs_bits / LC))
    nb_ld = ld_bits // LC
    nb = max(nb_rs, nb_ld)
    sigma = F.sigma_from_ebn0(snr, code.rate)
    inv = 1.0 / (2.0 * sigma ** 2)
    scale = np.sqrt(F.RS_RATE * 10.0 ** (snr / 10.0))

    rs_g, ld_g = [], []
    chunk = max(1, min(groups, int(1.2e6 // max(rs_bits, ld_bits)) or 1))
    done = 0
    while done < groups:
        g = min(chunk, groups - done)
        blocks = F.draw_h(g * nb, alpha, beta, XI, rng).reshape(g, nb)

        # --- RS on the leading nb_rs blocks ---
        hb = np.repeat(blocks[:, :nb_rs], LC, axis=1)[:, :rs_bits]
        err = rng.random((g, rs_bits)) < Q(hb * scale)
        err = (err.reshape(g, F.RS_N, D, F.RS_M)
                  .transpose(0, 2, 1, 3).reshape(g, D, F.RS_NBITS))
        for gi in range(g):
            ibe, _ = F.rs_frame_errors(err[gi])
            rs_g.append(ibe.sum() / (D * F.RS_KBITS))

        # --- LDPC on the SAME leading blocks ---
        info = rng.integers(0, 2, (g * D, code.k)).astype(np.uint8)
        cw = code.encode(info).reshape(g, D, code.n)
        tx = cw.transpose(0, 2, 1).reshape(g, ld_bits)
        hl = np.repeat(blocks[:, :nb_ld], LC, axis=1)
        y = hl * tx + rng.normal(0.0, sigma, (g, ld_bits))
        llr = (-(2.0 * hl * y - hl ** 2) * inv).astype(np.float32)
        llr = llr.reshape(g, code.n, D).transpose(0, 2, 1).reshape(g * D, code.n)
        dec = code.decode(llr, max_iter=100)
        be = (code.extract_info(dec) != info).sum(axis=1).reshape(g, D)
        ld_g.extend((be.sum(axis=1) / (D * code.k)).tolist())
        done += g

    rs = np.array(rs_g); ld = np.array(ld_g); G = len(rs)
    d = rs - ld
    return dict(
        rs_ber=float(rs.mean()), ld_ber=float(ld.mean()),
        rs_ci=1.96 * float(rs.std(ddof=1)) / np.sqrt(G),
        ld_ci=1.96 * float(ld.std(ddof=1)) / np.sqrt(G),
        diff=float(d.mean()),
        diff_ci_paired=1.96 * float(d.std(ddof=1)) / np.sqrt(G),
        diff_ci_indep=1.96 * float(np.sqrt(rs.var(ddof=1) + ld.var(ddof=1))) / np.sqrt(G),
        groups=G)


# ---------------------------------------------------------------- driver
def load(path, keyf):
    d = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path, newline="")):
            d[keyf(r)] = r
    return d


def append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


F1 = ["D", "ebn0_db", "ber", "ber_ci95", "fer", "groups", "frames", "info_bits"]
F2 = ["D", "ebn0_db", "rs_ber", "ld_ber", "rs_ci", "ld_ci",
      "diff", "diff_ci_paired", "diff_ci_indep", "groups"]


def main(which, budget=38.0):
    t0 = time.time()
    a, b = F.gg_params(SIGMA_R2)

    if which == "s1":
        code = build_dv3()
        print(f"  dv=3 code: n={code.n} m={code.m} k={code.k} R={code.rate:.6f}")
        done = load(CSV_S1, lambda r: (int(r["D"]), round(float(r["ebn0_db"]), 2)))
        for D in DEPTHS:
            snr = SNR_START
            while snr <= SNR_MAX + 1e-9:
                key = (D, round(snr, 2))
                if key in done:
                    if float(done[key]["ber"]) < STOP_BER:
                        break
                    snr += SNR_STEP
                    continue
                if time.time() - t0 > budget:
                    print(f"[budget] stop before {key}")
                    return 1
                rng = np.random.default_rng(seed("S1", D, f"{snr:.2f}"))
                ber, ci, fer, G, nf, nib = ldpc_point(code, a, b, D, snr, rng, 3)
                append(CSV_S1, F1, dict(D=D, ebn0_db=snr, ber=f"{ber:.6e}",
                                        ber_ci95=f"{ci:.6e}", fer=f"{fer:.6e}",
                                        groups=G, frames=nf, info_bits=nib))
                done[key] = {"ber": ber}
                print(f"  dv3 D={D:<4d}{snr:5.1f} dB  BER={ber:.3e} +/-{ci:.1e}  G={G}")
                if ber < STOP_BER:
                    break
                snr += SNR_STEP
        print("S1 DONE")
        return 0

    code = F.LDPCCode()
    done = load(CSV_S2, lambda r: (int(r["D"]), round(float(r["ebn0_db"]), 2)))
    for D in DEPTHS:
        for snr in (22.0, 24.0, 26.0, 28.0):
            key = (D, round(snr, 2))
            if key in done:
                continue
            if time.time() - t0 > budget:
                print(f"[budget] stop before {key}")
                return 1
            rng = np.random.default_rng(seed("S2", D, f"{snr:.2f}"))
            r = paired_point(code, a, b, D, snr, rng)
            append(CSV_S2, F2, dict(D=D, ebn0_db=snr,
                                    **{k: f"{v:.6e}" if isinstance(v, float) else v
                                       for k, v in r.items()}))
            print(f"  paired D={D:<4d}{snr:5.1f} dB  RS={r['rs_ber']:.2e} "
                  f"LDPC={r['ld_ber']:.2e}  diff={r['diff']:+.2e} "
                  f"paired±{r['diff_ci_paired']:.1e} vs indep±{r['diff_ci_indep']:.1e}")
    print("S2 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "s1",
                  float(sys.argv[2]) if len(sys.argv) > 2 else 38.0))
