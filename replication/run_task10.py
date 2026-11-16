"""
run_task10.py -- Task 10: turbulence sensitivity of the recommended
interleaver depths.

Question
--------
The Pareto-optimal operating points identified in Task 08 (RS D = 8 and
LDPC D = 128 at Lc = 64 bits) were obtained under Moderate turbulence
(sigma_R^2 = 0.75) only.  Do they remain the recommended depths under
Weak (0.20) and Strong (2.00) turbulence?

Design
------
  Channel   : Gamma-Gamma (Weak / Moderate / Strong) x Severe pointing (xi = 1.52)
  Coherence : Lc = 64 transmitted bits (fixed, as in Tasks 04C / 07B)
  Depths    : D in {1, 8, 32, 128}
  Codecs    : RS(255,223) hard-decision, LDPC(512,449) soft-decision min-sum
  Metric    : decoded information BER, group-aware 95% CI, required Eb/N0
              at BER = 1e-3

Statistics
----------
All frames inside one interleaver group share the same fading realisations,
so groups -- not frames -- are the independent unit.  Every point uses
G >= 40 independent interleaver groups and the CI is computed across groups.

Resumable:每次运行处理若干个 SNR 点后写入 CSV，可重复调用直到 DONE。
"""

import csv
import os
import zlib
import sys
import time

import numpy as np

import fso_lib as F
from qfunc import Q

LC_BITS = 64
XI = F.POINTING["Severe"]
DEPTHS = [1, 8, 32, 128]
TURBS = ["Weak", "Moderate", "Strong"]
MIN_GROUPS = 40
MIN_FRAMES = 2000
TARGET_BER = 1e-3
STOP_BER = 2e-4          # stop scanning once we are safely below target
CSV_PATH = "10_turbulence_sensitivity.csv"

SNR_START = {"Weak": 15.0, "Moderate": 19.0, "Strong": 25.0}
SNR_MAX = {"Weak": 32.0, "Moderate": 40.0, "Strong": 50.0}
SNR_STEP = 1.0

FIELDS = ["turbulence", "sigma_R2", "alpha", "beta", "codec", "D",
          "ebn0_db", "ber", "ber_ci95", "fer", "groups", "frames", "info_bits"]


# ---------------------------------------------------------------------------

def group_stats(per_group_ber):
    g = len(per_group_ber)
    mean = float(np.mean(per_group_ber))
    if g > 1:
        ci = 1.96 * float(np.std(per_group_ber, ddof=1)) / np.sqrt(g)
    else:
        ci = float("nan")
    return mean, ci


def sim_rs(alpha, beta, D, ebn0_db, rng):
    """Return (ber, ci, fer, groups, frames, info_bits)."""
    frames = max(MIN_GROUPS * D, MIN_FRAMES)
    groups = max(MIN_GROUPS, frames // D)
    frames = groups * D
    nbits = D * F.RS_NBITS
    nblocks = int(np.ceil(nbits / LC_BITS))
    gamma = 10.0 ** (ebn0_db / 10.0)
    scale = np.sqrt(F.RS_RATE * gamma)

    per_group, fails, tot_frames = [], 0, 0
    chunk = max(1, min(groups, int(4e6 // nbits) or 1))
    done = 0
    while done < groups:
        g = min(chunk, groups - done)
        h = F.draw_h(g * nblocks, alpha, beta, XI, rng).reshape(g, nblocks)
        hbits = np.repeat(h, LC_BITS, axis=1)[:, :nbits]
        p = Q(hbits * scale)
        err = rng.random((g, nbits)) < p                       # transmitted order
        # RS interleaving is at GF(256) SYMBOL level: transmitted symbol
        # s = i*D + d carries symbol i of codeword d, occupying bits [8s, 8s+8).
        err = (err.reshape(g, F.RS_N, D, F.RS_M)
                  .transpose(0, 2, 1, 3)
                  .reshape(g, D, F.RS_NBITS))                   # de-interleave
        for gi in range(g):
            ibe, failed = F.rs_frame_errors(err[gi])
            per_group.append(ibe.sum() / (D * F.RS_KBITS))
            fails += int(failed.sum())
            tot_frames += D
        done += g
    ber, ci = group_stats(np.array(per_group))
    return ber, ci, fails / tot_frames, groups, tot_frames, tot_frames * F.RS_KBITS


def sim_ldpc(code, alpha, beta, D, ebn0_db, rng, max_iter=100):
    frames = max(MIN_GROUPS * D, MIN_FRAMES)
    groups = max(MIN_GROUPS, frames // D)
    n = code.n
    nbits = D * n
    nblocks = nbits // LC_BITS
    sigma = F.sigma_from_ebn0(ebn0_db, code.rate)
    inv2s2 = 1.0 / (2.0 * sigma ** 2)

    per_group, fails, tot_frames = [], 0, 0
    chunk = max(1, min(groups, int(1.5e6 // nbits) or 1))
    done = 0
    while done < groups:
        g = min(chunk, groups - done)
        info = rng.integers(0, 2, (g * D, code.k)).astype(np.uint8)
        cw = code.encode(info).reshape(g, D, n)
        tx = cw.transpose(0, 2, 1).reshape(g, nbits)            # interleave
        h = F.draw_h(g * nblocks, alpha, beta, XI, rng).reshape(g, nblocks)
        hb = np.repeat(h, LC_BITS, axis=1)
        y = hb * tx + rng.normal(0.0, sigma, (g, nbits))
        # L = log P(x=0)/P(x=1) = -(2hy - h^2) / (2 sigma^2)
        llr = (-(2.0 * hb * y - hb ** 2) * inv2s2).astype(np.float32)
        llr = llr.reshape(g, n, D).transpose(0, 2, 1).reshape(g * D, n)  # de-int
        dec = code.decode(llr, max_iter=max_iter)
        bit_err = (code.extract_info(dec) != info)
        per_frame = bit_err.sum(axis=1).reshape(g, D)
        per_group.extend((per_frame.sum(axis=1) / (D * code.k)).tolist())
        fails += int((per_frame > 0).sum())
        tot_frames += g * D
        done += g
    ber, ci = group_stats(np.array(per_group))
    return ber, ci, fails / tot_frames, groups, tot_frames, tot_frames * code.k


# ---------------------------------------------------------------------------

def load_done():
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done[(row["turbulence"], row["codec"], int(row["D"]),
                      round(float(row["ebn0_db"]), 2))] = float(row["ber"])
    return done


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(budget=38.0):
    t0 = time.time()
    code = F.LDPCCode()
    done = load_done()
    finished, pending = 0, 0

    for turb in TURBS:
        sR2 = F.TURBULENCE[turb]
        a, b = F.gg_params(sR2)
        for codec in ("RS", "LDPC"):
            for D in DEPTHS:
                snr = SNR_START[turb]
                job_done = False
                while snr <= SNR_MAX[turb] + 1e-9:
                    key = (turb, codec, D, round(snr, 2))
                    if key in done:
                        if done[key] < STOP_BER:
                            job_done = True
                            break
                        snr += SNR_STEP
                        continue
                    pending += 1
                    if time.time() - t0 > budget:
                        print(f"[budget] stop before {key}")
                        return 1
                    rng = np.random.default_rng(
                        zlib.crc32(f"{turb}|{codec}|{D}|{snr:.2f}".encode()) & 0xFFFFFFFF)
                    ts = time.time()
                    if codec == "RS":
                        r = sim_rs(a, b, D, snr, rng)
                    else:
                        r = sim_ldpc(code, a, b, D, snr, rng)
                    ber, ci, fer, G, nf, nib = r
                    append_row(dict(turbulence=turb, sigma_R2=sR2,
                                    alpha=round(a, 4), beta=round(b, 4),
                                    codec=codec, D=D, ebn0_db=snr,
                                    ber=f"{ber:.6e}", ber_ci95=f"{ci:.6e}",
                                    fer=f"{fer:.6e}", groups=G, frames=nf,
                                    info_bits=nib))
                    done[key] = ber
                    print(f"{turb:8s} {codec:4s} D={D:3d} {snr:5.1f} dB  "
                          f"BER={ber:.3e} +/-{ci:.1e}  FER={fer:.3e}  "
                          f"G={G} ({time.time()-ts:.1f}s)")
                    if ber < STOP_BER:
                        job_done = True
                        break
                    snr += SNR_STEP
                if job_done or snr > SNR_MAX[turb]:
                    finished += 1
    print(f"ALL DONE ({finished} job curves)")
    return 0


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    sys.exit(main(budget))
