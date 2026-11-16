"""
run_rs_groupaware.py -- re-run the RS FSO experiments (Tasks 04C / 04D) with the
same group-aware confidence-interval methodology used for LDPC in Task 07B, so
that both codecs are reported under a single statistical convention.

Conditions (identical to Tasks 04C / 04D):
    Moderate Gamma-Gamma turbulence, sigma_R^2 = 0.75  (alpha 4.869, beta 3.165)
    Severe pointing error, xi = 1.52
    Lc = 64 bits (= 8 RS symbols)  and  Lc = 256 bits (= 32 RS symbols)
    D in {1, 8, 32, 128}
    RS(255,223) over GF(256), t = 16, symbol-level block interleaving

Statistics:
    G >= 40 independent interleaver groups per point.
    B_hat = mean over groups;  CI95 = 1.96 * s(B_g) / sqrt(G).

Resumable: writes each point to CSV, re-invoke until "ALL DONE".
"""

import csv
import os
import zlib
import sys
import time

import numpy as np

import fso_lib as F
from qfunc import Q

XI = F.POINTING["Severe"]
SIGMA_R2 = 0.75
DEPTHS = [1, 8, 32, 128]
LCS = [64, 256]
MIN_GROUPS = 40
MIN_FRAMES = 2000
TARGET_BER = 1e-3
STOP_BER = 2e-4
CSV_PATH = "rs_groupaware.csv"
SNR_START, SNR_MAX, SNR_STEP = 19.0, 40.0, 1.0

FIELDS = ["Lc_bits", "D", "ebn0_db", "ber", "ber_ci95", "fer",
          "groups", "frames", "info_bits"]


def sim_rs(alpha, beta, Lc, D, ebn0_db, rng):
    frames = max(MIN_GROUPS * D, MIN_FRAMES)
    groups = max(MIN_GROUPS, frames // D)
    nbits = D * F.RS_NBITS
    nblocks = int(np.ceil(nbits / Lc))
    scale = np.sqrt(F.RS_RATE * 10.0 ** (ebn0_db / 10.0))

    per_group, fails, tot = [], 0, 0
    chunk = max(1, min(groups, int(4e6 // nbits) or 1))
    done = 0
    while done < groups:
        g = min(chunk, groups - done)
        h = F.draw_h(g * nblocks, alpha, beta, XI, rng).reshape(g, nblocks)
        hbits = np.repeat(h, Lc, axis=1)[:, :nbits]
        err = rng.random((g, nbits)) < Q(hbits * scale)
        # symbol-level de-interleaving: transmitted symbol s = i*D + d
        err = (err.reshape(g, F.RS_N, D, F.RS_M)
                  .transpose(0, 2, 1, 3)
                  .reshape(g, D, F.RS_NBITS))
        for gi in range(g):
            ibe, failed = F.rs_frame_errors(err[gi])
            per_group.append(ibe.sum() / (D * F.RS_KBITS))
            fails += int(failed.sum())
            tot += D
        done += g
    arr = np.array(per_group)
    ber = float(arr.mean())
    ci = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return ber, ci, fails / tot, groups, tot, tot * F.RS_KBITS


def load_done():
    d = {}
    if os.path.exists(CSV_PATH):
        for r in csv.DictReader(open(CSV_PATH, newline="")):
            d[(int(r["Lc_bits"]), int(r["D"]),
               round(float(r["ebn0_db"]), 2))] = float(r["ber"])
    return d


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(budget=38.0):
    t0 = time.time()
    a, b = F.gg_params(SIGMA_R2)
    done = load_done()
    for Lc in LCS:
        for D in DEPTHS:
            snr = SNR_START
            while snr <= SNR_MAX + 1e-9:
                key = (Lc, D, round(snr, 2))
                if key in done:
                    if done[key] < STOP_BER:
                        break
                    snr += SNR_STEP
                    continue
                if time.time() - t0 > budget:
                    print(f"[budget] stop before {key}")
                    return 1
                rng = np.random.default_rng(
                    zlib.crc32(f"RSGA|{Lc}|{D}|{snr:.2f}".encode()) & 0xFFFFFFFF)
                ts = time.time()
                ber, ci, fer, G, nf, nib = sim_rs(a, b, Lc, D, snr, rng)
                append_row(dict(Lc_bits=Lc, D=D, ebn0_db=snr,
                                ber=f"{ber:.6e}", ber_ci95=f"{ci:.6e}",
                                fer=f"{fer:.6e}", groups=G, frames=nf,
                                info_bits=nib))
                done[key] = ber
                print(f"Lc={Lc:<4d} D={D:<4d} {snr:5.1f} dB  BER={ber:.3e} "
                      f"+/-{ci:.1e}  FER={fer:.3e}  G={G} "
                      f"({time.time()-ts:.1f}s)")
                if ber < STOP_BER:
                    break
                snr += SNR_STEP
    print("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 38.0))
