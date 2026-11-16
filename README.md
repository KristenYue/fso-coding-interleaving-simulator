# FSO Coding & Interleaving Simulator

[![Source validation](https://github.com/KristenYue/fso-coding-interleaving-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/KristenYue/fso-coding-interleaving-simulator/actions/workflows/ci.yml)

Python simulation code for studying how Reed–Solomon and LDPC forward-error-correction codes interact with interleaving depth and channel coherence in an IM/DD OOK free-space optical link.

The model combines AWGN, Gamma–Gamma atmospheric turbulence, pointing error and fixed-coherence block fading. The experiment chain separates channel validation, codec validation, coded-link simulation, statistical analysis and an independently implemented replication campaign.

## Engineering questions

- How should an RS/LDPC comparison normalize information-bit `Eb/N0` when code rates differ?
- How does interleaver depth `D` interact with physical coherence length `Lc`?
- When interleaved frames share fading realizations, how should confidence intervals account for grouped samples?
- What reliability, buffer and decoding-complexity trade-offs appear at a target BER?
- Can the main conclusions be reproduced with an independently implemented codec/channel path?

## Model and validation surface

| Layer | Implementation evidence |
|---|---|
| Channel | AWGN reference, Gamma–Gamma sampling and moments, zero-boresight pointing error, fixed-coherence fading |
| Coding | RS(255,223) with bounded-error self-checks; rate-matched LDPC(512,449) with parity and noiseless round-trip checks |
| Interleaving | Bit/symbol experiments, independently controlled `D` and `Lc`, shallow-to-deep sweeps |
| Statistics | Wilson intervals, zero-error upper bounds, group-aware intervals and common-random-number comparisons |
| Reproducibility | Fixed seeds, resumable sweeps and an independent NumPy replication implementation |

## Repository layout

```text
primary_experiments/   Task 01–10 main simulation and analysis chain
replication/           independently implemented replication experiments
docs/                  experiment map, engineering log and bad cases
tools/                 public-source and privacy validation
```

This is a code-only release. Generated CSV files, figures, documents, presentations and dissertation text are intentionally not part of the repository.

## Experiment chain

| Stage | Entry point | Purpose |
|---:|---|---|
| 01 | `01_channel_model_validation.py` | Validate AWGN, Gamma–Gamma and pointing-error components independently |
| 02 | `02_uncoded_fso_ber.py` | Establish the uncoded FSO BER baseline |
| 03 | `03_rs_awgn_validation.py` | Validate RS behavior before adding fading |
| 04 | `04c_*` and `04d_*` | Sweep RS interleaving under two fixed coherence lengths |
| 05 | `05_rs_coherence_depth_analysis.py` | Convert BER curves into required-SNR and depth trade-offs |
| 06 | `06_ldpc_awgn_validation.py` | Construct and validate the rate-matched LDPC code |
| 07 | `07b_ldpc_fso_group_aware_refinement.py` | Run LDPC over the matched fixed-coherence channel |
| 08 | `08_rs_ldpc_final_comparison.py` | Compare rate-matched RS and LDPC operating points |
| 09 | `09_uncoded_fixed_coherence_coding_gain.py` | Isolate coding gain against an uncoded matched-channel baseline |
| 10 | `10_turbulence_sensitivity.py` | Test selected configurations across turbulence regimes |

Earlier exploratory Task 04/07 scripts are retained because they record how the final fixed-coherence and group-aware designs evolved. Final comparisons use Task 04C/04D and 07B.

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run experiments from their own directory so generated output stays local and ignored by Git:

```bash
cd primary_experiments
python 01_channel_model_validation.py
python 03_rs_awgn_validation.py
```

The formal Monte Carlo sweeps are intentionally computationally expensive. Start with the validation tasks, then inspect each script's experiment settings before launching Tasks 04–10.

For the independent replication path:

```bash
cd replication
python fso_lib.py
python calc_baseline.py Moderate
```

## Reproducibility decisions

- Main experiments use explicit master seeds and spawned per-condition random streams.
- Replication seeds are derived from condition parameters with `zlib.crc32`, avoiding Python's process-dependent `hash()`.
- Long replication sweeps append completed points and skip them on restart.
- Analysis scripts consume stored experiment outputs rather than silently re-simulating them.
- The replication codec path is structurally independent from the primary library-backed path.

See [Experiment map](docs/EXPERIMENT_MAP.md), [Engineering log](docs/ENGINEERING_LOG.md) and [Bad cases](docs/BAD_CASES.md).

## License

[MIT](LICENSE)
