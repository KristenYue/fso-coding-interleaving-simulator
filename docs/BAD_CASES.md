# Bad cases and fixes

## BC-001: unfair coding gain from inconsistent `Eb/N0`

- Failure mode: compare codecs at the same transmitted-bit SNR even though their code rates differ.
- Consequence: redundancy differences are misreported as codec-structure gain.
- Fix: use information-bit `Eb/N0` and rate-aware noise variance; choose a rate-matched LDPC code.

## BC-002: interleaver depth confused with channel coherence

- Failure mode: change fading coherence when changing interleaver depth.
- Consequence: the effect of the design variable `D` cannot be separated from channel dynamics.
- Fix: Tasks 04C/04D hold `Lc` fixed and sweep `D` independently.

## BC-003: optimistic confidence intervals from correlated frames

- Failure mode: treat all frames or bits inside an interleaver group as independent samples.
- Consequence: confidence intervals become artificially narrow as depth increases.
- Fix: retain independent-group boundaries and compute group-aware uncertainty.

## BC-004: unsigned subtraction overflow in decoded bits

- Failure mode: subtract `uint8` arrays while counting LDPC residual errors.
- Consequence: negative differences wrap around and corrupt error metrics.
- Fix: convert decoded codewords to a signed integer representation before comparison.

## BC-005: zero observed errors presented as zero BER

- Failure mode: plot zero error events as exact zero.
- Consequence: a finite Monte Carlo run appears to prove an impossible zero probability.
- Fix: report a finite-sample upper bound and retain a flag that the point is upper-bound limited.

## BC-006: non-reproducible seeds from Python `hash()`

- Failure mode: derive per-condition seeds from the built-in hash function.
- Consequence: results can change with interpreter process and `PYTHONHASHSEED`.
- Fix: the replication path uses CRC32 over explicit condition strings.

## BC-007: long simulations lose all progress after interruption

- Failure mode: buffer the entire sweep in memory and write only at the end.
- Consequence: a stopped run must restart every completed operating point.
- Fix: replication sweeps append each completed point and skip existing conditions on restart.
