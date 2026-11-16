# Engineering log

## 1. Separate component validation before composition

AWGN OOK, Gamma–Gamma turbulence and pointing error are validated independently in Task 01. RS and LDPC are then validated in AWGN before either codec is introduced into the fading channel. This decomposition makes a failed BER curve attributable to a smaller model boundary.

## 2. Rate-normalized RS/LDPC comparison

The comparison uses RS(255,223) and LDPC(512,449), with rates approximately 0.8745 and 0.8770. Noise variance is normalized against information-bit `Eb/N0`, so redundancy is not silently counted as coding gain.

## 3. Decouple coherence length from interleaver depth

Early experiments coupled channel behavior too closely to the selected interleaving layout. Tasks 04C/04D fix physical coherence length and sweep `D` separately. Task 05 then extracts the depth at which additional buffer produces diminishing returns.

## 4. Group-aware uncertainty

Frames in an interleaver group share fading realizations and are not independent Bernoulli samples. The refined LDPC experiment and the independent replication path retain group boundaries and report group-aware uncertainty rather than relying only on naive bit counting.

## 5. Deterministic and resumable replication

The replication path derives seeds from condition strings with CRC32. Long sweeps append completed conditions to CSV and skip them after restart, so interruption does not require restarting a full Monte Carlo campaign.

## 6. Code-only public release

The public package merges the complete Task 01–10 source tree with the independent replication source. A repository validator rejects documents, presentations, images, generated datasets and common secret formats before CI compiles every script.
