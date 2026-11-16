# Experiment map

## Primary campaign

The primary campaign validates each layer before composing the full coded FSO link.

```text
channel components (01)
        |
        +--> uncoded baseline (02)
        |
RS in AWGN (03) --> RS + fixed coherence/interleaving (04C/04D) --> depth analysis (05)
        |
LDPC in AWGN (06) --> LDPC + matched channel (07B) ----------------+
                                                                  |
                                   rate-matched comparison (08) <--+
                                              |
                         matched uncoded coding gain (09)
                                              |
                           turbulence sensitivity (10)
```

The ordering is deliberate: codec or channel defects should fail in a smaller validation task before they can contaminate a final comparison.

## Independent replication

`replication/fso_lib.py` implements the channel and codec path separately from the primary campaign. The remaining scripts reproduce selected RS/LDPC, confidence-interval and turbulence-sensitivity results with deterministic condition-derived seeds.

The two campaigns do not share generated outputs. Agreement therefore checks the result across implementations rather than re-plotting the same data.

## Generated files

Scripts write CSV, NPZ and figure artifacts into local `results/`, `data/` or `figures/` directories. These paths are ignored because this repository is intentionally source-only.
