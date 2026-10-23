# CIMD numerical settings

Use the pinned CIMD source and adaptation in [COMPARISONS.md](COMPARISONS.md), deterministic Blackwell execution and eight CPU threads. Inference draws eight masks with all 1000 reverse steps. Keep the CPU model and CPU-library settings fixed as well as the GPU runtime.

The timestep embedding computes exponential frequencies on the CPU. AMD EPYC 7413 and Intel Xeon Gold 5317 produce three different values out of 64 under the tested setup, despite identical exponents and matching GPU libraries. These small differences propagate through sampling. The package records CPU model, capability, thread counts and CPU-library overrides with each run.

Deterministic training is repeatable on the specified setup. The paper's nondeterministic selected weights and every sampled mask are not recovered. Across the six endpoints, reproduced MLQ@8 changes range from −0.177037 to +0.133959. SSRB retains the higher mean on each endpoint in this reproduction. The complete comparison is included under [artifacts/reproduction/](../artifacts/reproduction/).
