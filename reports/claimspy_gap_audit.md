# ClaimSpy Large-Gap Audit

This note audits a small set of ClaimSpy examples with the largest stability gaps between source
conditions. The goal is to understand whether the parametric advantage looks like robustness,
rigidity, or a mix of both.

## Summary Pattern

Across the audited examples, the retrieved `all` and `nonweb` conditions often collapse to
near-zero stability and are tagged as `radius_insensitive_disagreement`, while
`ss_rate_dicts_all_parametric_v0` remains near-perfectly stable.

Two recurring modes appear:

1. **Stable-correct parametric cases**: the parametric condition stays aligned with a strong
   materials-science principle or a well-supported feasibility judgment.
2. **Stable-wrong parametric cases**: the parametric condition remains internally consistent, but
   the underlying ClaimSpy quality score is poor, suggesting rigid adherence to a mistaken line of
   reasoning.

This reinforces the main quantitative takeaway: stability is a distinct property from quality.

## Audited Examples

### 1. `alloys_0014` (`example_id=13`, quality `1.0`)

- `all`: `0.000`
- `nonweb`: `0.000`
- `parametric`: `1.000`
- Retrieved conditions: `added_claims_drive_more_positive_verdict`, `radius_insensitive_disagreement`

Assessment text describes a highly feasible copper-based alloy with complexion-stabilized
precipitates, >1 GPa yield strength, and long-term 800 C stability. This looks like a
**stable-correct parametric case**: the parametric run keeps the positive judgment, while the
retrieved conditions become extremely unstable when evidence is perturbed.

### 2. `computational_tools_0019` (`example_id=43`, quality `0.0`)

- `all`: `0.000`
- `nonweb`: `0.000`
- `parametric`: `1.000`
- Retrieved conditions: `added_claims_drive_more_positive_verdict`, `radius_insensitive_disagreement`

The explanation argues that nitrogen should substitute on arsenic rather than gallium sites in
GaAs, appealing to defect-physics priors and formation-energy logic. The source gap is maximal, but
the quality score is `0.0`, making this a **stable-wrong parametric case**: the parametric
condition is extremely consistent without being supported by the existing quality label.

### 3. `computational_tools_0025` (`example_id=49`, quality `0.0`)

- `all`: `0.001`
- `nonweb`: `0.000`
- `parametric`: `1.000`
- Retrieved conditions: `added_claims_drive_more_negative_verdict`, `radius_insensitive_disagreement`

This example rejects a claim about Fe-Al alloy phase behavior, arguing that 65 at.% Al cannot have
a primary BCC iron phase and would instead form brittle intermetallics. Again, the parametric
condition is perfectly stable while the retrieved conditions are maximally unstable. Because the
quality score is `0.0`, this is another **stable-wrong parametric case**.

### 4. `modalities_0003` (`example_id=66`, quality `1.0`)

- `all`: `0.000`
- `nonweb`: `0.000`
- `parametric`: `1.000`
- Retrieved conditions: `added_claims_drive_more_positive_verdict`, `radius_insensitive_disagreement`

The explanation invokes the Hall-Petch relation and links grain size reduction to improved yield
strength in swaged and annealed chromium. This reads like a **stable-correct parametric case**:
the parametric source cleanly anchors to a strong principle, while retrieved variants are brittle.

### 5. `alloys_0008` (`example_id=7`, quality `0.0`)

- `all`: `0.119`
- `nonweb`: `0.004`
- `parametric`: `1.000`
- Retrieved conditions: `added_claims_drive_more_negative_verdict`, `radius_insensitive_disagreement`

This case argues that operating AlBeMet AM162 at 700 C is impossible because the temperature is
above the alloy's solidus. The reasoning is physically assertive and internally coherent. Still, the
quality score is `0.0`, so the case looks like **highly stable but low-quality parametric
reasoning**.

### 6. `computational_tools_0033` (`example_id=57`, quality `0.25`)

- `all`: `0.010`
- `nonweb`: `0.000`
- `parametric`: `0.975`
- Retrieved conditions: `added_claims_drive_more_negative_verdict`, `radius_insensitive_disagreement`

The explanation distinguishes between small increases in elastic modulus and large increases in
strength, concluding that the word "significantly" is overstated. This is a useful **borderline
case**: parametric is still very stable, but the quality score is only `0.25`, suggesting the
judgment may be directionally plausible yet oversharpened.

### 7. `semiconductors_0004` (`example_id=73`, quality `0.0`)

- `all`: `0.001`
- `nonweb`: `0.000`
- `parametric`: `0.964`
- Retrieved conditions: `added_claims_drive_more_positive_verdict`, `radius_insensitive_disagreement`

This example rejects the possibility of a compositionally uniform In0.4Ga0.6N film with 1.5 eV
emission, citing phase separation and bandgap arguments. The parametric condition is highly stable,
but the quality score is `0.0`, making it another **stable-wrong parametric** instance.

### 8. `superconductors_0011` (`example_id=85`, quality `0.0`)

- `all`: `0.043`
- `nonweb`: `0.000`
- `parametric`: `0.930`
- Retrieved conditions: `added_claims_drive_more_negative_verdict`, `radius_insensitive_disagreement`

The explanation notes that A15 Nb3Si has higher Tc at ambient pressure but becomes worse than the
tetragonal phase at high pressure. The parametric source again stays stable while retrieved sources
are brittle, yet the quality score remains `0.0`.

## What These Examples Suggest

### 1. The parametric effect is real, not an averaging artifact

The source gap is not just a mean effect. Individual examples repeatedly show:

- retrieved `all` or `nonweb` near `0`
- parametric near `1`

### 2. Retrieved instability is systematic, not random

The retrieved conditions often exhibit:

- `radius_insensitive_disagreement`
- a clear disagreement direction
- majority labels with high shares

So the issue is not merely noisy parsing; it looks like consistent judgment shift under added
evidence.

### 3. Parametric stability contains both desirable and undesirable cases

Among the audited examples, we see both:

- **stable-correct**: e.g. `alloys_0014`, `modalities_0003`
- **stable-wrong**: e.g. `computational_tools_0019`, `computational_tools_0025`,
  `semiconductors_0004`, `superconductors_0011`

That is exactly why the paper should not equate stability with correctness.

## Working Interpretation

The current best interpretation is:

> Parametric knowledge produces much more perturbation-stable judgments on ClaimSpy, but this
> stability reflects consistency rather than guaranteed correctness. Retrieved evidence appears more
> sensitive to support-set perturbations and can induce systematic judgment shifts.

## Next Step

The next useful extension is to replicate this same audit logic on the first external dataset,
starting with PopQA or another benchmark where parametric versus retrieved knowledge is part of the
task design rather than just our local pipeline.
