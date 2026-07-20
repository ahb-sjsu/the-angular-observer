# SIMODS major-revision tracker — "Keep the Angle"

Target venue: **SIAM J. Mathematics of Data Science (SIMODS)**. Referee: major revision, encourage resubmit. Full 4-claim restructure selected.

## SIMODS hard constraints (from author instructions)
- **Length:** main text (incl. appendices, excl. references) ≤ **20 pp / 700 lines** (800 for revisions); **≤ 30 pp total**. Unlimited, unrefereed **supplement** for meticulous proofs + reproducibility.
- **Abstract:** one paragraph, **≤ 250 words**, minimal formulas, **references written out in full** (not \cite numbers).
- **Title:** brief; running head ≤ 50 chars (`\headers{Keep the Angle}{...}` set).
- **Keywords + MSC** required (added: 05C50, 58J50, 62H30, 05C81, 68T09).
- **SIAM macros** encouraged (siamart build in `simods/`; needs a matching TeX toolchain — cleveref/ntheorem version clash on this MiKTeX, compiles on Overleaf).
- **Cover letter** must note overlap w/ companion Observation Theory + turboquant (prior-publication policy) and respond point-by-point to the referee.
- Submit PDF at simods.siam.org via ORCID login.

## Target structure (referee's 4 claims)
1. Exact geometry of row normalization (Lemma `lem:norm`, Prop `prop:angdist`, Thm `thm:transfer`).
2. Continuum + graph transfer — scoped fixed-m theorem **corrected for L_rw/L_sym**, + flat-torus corollary.
3. Filter dependence — heat uniformity, commute scale-dependence, **plain-filter split** (concentration + exact S¹ collapse).
4. Controlled empirical validation — torus, 3-torus, swiss roll, non-uniform density, one emergent substrate; independent graph draws; true-manifold distances.

**Move to supplement/companion:** Wolfram/observer interpretation, semantic-moral appendix, KV-cache discussion, 151-rule dimension-scaling study, and all long proofs (S¹, uniformity, Theorem-8).

## Issue tracker
| # | Referee issue | Action | Status |
|---|---|---|---|
| 1 | Omitted prior art (Sharma–Horaud–Mateus) | cite + reframe novelty (metric-geometric explanation) | **DONE** (intro + bibitem, compiles) |
| 2 | Prop 14 doesn't prove rank collapse | split: general concentration `prop:plain` + **exact S¹ theorem `thm:s1collapse`** (ρ_S, τ_K → 0), full proof | **DONE** (proved + integrated, compiles) |
| 3 | Thm 8 (`thm:cond`) L_sym/L_rw normalization gap | prove angular transfer on L_rw eigenvectors + exact row-scale invariance (Ψ_sym=√dᵢΨ_rw) | **DONE** (Step 0(iv) cancellation; degree-concentration removed; non-uniform-valid) |
| 3′ | Gap 1: Step 3 sup-norm transfer as external lemma | cite CGLewicka (`calderlewicka2022`) for L∞+C^{0,1} eigenvector rates (was mis-attributed to `calder2022`=eigenvalue/gap); state "unconditional in m at fixed m, conditional on CGLewicka"; m-dependent-constant seam → bridge to Gap 2. **Verified:** 3 perturbation estimates re-derived (match Step 4); CGLewicka C^{0,1} confirmed *for eigenvectors* (not only Poisson) from abstract+SIAM/NSF summary — cited qualitatively, no exponents asserted | **DONE** (compiles) |
| 4 | Conjecture 3 (`conj:angle`) too broad | restrict to admissible family + name `(H1)-uniform`, prove 2 endpoints (torus unconditional / heat-filter thm vs commute non-uniform ⇒ filter-specific), state sharp `conj:sigmamin` (uniform truncated-Jacobian σ_min bound); nothing proved depends on it | **DONE** (compiles, 34pp) |
| 5 | Slides between full-unnorm / truncated-norm radius; d≥3 vs d≥2 | three-level registers (L1 proved d≥3 / L2 empirical truncated / L3 borderline d=2); "Status of the radial claim" block; intro+conclusion aligned | **DONE** |
| 6 | Prop 13 (`prop:weyl`) remainder | Option A: proved `lem:dweyl` (Hörmander + Canzani–Hanin mixed-derivative, both in bib); cutoff-insensitivity remark. **C7 verified** CGLewicka covers mixed ∂ₓ·∂ᵧ | **DONE** |
| 7 | "Scaling law" overclaim (5 rule families) | → "exploratory dimension–fidelity trend"; **rule-level table primary** (real per-rule numbers from todo_experiments_result.json); fit demoted to descriptive; Prop rank "consistent with, not derive" | **DONE** |
| 8 | Methods too compressed | protocol table (`tab:protocol`); d_G/d_M framing; existing draw-level uncertainty para kept. **9 FILL** items = ledger draw-counts | **DONE** (9 FILL pending user ledger) |
| A | Terminology "basis" | abstract now says "representation"; title still "…Basis" — decision pending | ABSTRACT DONE; title TBD |
| A | Abstract overloaded | rewrite ≤250 w (237), refs spelled out (0 \cite), foreground 5 items + registers + trend | **DONE** |
| A | Byline | "Senior Member, IEEE" dropped for this SIMODS submission (both builds); global preference kept in memory | **DONE** |
| A | Figure fixes | Fig 2 "25 vs 10 draws" discrepancy; label min–max as range not CI; Fig 1 graph-draw uncertainty | TODO (needs data pass) |
| — | Length | main text 36 pp > SIMODS 30 pp cap → split supplement (Wolfram/observer, semantic, KV, dim-scaling, long proofs) | TODO (next big item) |
| — | Cover letter | point-by-point response + overlap disclosures | **DONE** (`cover-letter.tex` → 4 pp) |
| — | R-IND-5 cold verification | fresh-agent adversarial re-derivation of S¹ theorem + L_rw transfer | **DONE — both CONFIRMED**; 2 tightenings applied (Step-3 portmanteau; L²(ρ) qualifier) |
| — | R_3D table contradiction (reviewer catch) | ledger check: seeds ARE independent graph draws (rewrite permutes match order) → "deterministic" was the error; note+caption corrected | **DONE** |
| — | SIAM build re-sync | `simods/main.tex` regenerated from complete `paper.tex`; gates pass | **DONE** |
