# Review: "Keep the Angle: A Geometry-Preserving Basis in Spectral Embeddings"

**Venue:** SIAM Journal on Mathematics of Data Science (SIMODS)
**Materials reviewed:** `main.tex` (1,276 lines) and `supplement.tex` (1,178 lines). Figures and the class file were not available, so the documents could not be compiled; all structural checks below were done directly on the sources.

---

## Summary of the paper

The paper studies the polar decomposition of the truncated, commute-weighted spectral embedding of a graph: writing each node's low-mode embedding as magnitude × direction, it argues that the geodesic content of the embedding lives almost entirely in the direction (the row-normalized embedding of Ng–Jordan–Weiss), while the magnitude is degree/density nuisance. The paper contributes (i) a corollary of von Luxburg–Radl–Hein transferring the pairwise commute-distance degeneracy to the diagonal, so the full unnormalized commute radius is asymptotically 1/√kᵢ (Cor. 3.1, with a new gauge-plus-trace argument in Appendix A); (ii) an exact identity showing the angular stretch is the tangential component of the embedding differential (Prop. 3.7), elevated to a deterministic finite-distance transfer theorem (Thm. 3.8) whose hypotheses (A1)–(A3) reduce the angular claim to continuum eigenmap properties; (iii) a conditional graph-to-manifold theorem via the sup-norm eigenvector rates of Calder–García Trillos–Lewicka, made unconditional on the flat torus and round sphere; (iv) a sharp filter dichotomy in the continuum — uniform-in-truncation bi-Lipschitzness for the heat filter, a proven failure of fixed-scale upper uniformity for the commute filter, an exact rank collapse of the plain filter on S¹, and a uniform-in-m lower angular bound on flat tori; and (v) torus experiments with controls, direct theorem verification, and a weighting ablation. An extended empirical program is deferred to a companion "Paper I.b."

## Overall evaluation

This is a substantial and, in its mathematical core, careful piece of work. The mechanism identified — tangential noncollapse rather than radius constancy — is clean and genuinely clarifying; the transfer-theorem abstraction (Thm. 3.8) is the right way to organize the conditional results; and the filter dichotomy (heat uniform, commute scale-dependent, plain rank-collapsing, with the exact S¹ computations) is a satisfying resolution of the ablation findings. Table 1 (claim status at a glance) is exemplary scientific practice, and the uncertainty reporting is unusually honest.

I checked the following arguments line by line and found them correct: the normalization identity (Lem. 3.6) and its infinitesimal form (Prop. 3.7); the truncation bound (Prop. 3.4); the arithmetic of the transfer theorem and its assembly in supplement Steps 1–4; the torus and sphere corollary constants; the small-ball ratio law μ(κ) = 1 − κ⁻ᵈ and the Daniels conversion in Prop. 3.13; the gauge/trace algebra of Appendix A (Steps 1–3, including the error bookkeeping); the S¹ rank-collapse proof (Dirichlet kernel, two-scale equidistribution, Spearman/Kendall limits); the dyadic-block lower bound of Prop. SM2.3; and the Weyl-order bookkeeping in Prop. 4.x/Remark SM2.2 and the filter-threshold table. I found no mathematical errors in these.

The problems are structural, not mathematical. The submission currently depends on an unpublished companion paper whose citation is undefined; the supplement is wired to a version of the main article that no longer exists (five cross-references to nonexistent sections, including in the proof setting of the central theorem); and the abstract claims empirical breadth (substrate robustness across five graph families) that the main paper's own experiments — tori only — do not contain.

**Recommendation: major revision.** The mathematics appears sound and the contribution is appropriate for SIMODS; the revision needed is primarily to make the submission self-contained and internally consistent.

---

## Major comments

1. **Undefined citation and load-bearing dependence on "Paper I.b."** `\cite{paperIb}` (main.tex lines 118, 1093) has no `\bibitem` — it will render as **[?]** — and "Paper I.b" is invoked roughly a dozen times (lines 85, 118, 181, 215, 291, 322, 718–719, 776, 881, 913, 1067, 1093) as the home of essential material: the α = 1 density-normalized angle under non-uniform sampling, the substrate-robustness results, the dimension–fidelity trend, temporal stability, the hyperbolic-reframe rejection, and the "honest negatives." A reviewer cannot verify any of this. Please either (a) provide a citable, accessible reference (e.g., an arXiv identifier) for the companion **and** restrict this paper's abstract/introduction to claims supported within this submission, or (b) fold the minimum necessary companion material into this paper. As written, the abstract advertises results the manuscript does not contain (see comment 3).

2. **The supplement is wired to a main article that no longer exists.** Five referenced labels are defined nowhere in either file and will render as "??": `sec:density` (supplement lines 45, 67), `sec:law` (990, 996), `sec:univ` (1026, 1066), `sec:stability` (1072), `sec:continuum` (1106). Three of the supplement's sections open with "This section supports the summary in §?? of the main article," but no such summaries exist — the main text instead says this material *is* Paper I.b. Two of these broken references are not cosmetic: the proof setting of Theorem 3.10 (supplement lines 41–46 and 67) invokes "the α = 1 normalization of §`sec:density`" to define the operator against which the non-uniform-sampling version of the theorem runs. That normalization is defined nowhere in the submission, so the non-uniform-sampling reading of the central theorem currently rests on an undefined construction. Please decide on a single home for the extended empirical program (supplement *or* companion — housing full analyses in both invites duplicate-publication concerns), restore or remove the "summary" sections, and define the α = 1 construction wherever the proof needs it.

3. **Empirical scope of the main paper is narrower than its claims.** The abstract claims the angular representation "is substrate-robust — across random-geometric lattices, manifold samples, and hypergraph-rewriting graphs" (lines 44–46), and the introduction claims it "preserved geodesics on every tested graph family" (lines 99–101). But the paper's own protocol table (Table 2, lines 891–905) covers only the 2-torus and 3-torus, and the Methods section (lines 937–952) carefully describes swiss-roll, king-lattice, and Lorentzian causal-set substrates **that never appear in any result in this paper** — orphaned text supporting experiments that were moved out. Either restore a compact substrate-robustness experiment to the main results, or rewrite the abstract, introduction, discussion, and Methods to match what the paper actually shows.

4. **All geodesic-preservation evidence is on homogeneous spaces — the trivially favorable case.** On flat tori the *continuum* radius is exactly constant (the paper proves this, Cor. 3.11 and the remark after Cor. 3.12), so the angle discards nothing at the continuum level by construction; hypothesis (H2)/(A2) is non-trivial precisely on non-homogeneous manifolds, where a radial fold could in principle occur — the paper itself flags "whether the eigenmap image ever develops a radial fold on some manifold" as the sharp remaining question (lines 853–857). Yet no non-homogeneous manifold is tested in the main paper. The swiss roll is already fully described in Methods; reporting angle/magnitude/full/random ρ on it (and ideally the measured (A2) margin, as done for the torus at lines 441–444) would materially strengthen the empirical case at low cost.

5. **Two strong supplement results deserve main-text visibility.** (a) Prop. SM2.6 ("rank stability on flat tori") proves that for d ≤ 3 the commute angular *ranking* converges to a fixed Green-kernel ranking — exact rank 1 on S¹. This is the strongest theoretical support in the entire submission for the headline "flat in m" rank claim, yet it is absent from the main text and from Table 1. (b) The filter phase diagram (Prop. SM2.5) yields a falsifiable structural prediction — the commute filter sits at the critical exponent s = d/4 for rank-kernel stability, with boundary dimension d = 4 — which the supplement itself calls "a sharper structural prediction than the five-family linear trend." Both warrant a sentence and a Table 1 row in the main article.

6. **Precision of Conjecture 3.5.** As stated, the quantifier structure of "Spearman ≥ ρ₀ > 0 … uniformly over the admissible mode schedule" is ambiguous: is ρ₀ universal over the density class, per-manifold, or per-family? Please state the rank clause as an explicit limit statement over admissible sequences (n, m(n)) with the dependence of ρ₀ spelled out. Separately, the paragraph after the conjecture invokes "the ω-screen" (line 327), a term defined nowhere in this submission (it appears to be the small-world screen of the missing §`sec:univ`).

7. **Lemma SM1.3, Step (a): the Gram-matrix justification reads circular as written.** The parenthetical "(each off-diagonal pairs two sup-norm-εₙ perturbations of orthonormal continuum eigenfunctions)" (supplement lines 199–206) does not quite parse: the limit partners φ̃ₖ within a multiplet are exact continuum eigenfunctions, not perturbations of a fixed orthonormal basis, and their near-orthonormality should be routed through the discrete vectors — orthonormality of the vₖ in the empirical inner product, sup-norm closeness, and concentration of empirical inner products on L²(ρ). The conclusion is right and the fix is a rewrite of one sentence, but since this step anchors the conditional theorem, please also double-check the quoted statement numbers of Calder–García Trillos–Lewicka (Theorem 2.6 / Remark 2.8) against the published version.

8. **Prop. SM2.6 hypothesis is slightly too weak.** L² convergence G_Λ → G plus atomlessness of the law of G(X−Y) does not by itself give G_Λ(X−Y) → G(X−Y) in probability: if the pair law of Z = X−Y charged a Lebesgue-null set, L² convergence would say nothing there. Absolute continuity of the pair law (uniform sampling suffices) should be added to the hypotheses; the proof then goes through exactly as written.

9. **Statistical hygiene for headline numbers.** Several headline entries are single graph draws (the † rows of Table 2), including the entire ablation table of §6 and the empirical distortion κ ≈ 2.6 of §3.2 — the latter being the paper's only *metric* (as opposed to rank) evidence. The paper is admirably explicit about this, but given how much narrative weight the ablation separation (0.06 vs 0.89 at m = 20) and κ carry, please report draw-to-draw spread (≥ 5 draws) for both, as already done for the m = 10 core number (0.914 ± 0.009).

---

## Moderate comments

10. **Eigenvector normalization is inconsistent between documents.** The proof of Prop. 3.4 uses VVᵀ = I, i.e. unit normalization Σᵢ vₖ(i)² = 1 (main lines 275–278), while the supplement's setting fixes the empirical normalization (1/n)Σᵢ vₖ(i)² = 1 (supplement line 56). The angular statements are scale-invariant, but the numerical bounds 1/λ_{m+1} and 2/λ_{m+1} in Prop. 3.4 are not — under the empirical normalization they scale by n. State the convention where Ψ is defined in §3 and note the invariance.

11. **Cor. 3.12 (sphere): normalization constant.** With volume normalized to 1 (the supplement's stated convention), the ℓ = 1 orthonormal eigenfunctions are √(d+1)·xᵢ, so F = √((d+1)/d)·x, not x/√d. Harmless — N = x either way — but as written it contradicts the stated convention; either fix the constant or drop it.

12. **"Unconditional" in Cor. 3.11.** The corollary is unconditional *among (H1)–(H3)*; the sampling model and the spectral-consistency machinery of Thm. 3.10 are still assumed. The text mostly frames it this way, but "needs no hypothesis at all" (line 861 and supplement line 435) overshoots; a qualifier would prevent misreading. Relatedly, it may be worth noting that the theorem-certified distortion on the torus is B/A = 8, so the measured κ ≈ 2.6 sits between the sharp continuum π/2 and the certified 8 — a useful calibration of how loose the transfer constants are.

13. **Revision-round artifacts left in the manuscript.** (a) "(a referee's point)" appears in the Appendix A text (main line 1100) — this belongs in the response letter, not the paper. (b) Bare groups `{% entire theorem + remark are new in this revision` … `}%` (lines 218/309) and `{% entire appendix new in this revision` … `}% end teal` (lines 1096/1176) are leftovers of stripped revision coloring; `xcolor` is loaded but unused in both files, and the supplement defines `\rev` as the identity (line 8). Please clean these out.

14. **Bibliography: four entries are never cited** — `alamgir2011`, `hein2007`, `lipman2010biharmonic`, `vonluxburg2010getting`. Either drop them or cite them where they naturally belong: the sentence introducing the amplified commute distance as "the natural competitor" (lines 156–158) is the obvious home for von Luxburg–Radl–Hein 2010, and p-resistances (Alamgir–von Luxburg) and biharmonic distance (Lipman–Rustamov–Funkhouser) are the other standard corrected distances a reader will expect to see acknowledged there.

15. **Cross-document referencing mechanics.** The two files reference each other via `\externaldocument[][nocite]` (main line 13, supplement line 15); neither loads `xr`/`xr-hyper` explicitly, so this relies on the SIAM class providing it. Since I could not compile, please verify in the submission build that (i) both documents resolve all cross-references after the required multi-pass, and (ii) the wording is consistent — the main text says "Appendix \ref{app:proof}" for what the supplement labels and announces as "Section SM1"; pick one ("supplement section SM1" is SIAM's convention).

---

## Minor comments

16. Supplement lines 376–379: the ratio range [√2/π, 1/√2] is described as "of width π/2" — the *ratio* of the extremes is π/2 (≈ 1.571); the width of the interval is ≈ 0.257. Say "ratio."

17. The abstract is a single dense paragraph carrying three author-name citations and heavy technical vocabulary ("tangential noncollapse," "sup-norm eigenvector rates"); the Contributions paragraph (lines 103–120) is one sentence with five levels of nesting. Both would benefit from splitting. More broadly, the manuscript's em-dash-and-parenthetical density is high even by the standards of a technical audience; a pass for sentence length would help referees and readers alike.

18. Symbol overloading: A is the anchors count (Table 2), the lower transfer constant (Thm. 3.10), the measured lower stretch (line 442), and the kernel A_R(z) (supplement). d is well-policed (the paper explicitly reserves it); A is not.

19. The gap-closed cutoffs on 𝕋² are correctly listed as 4, 8, 12, 20, … (line 663) — verified against the lattice multiplicities — and the m = 10 caveat is a good catch; consider also reporting the core table (§5.1) at a gap-closed m (e.g., 12 or 20) so the headline numbers match the theory's admissible schedule.

20. Reference cosmetics: `ng2002` is labeled "NeurIPS" (it was NIPS 14, 2001/2002); `nickel2017` likewise predates the rename. The Wolfram citation is a blog post — acceptable as substrate provenance given Gorard's *Complex Systems* article is cited alongside, but the physics-interpretation paragraph in §7 could be trimmed further without loss; the paper's repeated firewalling of that material is appreciated and mostly effective.

21. Figure files are named `fig1_core.pdf`, `fig3_control.pdf` (main) and `fig2_scaling.pdf` (supplement) — a filename trace of the restructuring; harmless, but renumber to avoid confusion in production. (Figures were not provided, so their content was not reviewed.)

22. The pre-registration, dated amendments (including the candid snapshot-inflation correction in SM4), and per-script committed result JSONs are excellent reproducibility practice and should be retained prominently.

---

## Questions for the authors

A. Is "Paper I.b" publicly available, and what is its publication status? If it is under review elsewhere, please clarify the division of contributions to rule out overlap between it and supplement sections SM3–SM5.

B. For Conjecture 3.5: is ρ₀ intended to be uniform over the stated density class at fixed (M, m-schedule), or per-manifold? And is the empirical ρ₀ ≈ 0.9 meant as the conjectured constant or an observed instance?

C. Has (A2) — the two-point tangential-noncollapse margin — been measured on any non-homogeneous substrate, as it was for the torus (0/9200 degenerate local pairs, local angular constant 2.2)? Even a negative or marginal result here would sharpen the paper's own "radial fold" open question.
