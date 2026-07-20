# Distillation plan: 23 pp → 20 pp (main article)

**Principle.** No theorem, lemma, corollary, conjecture, proof, table of results, or
empirical number is cut. Savings come from (1) one figure merge and (2) collapsing the
four status-narration passages whose function Table 1.1 now performs. Every unique
number retained exactly once; coverage noted per cut.

**Budget** (review format ≈ 44 lines/page):

| # | Target | Now | After | Saves |
|---|--------|-----|-------|-------|
| 1 | Figs 5.1+5.2 → one two-panel figure | 2 floats | 1 float | ≈ 0.8 pp |
| 2 | "Metric evidence on the torus" | 17 ln | 11 | 6 ln |
| 3 | "Uniform-in-m frontier" block | 35 ln | 15 | 20 ln |
| 4 | Post-Lemma-norm discussion | 18 ln | 12 | 6 ln |
| 5 | Post-S¹-collapse mechanism ¶ | 24 ln | 11 | 13 ln |
| 6 | "What remains is delimited" | 41 ln | 15 | 26 ln |
| 7 | "Status." paragraph | 30 ln | 18 | 12 ln |
| 8 | "Status of the radial claim" ¶ | 8 ln | 0 | 8 ln |
| 9 | §5.10 qualification ¶ | 24 ln | 10 | 14 ln |
| 10 | §7 open-problems sentence | 13 ln | 6 | 7 ln |
| 11 | §5.3 rewiring-sweep block | 11 ln | 6 | 5 ln |
| | **Total** | | | **≈ 117 ln + 0.8 pp ≈ 3.3–3.5 pp** |

Lands at ≈ 19.5–20 pp with margin for float drift.

---

## Cut 1 — Merge Figures 5.1 and 5.2 (biggest single win)

Both are m-sweeps on the same torus with identical axes; they are a natural pair.
Replace both figure environments with:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{fig1_core.pdf}\hfill
\includegraphics[width=0.48\linewidth]{fig3_control.pdf}
\caption{Left: Spearman correlation with graph geodesics on a 2-torus
($n=2000$) as a function of mode-count $m$: angle carries the geometry,
magnitude is throwaway, and the full commute embedding decays with $m$.
Right: geometry lives in the low-eigenvalue subspace---the lowest $m$ modes
preserve geodesics while an equal-size random-mode basis does not (random
curve: mean over $25$ draws; shaded band: full min--max envelope---even the
best-case random draw stays far below the low-mode curve).}
\label{fig:core}
\end{figure}
```

Then retarget the two in-text pointers: `(Fig.~\ref{fig:control})` →
`(Fig.~\ref{fig:core}, right)` in §5.1, and delete the `\label{fig:control}` figure.
**Font check:** the panel titles/legends are baked into the PDFs at full-width sizing;
at 0.48\linewidth they render ≈7 pt. Preferable: regenerate both panels from the repo
scripts with `figsize` halved and font sizes up two points (10 min); acceptable
fallback: ship as-is — review-format tolerance covers it, and production will ask for
final art anyway.

---

## Cut 2 — "Metric evidence on the torus" (§3, before the frontier block)

Replace the whole paragraph with:

```latex
\paragraph{Metric evidence on the torus.} On the 2-torus we compare embedded
distance to graph geodesic directly. After a single global scale the angular
map has a robust empirical bi-Lipschitz constant $\kappa\approx2.6$ (ratio of
the $97.5$th to $2.5$th percentiles of $d_{\mathrm{emb}}/d_{\mathrm{geo}}$
over $19{,}900$ pairs), against $3.1$ for the full commute embedding, $8.3$
for an equal-size random-mode basis, and $125$ for the magnitude; its leading
four coordinates Procrustes-match the canonical flat-torus embedding in
$\mathbb{R}^{4}$ at disparity $0.07$ (random modes: $0.995$). This is metric
distortion, not rank correlation---one substrate, but a well-understood one,
and one datapoint at the tested budget: the correct uniform metric target is
scale-dependent for the commute filter (Proposition~\ref{prop:weyl}).
```

**Coverage:** the deleted near-antipodal parenthetical survives in Cor. 3.12's
chord/arc discussion. **Required micro-edit there:** Cor. 3.12 currently says
"the near-antipodal dip of the metric-evidence paragraph" — change to
"the source of the near-antipodal dip in the measured $\kappa$" so the pointer
doesn't dangle.

---

## Cut 3 — "The uniform-in-$m$ frontier, precisely located"

Keep the **(H1)-uniform** display and **Conjecture `conj:sigmamin` verbatim**
(both are cross-referenced). Replace the framing, the "what is known" paragraph,
the two-bullet itemize, and the closing "none of the paper's proved results"
paragraph with:

```latex
\paragraph{The uniform-in-$m$ frontier.} The metric form of
Conjecture~\ref{conj:angle} rests on one named hypothesis, which we isolate
rather than assume away.

\smallskip\noindent\textbf{(H1)-uniform.} \emph{At mode-counts $m$ taken at a
spectral gap, the truncated eigenmap $F=(\varphi_k/\sqrt{\mu_k})_{k\le m}$ is
$L_0$-bi-Lipschitz onto its image on geodesic $r_0$-balls, with $L_0$ and $r_0$
independent of $m$.}

\noindent What is known is weaker: \cite{berard1994} and the quantitative
truncated-eigenmap embeddings of \cite{bates2014,portegies2016} give, for each
fixed $m$ past an immersion threshold, \emph{some} constant $L_0(m)$, and as
$m$ grows the added eigenfunctions oscillate on wavelength $\mu_m^{-1/2}$, so
holding $L_0$ fixed asks that no mode ever create a near-fold---the
degeneration a symmetry or thin neck can force. Two endpoints bound the
question: on the flat torus (H1)-uniform holds globally and unconditionally
(Corollary~\ref{cor:torus}, through the Clifford embedding), and it is
\emph{filter-specific}---a theorem for the heat weighting
(Theorem~\ref{thm:heat}), false at fixed scale for the commute weighting
(Proposition~\ref{prop:weyl}). The empirical flatness in $m$
(\S\ref{sec:verify}) supports the rank form; the metric form reduces to one
clean question in spectral geometry:

[ conj:sigmamin — UNCHANGED ]

\noindent A proof---or a counterexample forced by symmetry---is a
self-contained problem we do not attempt; none of the paper's proved results
(Table~\ref{tab:claims}) depends on its resolution.
```

**Bonus:** this adds the first in-text pointer to Table 1.1, fixing the
unreferenced-table item.

---

## Cut 4 — Discussion after Lemma `lem:norm`

Replace the 18-line paragraph with (all numbers preserved, single occurrence):

```latex
The condition $\Lambda<1/L$ is a quantitative radial degeneracy: the radius
must vary slower than the embedding moves, and as $\Lambda\to0$ (the limit of
Corollary~\ref{thm:radial}) normalization becomes a pure scaling with lower
constant $1/(L\rho^+)$. On the 2-torus the identity is exact to $10^{-15}$,
but the margin is razor-thin ($\Lambda=0.0918$ against lower stretch
$A=0.0920$), so the lemma's worst-case constant is near-vacuous there---while
the pointwise tangential component never degenerates ($0/9200$ local pairs
radially degenerate; measured local angular constant $2.2$).
Proposition~\ref{prop:angdist} explains why, and
Theorem~\ref{thm:transfer} makes the resulting two-point tangential
noncollapse (A2) the hypothesis itself, with $\Lambda<1/L$ only one
sufficient route to it---so the thin radius margin costs the theory nothing.
```

---

## Cut 5 — Mechanism paragraph after Theorem `thm:s1collapse`

Replace with:

```latex
The mechanism in one line: the $1/\mu$ commute weighting is a low-pass filter
on the \emph{correction} kernel that carries the ranks, while equal weighting
hands the correction to the oscillatory top of the spectral window---which is
why the ablation separates the filters at the rank level ($0.06$ vs $0.89$ at
$m{=}20$, \S\ref{sec:ablation}) even though their crude metric orders
coincide. For graph transfer at growing $m=m(n)$, the calibration of
Lemma~\ref{lem:transfer} extends whenever
$\mathcal E_n(\mu_m)\,\gamma_m^{-1}\to0$ ($\gamma_m$ the gap isolating the
retained block; $\mathcal E_n$ the graph-consistency error, polynomially
growing in $\mu_m$), and a $C^{0,1}$ eigenvector theory
\cite{calderlewicka2022} is the route to shrink the resolution cutoff toward
the connectivity scale (supplement, \S SM2).
```

**Coverage:** "the theory now covers both empirical laws" → Table 1.1 rows;
the s1collapse restatement → the theorem statement immediately above.

---

## Cut 6 — "What remains is delimited"

Replace the 41-line paragraph with:

```latex
What remains is delimited. \textbf{(H1)} is subtle: Jones--Maggioni--Schul
\cite{jones2010} guarantee bi-Lipschitz charts only from a well-chosen
$d$-subset of eigenfunctions per ball, and low-spectrum multiplicity makes a
truncation inside a multiplet basis-dependent---so $m$ is taken at a spectral
gap (the $m{=}4$ of Corollary~\ref{cor:torus}) or the map phrased through
spectral projectors. \textbf{(H2)} is, by Proposition~\ref{prop:angdist},
sharper than radius flatness: the eigenmap's velocity must be nowhere purely
radial, with (A2) its two-point form and the hypothesis to verify in
practice; whether the eigenmap image ever develops a radial fold on some
manifold is the sharp remaining question. \textbf{(H3)} is genuinely extra:
injectivity of $F$ \cite{portegies2016} does not exclude a radial alignment
$F(x)=tF(y)$, so the global statement is carried by (H3)
(Corollary~\ref{cor:global})---while on the torus (H1) holds globally and
Corollary~\ref{cor:torus} needs no hypothesis at all.
```

**Coverage:** the filter-split recap → Cut 3's endpoints + Table 1.1; the
"local constant 2.2" echo → kept in Cut 4; the κ/Procrustes anticipation →
kept in Cut 2; the closing "(H1)–(H3) + distance-ratio" reduction → Cut 7.

---

## Cut 7 — "Status." paragraph

Replace with:

```latex
\noindent\emph{Status.} Table~\ref{tab:claims} records what is proved,
conditional, empirical, and open; at fixed $m$ the rank clause reduces to
(H1)--(H3) plus a distance-ratio condition on the substrate
(Corollary~\ref{cor:global}, Proposition~\ref{prop:rank}), and uniformity in
$m$ is settled in the continuum by filter. The honest gap:
Corollary~\ref{thm:radial} explains why the full unnormalized radius
\emph{dies} ($d\ge3$), not why the angle \emph{lives}---angular fidelity
persists ($\rho\approx0.88$--$0.90$) even on quasi-1D graphs where the
corollary does not apply, so the angular half is carried by the
eigenmap-embedding mechanism. That mechanism is analyzable: if the low
eigenvectors converge to Laplace--Beltrami eigenfunctions
\cite{belkin2007,garciatrillos2018} and eigenfunction coordinates furnish
locally bi-Lipschitz charts \cite{jones2010}, geodesic fidelity should follow
from the chart geometry---the natural route to
Conjecture~\ref{conj:angle}, stated sharply as
Conjecture~\ref{conj:sigmamin}. Together these are the
\textbf{Keep-the-Angle Principle}: keep $\hat u_i$, discard $r_i$---and,
under strongly non-uniform sampling, keep the $\alpha{=}1$
density-normalized angle (\S\ref{sec:density}).
```

---

## Cut 8 — Delete "Status of the radial claim" paragraph entirely

**Coverage:** (i) proved → Table 1.1 row 1 + the corollary itself; (ii) the
numbers ρ(rᵢ,1/√kᵢ) ∈ [0.79, 0.87] vs 0.98 → already in the Remark after the
corollary and in §5.5; (iii) borderline d=2 / no-degeneracy d=1 → the
"Dimension gating" paragraph, which stays.

---

## Cut 9 — §5.10 qualification paragraph (detail now in SM5)

Replace with:

```latex
One qualification deepens the reading: the dissociation is stated at a
\emph{fixed} mode budget ($m=10$ while $N$ grows $5\times$). Rerunning with
$m$ spectral-gap-matched inverts it---the RGG now falls ($0.856\to0.775$) as
the added high modes inject sub-geodesic detail, while the emergent substrate
stops decaying---so no single budget is optimal for both, and ``clean
geometry'' is a relation between mode-budget resolution and substrate scale
(\S\ref{sec:verify}), the motivation for a dimension-adaptive basis
(\S\ref{sec:disc}). Modest evidence---one emergent substrate,
$N\le10^{4}$---so we state the direction, not a law; full curves, gating,
and the pre-registration are in the supplement (\S SM5).
```

---

## Cut 10 — §7 open-problems sentence

Replace the long parenthetical sentence with:

```latex
Open problems: the truncated-Jacobian bound
(Conjecture~\ref{conj:sigmamin}, which closes the metric form); extending
Corollary~\ref{thm:radial} to the truncated, normalized radius in actual
use; verifying (H1)--(H3) beyond the flat torus; graph transfer at growing
spectral windows (supplement, \S SM2); synthesizing clean manifolds above
$d\approx2$; and a dimension-adaptive angular basis that tunes mode-count
and diffusion time against the fidelity decay.
```

---

## Cut 11 — §5.3 rewiring-sweep block

Replace the sweep sentences ("A king-lattice rewiring sweep both sharpens…"
through "…dimension-trend outlier below") with:

```latex
A king-lattice rewiring sweep bounds the screen honestly: angle-$\rho$
collapses almost at once ($0.81\!\to\!0.47$ by $1\%$ rewiring) while $\omega$
does not cross zero until ${\approx}5\%$, so $\omega<0$ is a
\emph{conservative} flag and, in the $1$--$3\%$ transition band, angular
fidelity is itself the finer detector of geometric corruption---the
mechanism behind the dimension-trend outlier (\S\ref{sec:law}).
```

---

## Net-zero fix to fold in while editing §5.3 (the standing causal-set item)

Table row: `causal set (Lorentzian) & 3.9 & 0.57 & 0.01 & 0.55` →
`causal set (Lorentzian) & $X$--$Y^{*}$ & 0.57 & 0.01 & 0.55` with X–Y the
ball/spectral/effective-rank range from the ledger (spread 7.87), sharing the
small-world asterisk convention. Sentence: "the metric degrades where geometry
is Lorentzian (ρ = 0.57, a degradation not a failure) or destroyed" →
"the metric fails where the substrate fails the manifold criterion---the
Lorentzian causal set ($\rho=0.57$) and the rewired lattice ($0.45$)---the
predicted failures, not exceptions."

## Reminder (orthogonal to length)

The 9 FILLs + d_G, the †/spread markers, the two-sentence SM1 Lemma repair,
and the four seam theorem numbers are unchanged by this plan and still gate
submission.
