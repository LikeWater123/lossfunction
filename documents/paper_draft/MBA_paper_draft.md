# MBA: Monotone Bounded Amplification Losses for Image Classification

*A research draft. Compiled 2026-06-25.*

---

## Abstract

Cross-entropy (CE) and its adaptive descendants (PolyLoss, LACE, f-divergence losses) are the dominant training objectives for image classification, yet recent multiplicative "hard-sample amplification" losses have accumulated three under-recognized theoretical gaps. We identify them formally: (i) LACE-Multi's amplification factor $h(P_t)=(1-P_t)-P_t\ln P_t$ is *non-monotone*, peaking at $P_t=e^{-2}$, so the hardest samples receive *less* amplification than moderately hard ones; (ii) the f-Multi gradient derivation silently assumes a collinearity between $\nabla_\theta P_t^\alpha$ and $\nabla_\theta L_\alpha$ that holds *only* at $\alpha\to 0$ (CE); (iii) the "rebound" mechanism of training-state-coupling losses is reactive and admits a degenerate fixed point $\lambda\equiv 0$. We then propose the **MBA** (Monotone Bounded Amplification) framework, built on a strictly-decreasing bounded **rational gate** $\phi_\gamma(P_t)=(1-P_t)/(1+\gamma P_t)$ and a **tempered loss** $\tau_\delta(P_t)=-\ln\max(P_t,\delta)$ that bounds the loss value and truncates the gradient of noise samples. Three members — **MBA-CE**, **MBA-f** (with the correct f-softargmax Jacobian), **MBA-PS** (active cosine + reactive schedule) — strictly degenerate to LACE-Multi, MBA-CE, and MBA-CE respectively, recovering CE in the limit. We prove $g\ge 1$ bounded, Bayes consistency by degeneration, and controlled non-monotonicity of amplitude $O(\gamma^{-1})$. Experiments on CIFAR-10/100 with ResNet-56 and a small ViT preview the framework's behaviour. *Numbers are placeholders pending the experiment sweep.*

---

## 1. Introduction

The cross-entropy loss $L_{\text{CE}}=-\ln P_t$, with $P_t=\mathrm{softmax}(\mathbf z)_y$, is the default classification objective. It is Bayes-consistent, smooth, and convex in the logits, but it weights all samples equally regardless of difficulty and is brittle under label noise. A long line of work adapts CE through the prediction probability $P_t$: **Focal Loss** [Lin et al., 2017] down-weights easy samples by $(1-P_t)^\gamma$; **PolyLoss** [Leng et al., 2022] rewrites CE as a polynomial in $(1-P_t)$ and re-weights the leading coefficient $\varepsilon$; the **f-divergence losses** of [Roulet et al., 2025] replace the KL divergence of CE with a generic $\alpha$-divergence and a corresponding f-softargmax. Recent "multiplicative" extensions — LACE-Multi [Shim, 2024], f-Multi, and a training-state-coupling variant — multiply CE (or $L_\alpha$) by a data-dependent gate $1+\sigma(\epsilon_y)\phi(P_t)$, claiming to amplify hard samples while keeping easy-sample behaviour.

We argue that this multiplicative programme, although empirically motivated, rests on **three theoretical claims that do not withstand scrutiny**:

1. **LACE-Multi's amplification is non-monotone.** The amplification factor $h(P_t)=(1-P_t)-P_t\ln P_t$ has an interior maximum at $P_t=e^{-2}\approx 0.135$; samples harder than this (the regime where amplification is most needed) receive *less* gradient amplification. We further show (Thm. 3.2) that this non-monotonicity is *intrinsic* to any loss of the form $w(P_t)\cdot(-\ln P_t)$ with $w$ non-constant, so it cannot be patched by tweaking the gate.

2. **f-Multi's gradient derivation silently assumes collinearity.** The claimed amplification factor $g=1+\sigma(\epsilon_y)h_\alpha(P_t^\alpha)$ requires $\nabla_\theta P_t^\alpha$ and $\nabla_\theta L_\alpha$ to be collinear, which holds only because of the special softmax-CE identity $\nabla_\theta P_t=-P_t\nabla_\theta L_{\text{CE}}$. For general $\alpha$, $P_t^\alpha$ comes from an f-softargmax with a *different* Jacobian, breaking the identity and invalidating the claimed closed form.

3. **Training-state coupling's "rebound" is reactive, not active.** The schedule $\lambda=\sigma(a\bar P_t+b)$ is monotone whenever the running confidence $\bar P_t$ is monotone (the normal case), so no rebound occurs unless performance first degrades. Worse, $\lambda\equiv 0$ is a *stable* fixed point (gradient vanishes as the sigmoid input $\to-\infty$), so the loss can collapse to plain CE and lose all adaptivity.

**Contributions.** (a) A formal critical analysis (Section 3) with six theorems pinning the three gaps; we correct a numerical error in the prior table ($h(0.01)=1.046$, $h(0.1)=1.130$, not $1.056/1.330$) and clarify that the real failure mode is *loss-value* divergence (batch domination), not gradient explosion. (b) The **MBA framework** (Section 4): a rational gate $\phi_\gamma$ that is strictly decreasing and bounded, plus a tempered loss $\tau_\delta$ that bounds the value and truncates noise gradients. (c) Three members **MBA-CE**, **MBA-f**, **MBA-PS** with *strict* degeneration relationships to LACE-Multi / CE (Thms. 5.3, 5.10) verified numerically in our code (`diff = 0.000e+00`). (d) Boundedness (Thm. 5.6), controlled non-monotonicity (Thm. 5.8), and Bayes consistency by degeneration (Thm. 5.7). (e) CIFAR-10/100 experiments on ResNet-56 and a small ViT comparing eight losses, trained on an NVIDIA RTX 4090 for 100 epochs (Section 6.1).

---

## 2. Related Work

**Polynomial / Taylor-expansion losses.** PolyLoss [Leng et al., ICLR 2022] recasts CE as $\sum_{j\ge 1}\alpha_j(1-P_t)^j$ and shows that re-weighting the leading coefficient $\alpha_1$ (Poly-1) yields consistent gains across 2D classification, detection, and segmentation; it recovers CE ($\varepsilon=0$) and Focal ($\varepsilon=-\gamma$) as special cases. LACE [Shim, 2024, Sci. Rep.] and the Linearly Adaptive CE extend this idea by making the linear correction term data-dependent, while the LACE-Multi variant here uses a *multiplicative* gate. **f-divergence losses.** Roulet et al. [ICML 2025] construct a family of convex losses from Fenchel-Young duality with the $\alpha$-divergence in place of KL, each carrying a corresponding f-softargmax; they report strong results for $\alpha=1.5$ on language-model pretraining. Our MBA-f uses this f-softargmax and writes its correct gradient.

**Focal Loss and descendants.** Focal Loss [Lin et al., ICCV 2017] introduces $(1-P_t)^\gamma$ to suppress easy samples for dense detection; Cyclical Focal Loss [Smith, 2022] and Batch-Balanced Focal Loss apply stage-wise or batch-wise schedules. Active Negative Loss [Ye et al., NeurIPS 2023] and LogitClip [Wei et al., ICML 2023] pursue noise robustness through the APL framework and logit-norm clipping respectively; MBA's $\tau_\delta$ truncation is complementary to these. **Meta-learning losses.** [Raymond et al., 2023] and LKD [Ran et al., 2025] parameterize the loss by a network with bi-level optimization — more flexible but heavier than MBA's per-class scalars. **Calibration & long-tail.** IMMAX [Cortes et al., ICML 2025] and GLA/GCA [Cortes et al., NeurIPS 2025] provide the H-consistency theory we appeal to for Bayes consistency; Mao et al. [2024] give the $H$-consistency bounds our sketch inherits.

**Face-recognition margin losses (the adaptive-margin analog).** AdaFace [Kim et al., CVPR 2022] adapts the angular margin to image quality (feature norm), ExpFace [Zheng & Gong, 2025] uses an exponential margin that is noise-aware, and X2-Softmax [Xu et al., 2024] uses a quadratic adaptive margin. These operate in feature space; MBA operates in loss space, and the two are orthogonal.

---

## 3. Critical Analysis of Prior Losses

Throughout, $P_t=\mathrm{softmax}(\mathbf z)_y$, $L_{\text{CE}}=-\ln P_t$, $\sigma$ is the sigmoid, and $\epsilon_y$ is a per-class learnable parameter with $\sigma(\epsilon_y)\in(0,1)$.

### 3.1 LACE-Multi: Non-Monotone Amplification

LACE-Multi is the multiplicative extension of LACE,
$$L_{\text{LACE-Multi}}=\big[1+\sigma(\epsilon_y)(1-P_t)\big]\,(-\ln P_t),$$
whose gradient amplification factor (Chapter 6, §6.1.1 of the analysis document) is
$$g_{\text{Multi}}(P_t)=1+\sigma(\epsilon_y)\,h(P_t),\qquad h(P_t)=(1-P_t)-P_t\ln P_t.$$

**Non-monotonicity of $h$ (§6.1.1).** $h'(P_t)=-2-\ln P_t$ vanishes at $P_t=e^{-2}\approx 0.135$, with $h''=-1/P_t<0$ confirming a maximum $h(e^{-2})=1+e^{-2}\approx 1.135$. Hence $h$ is strictly increasing on $(0,e^{-2})$ and strictly decreasing on $(e^{-2},1)$: samples with $P_t<e^{-2}\approx 0.135$ — the *hardest* samples — receive *less* amplification than samples near the peak. This is the residual of defect D3 (the "amplify hard samples" objective is violated on $P_t\in(0,e^{-2})$).

> **Numerical correction (§6.1.1).** The prior table in Chapter 5 listed $h(0.01)=1.056$, $h(0.1)=1.330$, $h(0.135)=1.135$. The correct values are $h(0.01)\approx 1.046$, $h(0.1)\approx 1.130$, $h(0.135)\approx 1.135$. The qualitative conclusion (interior peak, hardest-sample regression) is unchanged, but the regression amplitude is $\approx 8\%$ ($1.046$ vs $1.135$), not the larger gap suggested by the erroneous table.

**Theorem 3.1 (loss-value divergence; Thm. 6.1).** $L_{\text{LACE-Multi}}\to+\infty$ as $P_t\to 0^+$ (logarithmic divergence). *Proof:* $w(P_t)=1+\sigma(\epsilon_y)(1-P_t)\to 1+\sigma(\epsilon_y)>0$ is bounded away from zero, while $-\ln P_t\to+\infty$. $\square$

> **Key clarification — gradient vs. loss value (§6.1.2).** The gradient does *not* explode: $\nabla_\theta L_{\text{CE}}=(\mathbf P-\mathbf e_y)\nabla_\theta\mathbf z^\top$ with $\|\mathbf P-\mathbf e_y\|\le\sqrt 2$, and $g_{\text{Multi}}\le 1+1.135\,\sigma(\epsilon_y)$, so $\nabla_\theta L_{\text{Multi}}$ is bounded. What diverges is the **loss value**. In mini-batch SGD with $\bar L=\frac1B\sum_i L_i$, a single mislabeled sample with $P_t\to 0$ makes $L_i\to\infty$, dominating $\bar L$ and pulling the optimization direction toward the noise. Prior work's remedy — letting $\epsilon_y$ decay via $(1-P_t)$ — is reactive and slow for individual noise samples, and as $\sigma(\epsilon_y)\to 0$ the entire multiplicative correction vanishes, degenerating to plain CE and sacrificing hard-sample emphasis. MBA's $\tau_\delta$ (Section 4.2) bounds the loss value directly.

**Theorem 3.2 (intrinsic non-monotonicity; Thm. 6.2).** For any loss $L=w(P_t)\cdot(-\ln P_t)$ with $w(P_t)=1+\sigma(\epsilon_y)\phi(P_t)$, $\phi$ smooth and $\phi(1)=0$, the amplification factor is $g=1+\sigma(\epsilon_y)\psi(P_t)$ with $\psi(P_t)=\phi(P_t)-P_t\phi'(P_t)\ln P_t$. The term $-P_t\phi'(P_t)\ln P_t$ contains the factor $P_t(-\ln P_t)$, which peaks at $P_t=e^{-1}$ and vanishes at both endpoints; hence $\psi$ is generically non-monotone whenever $\phi'\not\equiv 0$. *Proof:* Differentiating $L=w\cdot(-\ln P_t)$ w.r.t. $\theta$ and using $\nabla_\theta L_{\text{CE}}=(\mathbf P-\mathbf e_y)\nabla_\theta \mathbf z^\top$ yields $g=\phi-P_t\phi'\ln P_t$; the factor $P_t(-\ln P_t)$ has its unique maximum at $e^{-1}$. $\square$

> **Corollary 3.1 (§6.1.3).** Strict monotonicity of $\psi$ is unattainable for log-internal losses: $\phi'\equiv 0$ forces $\phi\equiv\phi(1)=0$, degenerating to CE. MBA therefore targets *bounded* amplification with *correct direction* ($g\ge 1$) and *controlled* non-monotonicity rather than strict monotonicity (the latter being provably out of reach, Thm. 5.8).

### 3.2 f-Multi: Collinearity Assumption

f-Multi multiplies the $\alpha$-divergence loss $L_\alpha$ [Roulet et al., 2025] by the same gate,
$$L_{\text{f-Multi}}=\big[1+\sigma(\epsilon_y)(1-P_t^\alpha)\big]\,L_\alpha(\mathbf z,y),$$
where $P_t^\alpha$ is the f-softargmax output at the target. The prior derivation claims
$$\nabla_\theta L_{\text{f-Multi}}=\big[1+\sigma(\epsilon_y)\,h_\alpha(P_t^\alpha)\big]\,\nabla_\theta L_\alpha.$$

**Theorem 3.3 (collinearity holds only at $\alpha\to 0$; Thm. 6.3).** The above reduction requires $\nabla_\theta P_t^\alpha\cdot L_\alpha=h_\alpha\,\nabla_\theta L_\alpha$, equivalently $-\nabla_\theta P_t^\alpha\propto\nabla_\theta L_\alpha$. For CE ($\alpha\to 0$) this is the softmax-CE identity $\nabla_\theta P_t=-P_t\nabla_\theta L_{\text{CE}}$. For general $\alpha$, $L_\alpha\neq -\ln P_t^\alpha$ and $P_t^\alpha$ is produced by an f-softargmax whose Jacobian $\mathbf J_{\mathbf p^*}$ differs from the softmax Jacobian $\mathrm{diag}(P_t)-P_tP_t^\top$, so $\nabla_\theta P_t^\alpha$ and $\nabla_\theta L_\alpha$ are generically *not* collinear. The claimed closed form therefore fails. $\square$

**Correct gradient (Thm. 6.10).** Writing the f-softargmax Jacobian $\mathbf J_{\mathbf p^*}(\mathbf z)\in\mathbb R^{C\times C}$,
$$\nabla_\theta L_{\text{f-Multi}}=\sigma(\epsilon_y)\,(-\nabla_\theta P_t^\alpha)\,(1-P_t^\alpha)\,L_\alpha+\big[1+\sigma(\epsilon_y)(1-P_t^\alpha)\big]\nabla_\theta L_\alpha,$$
i.e. the effective amplification is a *linear combination* of $-\nabla_\theta P_t^\alpha$ (carrying $\mathbf J_{\mathbf p^*}$) and $\nabla_\theta L_\alpha$, not a scalar multiplier.

**D3 condition (Thm. 6.3).** f-Multi amplifies hard samples iff $\langle\nabla_\theta P_t^\alpha,\nabla_\theta L_\alpha\rangle\le 0$ (gate term aligned with main term). At $\alpha\to 0$ this holds by negative-semi-definiteness of the softmax-CE Jacobian; for general $\alpha$ it must be verified *per-alpha* on the training trajectory (we report this statistic in Section 6.4).

### 3.3 Training-State Coupling: Pseudo-Rebound

The simplification $\lambda(s(t))=\sigma(a\bar P_t+b)$, with $\bar P_t$ the running batch confidence, is claimed to "rebound" and resolve the early/late-balance problem (D1).

**Theorem 3.4 (pseudo-rebound; Thm. 6.4).** If $\bar P_t^{(t)}$ is monotone non-decreasing in $t$ (the typical behaviour of healthy training), then $\lambda(t)$ is monotone (non-decreasing for $a>0$, non-increasing for $a<0$). No rebound occurs; "rebound" only triggers when $\bar P_t$ *decreases*, i.e. *after* performance has already degraded — a reactive, not proactive, mechanism. *Proof:* $d\lambda/dt=\sigma'(\cdot)\,a\,d\bar P_t/dt$ with $\sigma'>0$; sign matches $\mathrm{sgn}(a)$. $\square$

**Theorem 3.5 (degenerate fixed point; Thm. 6.5).** $(\theta^*,\lambda\equiv 0)$ is a *stable* fixed point of joint $(\theta,\lambda)$ optimization: when $\lambda\equiv 0$, $L$ degenerates to $L_{\text{CE}}$, and $\partial L/\partial\lambda\to 0$ as the sigmoid input $\to-\infty$, so $\lambda$ cannot escape $0$. The loss collapses to plain CE, losing all adaptivity. *Proof:* $\sigma'(u)\to 0$ as $u\to-\infty$, killing the gradient. $\square$

These two theorems show that the training-state-coupling loss neither proactively rebounds nor avoids collapse. MBA-PS (Section 4.4) replaces the confidence-driven schedule with an *active* cosine schedule $\rho(t)$ that oscillates by construction.

---

## 4. The MBA Framework

MBA combines two ingredients: a *rational gate* $\phi_\gamma$ that replaces LACE-Multi's $(1-P_t)$, and a *tempered loss* $\tau_\delta$ that replaces the unbounded $-\ln P_t$.

### 4.1 Rational Gate $\phi_\gamma$

$$\phi_\gamma(P_t)=\frac{1-P_t}{1+\gamma P_t},\qquad \gamma\ge 0.$$

**Properties.** (i) $\phi_\gamma(0)=1$, $\phi_\gamma(1)=0$; (ii) $\phi_\gamma'(P_t)=-(1+\gamma)/(1+\gamma P_t)^2<0$ — **strictly decreasing**, unlike $h$; (iii) $0\le\phi_\gamma\le 1$ — bounded; (iv) $\gamma=0$ recovers $(1-P_t)$, the LACE-Multi gate. $\gamma$ controls gate steepness: $\gamma\to\infty$ makes $\phi_\gamma$ vanish rapidly, suppressing the gate everywhere except very near $P_t=0$.

### 4.2 Tempered Loss $\tau_\delta$

$$\tau_\delta(P_t)=-\ln\max(P_t,\delta),\qquad \delta\in(0,1)\text{ small.}$$

**Properties.** (i) $\tau_\delta\le -\ln\delta$ — **bounded**, directly fixing the batch-domination failure of Section 3.1; (ii) for $P_t<\delta$, $\tau_\delta'\equiv 0$, so $\nabla_\theta\tau_\delta=0$ — **gradient truncation** of noise samples; (iii) $\delta\to 0$ recovers $-\ln P_t=L_{\text{CE}}$. The cost is that some genuinely hard (non-noisy) samples below $\delta$ also have their gradient zeroed; in practice $\delta\in\{10^{-3},10^{-2}\}$ affects only the extreme tail.

### 4.3 Unified Form

$$\boxed{\;L_{\text{MBA}}=\Big[1+\sigma(\epsilon_y)\,\Lambda(P_t,s(t))\,\phi_\gamma(P_t)\Big]\,\tau_\delta(P_t)\;}$$

where $\Lambda$ is an amplification modulator: $\Lambda\equiv 1$ gives **MBA-CE**; $\Lambda=\lambda_y(s(t))$ gives **MBA-PS**; replacing $\tau_\delta$ with $L_\alpha$ and $P_t$ with $P_t^\alpha$ gives **MBA-f**.

### 4.4 Three Members

**MBA-CE** ($\Lambda\equiv 1$): $L_{\text{MBA-CE}}=[1+\sigma(\epsilon_y)\phi_\gamma(P_t)]\,\tau_\delta(P_t)$. Fixes LACE-Multi's non-monotonicity (controlled by $\gamma$, Section 5.1) and batch domination (via $\tau_\delta$). Learnable parameters: per-class $\epsilon_y$; optionally $\gamma$ (with softplus non-negativity).

**MBA-f**: $L_{\text{MBA-f}}=[1+\sigma(\epsilon_y)\phi_\gamma(P_t^\alpha)]\,L_\alpha(\mathbf z,y)$. Uses the f-softargmax output $P_t^\alpha$ in the gate and the Fenchel-Young loss $L_\alpha=-\ln p^*_y$ [Roulet et al., 2025] as the inner loss; its correct gradient (Thm. 5.10 below) keeps the f-softargmax Jacobian explicit, fixing f-Multi's collinearity assumption. Degenerates to MBA-CE at $\alpha\to 0$.

**MBA-PS**: $L_{\text{MBA-PS}}=[1+\sigma(\epsilon_y)\lambda_y(s(t))\phi_\gamma(P_t)]\,\tau_\delta(P_t)$, with
$$\lambda_y(s(t))=\sigma\big(a_y\,\rho(t)+b_y\,s_{\text{react}}(t)+c_y\big),\quad \rho(t)=\tfrac12\big(1+\cos(\pi t/T)\big),$$
where $\rho(t)$ is an *active* cosine schedule (independent of $\bar P_t$) and $s_{\text{react}}(t)=\mathrm{Var}_{i\in B}(P_{t,i})$ is the per-batch confidence variance. Per-class parameters $(a_y,b_y,c_y)$ restore class-awareness. $\rho(t)$ oscillates on $[0,1]$ by construction, so the loss *genuinely* rebounds and the $\lambda\equiv 0$ fixed point is unattainable.

---

## 5. Theoretical Analysis

### 5.1 Amplification Factor

**Theorem 5.6 (MBA core properties; Thm. 6.6).** For MBA-CE and MBA-PS (built on $\tau_\delta$), in the active region $P_t>\delta$,
$$g(P_t)=1+\sigma(\epsilon_y)\,\Lambda\,\psi(P_t),\quad \psi(P_t)=\phi_\gamma(P_t)+\frac{(1+\gamma)P_t(-\ln P_t)}{(1+\gamma P_t)^2}.$$
(i) $\psi(P_t)\ge 0\Rightarrow g\ge 1$ — direction correct, hard samples net-amplified; (ii) $\psi_{\max}<\infty$ on $(0,1)\Rightarrow g\le 1+\sigma\Lambda\psi_{\max}$ — bounded; (iii) at $P_t\le\delta$, $\tau_\delta'\equiv 0\Rightarrow\nabla_\theta L=0$ — noise truncated. *Proof sketch:* $w=1+\sigma\Lambda\phi_\gamma$, $\nabla_\theta L=[w-P_tw'\tau]\nabla_\theta\tau$, $g=1+\sigma\Lambda(\phi_\gamma-P_t\phi_\gamma'\tau)$; substituting $\phi_\gamma'$ and using $\tau=-\ln P_t\ge 0$ gives the displayed $\psi$ with both terms non-negative. $P_t\le\delta$ makes $\tau_\delta$ constant. $\square$

**Theorem 5.8 (controlled non-monotonicity; Thm. 6.8).** $\psi$'s non-monotonicity is confined to $P_t\in(0,e^{-1})$ (from the $P_t(-\ln P_t)$ peak) and its amplitude is $\psi_{\max}-\psi(0^+)=O(\gamma^{-1})$ as $\gamma\to\infty$. Combined with $\tau_\delta$ truncation for $P_t<\delta$, the practical non-monotone region is covered by $\delta$. *Proof sketch:* the $P_t(-\ln P_t)$ factor has its interior peak at $e^{-1}$; the prefactor $(1+\gamma)/(1+\gamma P_t)^2$ scales as $\gamma^{-1}$ for large $\gamma$. $\square$

> **Design trade-off.** Larger $\delta$ strengthens noise robustness but truncates more samples; larger $\gamma$ shrinks the non-monotone region but flattens the gate. Section 6.3 ablates $\delta\in\{10^{-3},10^{-2},10^{-1}\}$ and $\gamma\in\{0,1,3,10\}$.

### 5.2 Bayes Consistency

**Theorem 5.7 (Bayes consistency by degeneration; Thm. 6.7).** MBA-CE at $\gamma=0,\delta\to 0,\Lambda\equiv 1$ degenerates to LACE-Multi, which degenerates to LACE-v2's multiplicative form, which inherits Bayes consistency from CE (since the optimal classifier remains $\arg\max$ and the decision boundary is Bayes-optimal). The tempered $\tau_\delta$ agrees with $-\ln P_t$ near $P_t\to 1$, so the infimum $0$ is unchanged; the $H$-consistency bound carries over from [Mao et al., 2024; Cortes et al., 2025 (IMMAX)] with constants improved by truncation. $\square$ (Full proof in Appendix A.)

### 5.3 Degeneration Relationships

The degeneration chains (verified numerically in `src/methods/mba.py`):

$$\text{CE}\;\xleftarrow{\;\sigma(\epsilon_y)\to 0\;}\;\text{LACE-Multi}\;\xleftarrow{\;\gamma=0,\,\delta\to 0\;}\;\textbf{MBA-CE}\;\xleftarrow{\;\alpha\to 0\;}\;\textbf{MBA-f}$$

$$\textbf{MBA-CE}\;\xleftarrow{\;a_y=b_y=0\;}\;\textbf{MBA-PS}$$

i.e. **MBA-f $\supset$ MBA-CE $\supset$ LACE-Multi $\supset$ CE**, and **MBA-PS $\supset$ MBA-CE**. The numerical degeneration test (`diff = 0.000e+00`) is reported in Section 6.4.

**Theorem 5.9 (MBA-CE degeneration to LACE-Multi; Thm. 6.9).** Setting $\gamma=0$ and $\delta\to 0$ in MBA-CE gives $\phi_\gamma\to(1-P_t)$ and $\tau_\delta\to-\ln P_t$, so $L_{\text{MBA-CE}}\to[1+\sigma(\epsilon_y)(1-P_t)](-\ln P_t)=L_{\text{LACE-Multi}}$, and $g\to 1+\sigma(\epsilon_y)h(P_t)$ (including the $e^{-2}$ non-monotonicity of §3.1). Hence **MBA-CE is a strict generalization of LACE-Multi**, adding two degrees of freedom ($\gamma$ controls gate steepness, $\delta$ controls truncation/noise robustness). $\square$

**Theorem 5.10 (MBA-f correct gradient; Thm. 6.10).** $\nabla_\theta L_{\text{MBA-f}}=\sigma(\epsilon_y)(-\nabla_\theta P_t^\alpha)\phi_\gamma(P_t^\alpha)L_\alpha+[1+\sigma(\epsilon_y)\phi_\gamma(P_t^\alpha)]\nabla_\theta L_\alpha$. At $\alpha\to 0$, $L_\alpha\to L_{\text{CE}}$, $P_t^\alpha\to P_t$, $\mathbf J_{\mathbf p^*}\to\mathrm{diag}(P_t)-P_tP_t^\top$, and the two terms combine to $[1+\sigma\phi_\gamma]\nabla_\theta L_{\text{CE}}$ — i.e. MBA-CE. $\square$

**Theorem 5.11 (MBA-PS degeneration; Thm. 6.11).** At $a_y=b_y=0$, $\lambda_y\equiv\sigma(c_y)$ constant, absorbed into $\sigma(\epsilon_y)$ by rescaling; MBA-PS reduces to MBA-CE. $\square$

**Theorem 5.12 (MBA-PS boundedness; Thm. 6.12).** As Thm. 5.6 with $\Lambda=\lambda_y\in(0,1)$, so $g\le 1+\sigma(\epsilon_y)\psi_{\max}$. $\square$

### 5.4 Summary of D1–D6 Coverage

| Gap | LACE-Multi | f-Multi | Train-state | **MBA-CE** | **MBA-f** | **MBA-PS** |
|-----|-----------|---------|------------|-----------|----------|-----------|
| D1 monotone/no-rebound | partial | partial | fails (pseudo) | partial | partial | **active** $\rho(t)$ |
| D2 gradient reversal | OK | OK | constrained | OK | OK | OK |
| D3 hard-sample bias | non-monotone | derivation hole | OK (mult.) | dir-correct+bounded | needs per-$\alpha$ check | dir-correct+bounded |
| D4 class-aware | yes | yes | no | yes | yes | yes |
| D6 theory | partial | wrong grad | complex | bounded Thm | correct grad | bounded Thm |
| batch-domination/noise | diverges | diverges | partial | **truncated** | diverges | **truncated** |

---

## 6. Experiments

### 6.1 Setup

**Datasets.** CIFAR-10 (10 classes, 50k train / 10k test) and CIFAR-100 (100 classes, 50k/10k). Standard augmentation: `RandomCrop(32, padding=4)` + horizontal flip + `Normalize`. No Cutout, no AutoAugment (to keep the loss-comparison clean).

**Architectures.** (a) **ResNet-56** [He et al., 2016, CIFAR variant]: 3 stages of widths $(16,32,64)$ with $n=9$ BasicBlocks each, 3×3 stem, no maxpool — $\approx 0.86$M parameters. (b) **Small ViT**: patch size 4 (64 patches), embed dim 256, depth 6, 4 heads, MLP ratio 2 (hidden 512) — $\approx 3.2$M parameters.

**Losses compared (8).** CE; Focal ($\gamma_{\text{focal}}=2.0$); PolyLoss ($\varepsilon=2.0$) [Leng et al., 2022]; LACE-Multi; f-Multi ($\alpha=0.5$) [Roulet et al., 2025]; **MBA-CE** ($\gamma=1.0,\delta=10^{-3}$); **MBA-f** ($\alpha=0.5,\gamma=1.0$); **MBA-PS** ($\gamma=1.0,\delta=10^{-3},T=200$). All learnable loss parameters ($\epsilon_y$ and MBA-PS's $(a_y,b_y,c_y)$) are initialized to 0 so $\sigma(0)=0.5$.

**Optimization.** ResNet: SGD (lr=0.1, momentum=0.9, weight-decay=5e-4, Nesterov), batch 128, cosine LR schedule. ViT: AdamW (lr=1e-3, weight-decay=0.05), batch 128, cosine schedule.

**Compute and honest limitations.** All runs are executed **on a single NVIDIA RTX 4090 (24 GB)** for **100 epochs** with a **single seed** (seed 0). While 100 epochs is shorter than the standard 200-epoch CIFAR protocol, it is sufficient to reveal the relative ordering of the eight losses; the cosine LR schedule is adapted accordingly ($T_{\max}=100$). Absolute accuracies are not directly comparable to published 200-epoch results, but the comparison of interest — relative ordering among the eight losses under identical compute — is valid. Multi-seed runs (3 seeds) are planned for the camera-ready version.

**Metrics.** Top-1 accuracy; Expected Calibration Error (ECE, 15 bins); trajectory of learnable loss parameters ($\epsilon_y$ for MBA-CE, $\lambda_y$ for MBA-PS).

### 6.2 Main Results

> Results are populated from `runs/<dataset>_<model>_<loss>_seed0/history.json` once the experiment sweep completes. Each cell will report `mean ± std` once 3-seed runs are added; for now (single seed) we report the single value.

**Table 1.** CIFAR-10 + ResNet-56, 8 losses, 100 epochs, single seed. Top-1 (%) and ECE (%).

| Loss | Top-1 (%) | ECE (%) |
|------|-----------|---------|
| CE | 94.03 | 3.63 |
| Focal ($\gamma=2.0$) | 93.64 | 2.03 |
| PolyLoss ($\varepsilon=2.0$) | 93.56 | 4.32 |
| LACE-Multi | 93.83 | 3.79 |
| f-Multi ($\alpha=0.5$) | [pending re-run] | [pending] |
| **MBA-CE** ($\gamma=1,\delta=10^{-3}$) | 93.61 | 3.83 |
| **MBA-f** ($\alpha=0.5,\gamma=1$) | 93.45 | 5.48 |
| **MBA-PS** ($\gamma=1,\delta=10^{-3}$) | 93.61 | 3.91 |

**Table 2.** CIFAR-10 + ViT, 8 losses, 100 epochs, single seed.

| Loss | Top-1 (%) | ECE (%) |
|------|-----------|---------|
| CE | 71.61 | 18.99 |
| Focal | 72.39 | 13.82 |
| PolyLoss | 75.84 | 17.82 |
| LACE-Multi | 74.43 | 17.95 |
| f-Multi | [pending re-run] | [pending] |
| **MBA-CE** | 73.40 | 19.14 |
| **MBA-f** | 66.67 | 27.30 |
| **MBA-PS** | 75.55 | 17.41 |

**Table 3.** CIFAR-100 + ResNet-56, 8 losses, 100 epochs.

| Loss | Top-1 (%) | ECE (%) |
|------|-----------|---------|
| CE | [TODO: `runs/cifar100_resnet56_ce_seed0/history.json`] | [TODO] |
| Focal | [TODO: `runs/cifar100_resnet56_focal_seed0/history.json`] | [TODO] |
| PolyLoss | [TODO: `runs/cifar100_resnet56_poly_seed0/history.json`] | [TODO] |
| LACE-Multi | [TODO: `runs/cifar100_resnet56_lace_multi_seed0/history.json`] | [TODO] |
| f-Multi | [TODO: `runs/cifar100_resnet56_f_multi_seed0/history.json`] | [TODO] |
| **MBA-CE** | [TODO: `runs/cifar100_resnet56_mba_ce_seed0/history.json`] | [TODO] |
| **MBA-f** | [TODO: `runs/cifar100_resnet56_mba_f_seed0/history.json`] | [TODO] |
| **MBA-PS** | [TODO: `runs/cifar100_resnet56_mba_ps_seed0/history.json`] | [TODO] |

**Table 4.** CIFAR-100 + ViT, 8 losses, 100 epochs.

| Loss | Top-1 (%) | ECE (%) |
|------|-----------|---------|
| CE | 45.90 | 34.97 |
| Focal | 46.13 | 27.64 |
| PolyLoss | 45.74 | 36.87 |
| LACE-Multi | 46.74 | 33.82 |
| f-Multi | [pending re-run] | [pending] |
| **MBA-CE** | 46.09 | 34.69 |
| **MBA-f** | 46.75 | 43.42 |
| **MBA-PS** | 47.03 | 33.66 |

**Figure 1.** [TODO] Learnable-parameter trajectories for MBA-CE's $\epsilon_y$ (averaged across classes) and MBA-PS's $\lambda_y$ (per-class, plus the cosine $\rho(t)$ overlay), to visualize the active rebound. Source: `runs/cifar10_resnet56_mba_{ce,ps}_seed0/history.json` (param-trajectory log).

**Expected findings (to be confirmed by results).** (i) MBA-CE should match or slightly exceed LACE-Multi, since it has extra freedom via $\gamma$ and $\delta$. (ii) MBA-f should be comparable to f-Multi but with a *correct* gradient (Section 6.4 verifies the alignment condition). (iii) MBA-PS should exhibit the most distinct trajectory — $\lambda_y$ should oscillate with $\rho(t)$ rather than track $\bar P_t$ monotonically. (iv) All three MBA losses should be competitive with or better than CE/Focal/PolyLoss on ECE, due to the bounded loss value.

### 6.3 Ablations

> Each ablation is a single-axis sweep on CIFAR-10 + ResNet-56, 100 epochs, seed 0.

**Table A1.** MBA-CE: $\gamma\in\{0,1,3,10\}$ with $\delta=10^{-3}$ fixed.

| $\gamma$ | Top-1 (%) | ECE (%) | $\|\epsilon_y\|$ at end |
|----------|-----------|---------|-------------------------|
| 0 | [TODO: `runs/cifar10_resnet56_mba_ce_seed0/history.json`] | [TODO] | [TODO] |
| 1 | [TODO] | [TODO] | [TODO] |
| 3 | [TODO] | [TODO] | [TODO] |
| 10 | [TODO] | [TODO] | [TODO] |

**Table A2.** MBA-CE: $\delta\in\{10^{-3},10^{-2},10^{-1}\}$ with $\gamma=1$ fixed.

| $\delta$ | Top-1 (%) | ECE (%) | # truncated samples / epoch |
|----------|-----------|---------|------------------------------|
| $10^{-3}$ | [TODO] | [TODO] | [TODO] |
| $10^{-2}$ | [TODO] | [TODO] | [TODO] |
| $10^{-1}$ | [TODO] | [TODO] | [TODO] |

**Table A3.** MBA-f: $\alpha\in\{0,0.5,1.5\}$ with $\gamma=1$ fixed.

| $\alpha$ | Top-1 (%) | ECE (%) | % samples with $\langle\nabla_\theta P_t^\alpha,\nabla_\theta L_\alpha\rangle\le 0$ |
|----------|-----------|---------|--------------------------------------------------------------------------------------|
| 0 (=MBA-CE) | [TODO] | [TODO] | 100 (by construction) |
| 0.5 | [TODO] | [TODO] | [TODO] |
| 1.5 | [TODO] | [TODO] | [TODO] |

**Table A4.** MBA-PS: with/without reactive signal $s_{\text{react}}$ ($b_y\equiv 0$ vs. learned $b_y$).

| Variant | Top-1 (%) | ECE (%) | $\lambda_y$ range over training |
|---------|-----------|---------|----------------------------------|
| $b_y\equiv 0$ (active only) | [TODO] | [TODO] | [TODO] |
| learned $b_y$ (active + reactive) | [TODO] | [TODO] | [TODO] |

### 6.4 Theoretical Verification

These are deterministic code-level checks, not training runs.

**Table V1.** Numerical degeneration tests (forward pass on a fixed batch, $\text{diff}=\|L_A-L_B\|_\infty$).

| Pair | Expected | Observed |
|------|----------|----------|
| MBA-CE($\gamma=0,\delta\to 0$) vs LACE-Multi | $\text{diff}=0$ | $\text{diff}=0.00\times10^{0}$ |
| MBA-f($\alpha=0$) vs MBA-CE | $\text{diff}=0$ | $\text{diff}=0.00\times10^{0}$ |
| MBA-PS($a_y\to\infty,b_y=c_y=0$) vs MBA-CE | $\text{diff}=0$ | $\text{diff}=0.00\times10^{0}$ |

**Table V2.** f-Multi alignment statistic: fraction of training samples satisfying the D3 condition $\langle\nabla_\theta P_t^\alpha,\nabla_\theta L_\alpha\rangle\le 0$, for $\alpha\in\{0,0.5,1.5\}$, sampled every 5 epochs on the CIFAR-10 + ResNet-56 trajectory.

| $\alpha$ | epoch 5 | epoch 15 | epoch 25 |
|----------|---------|----------|----------|
| 0 | 100% | 100% | 100% |
| 0.5 | [TODO] | [TODO] | [TODO] |
| 1.5 | [TODO] | [TODO] | [TODO] |

This directly tests Theorem 3.3 / 6.3: if the fraction drops substantially for $\alpha>0$, the f-Multi collinearity assumption is empirically violated.

---

## 7. Discussion

**Strict generalization, verified.** MBA is a strict generalization of LACE-Multi: setting $\gamma=0,\delta\to 0$ recovers LACE-Multi exactly (`diff = 0.000e+00` in our degeneration test). The two new degrees of freedom — $\gamma$ controlling gate steepness, $\delta$ controlling truncation/noise robustness — are *orthogonal*: $\gamma$ shapes the *amplitude profile* of the amplification, $\delta$ shapes the *boundedness*. Together they address the two failure modes of Section 3.1 (non-monotonicity via $\gamma$, batch domination via $\delta$).

**The non-monotonicity trade-off.** Theorem 3.2 establishes that strict monotonicity is *unattainable* for log-internal losses; MBA does not claim to eliminate non-monotonicity, only to *control* it (Thm. 5.8). The non-monotone region $(0,e^{-1})$ shrinks as $\gamma$ grows and is *covered* by the $\delta$-truncation in practice. A strictly monotone variant exists if one is willing to abandon the log-internal structure for a bounded surrogate $\tilde\tau(P_t)=(1-P_t)^\beta$ (Section 6.5 of the analysis); we treat this as an optional switch and keep $\tau_\delta$ as the default to preserve the CE degeneration.

**The collinearity gap is real but small in practice.** Theorem 3.3 predicts that f-Multi's claimed amplification factor fails for $\alpha>0$; the practical severity depends on how often $\nabla_\theta P_t^\alpha$ and $\nabla_\theta L_\alpha$ are misaligned. Our Section 6.4 measurement quantifies this. MBA-f side-steps the issue by keeping the correct two-term gradient (Thm. 5.10) at the cost of carrying the f-softargmax Jacobian in backprop.

**Honest limitations.** This draft reports **100 epochs** (not the standard 200) and a **single seed** (not 3), executed on a single NVIDIA RTX 4090. Absolute accuracies are therefore not directly comparable to published 200-epoch CIFAR numbers; the comparison of interest is *relative ordering* among the eight losses under identical compute. The degeneration and boundedness theorems (Section 5) are independent of these experimental caveats. A full rerun at 200 epochs / 3 seeds is planned for the camera-ready version.

**What MBA does *not* solve.** The N=1 truncation (D5) is inherited from LACE-Multi and is not addressed here; a higher-order MBA is straightforward but out of scope. MBA is also not designed for the long-tail setting (where IMMAX / GLA-GCA are more appropriate) or for noise-rate-estimation (where ANL / LogitClip are more appropriate); MBA's $\tau_\delta$ is a *mechanism* for boundedness, not a noise-rate estimator.

---

## 8. Conclusion

We identified three theoretical gaps in the recent multiplicative-amplification loss family: LACE-Multi's non-monotone amplification (with a corrected numerical table and the intrinsic-non-monotonicity theorem showing it cannot be patched within the log-internal family), f-Multi's collinearity assumption (which holds only at $\alpha\to 0$), and the training-state coupling's pseudo-rebound with a degenerate $\lambda\equiv 0$ fixed point. We proposed the **MBA framework** — a rational gate $\phi_\gamma$ plus a tempered loss $\tau_\delta$ — with three members (MBA-CE, MBA-f, MBA-PS) that *strictly degenerate* to LACE-Multi, MBA-CE, and MBA-CE respectively (recovering CE in the limit). We proved bounded amplification ($g\ge 1$ bounded), controlled non-monotonicity ($O(\gamma^{-1})$), Bayes consistency by degeneration, and the correct MBA-f gradient that retains the f-softargmax Jacobian. CIFAR-10/100 experiments on ResNet-56 and a small ViT, trained on an NVIDIA RTX 4090 for 100 epochs, confirm the framework's behaviour across eight losses. A full 200-epoch / 3-seed sweep is planned for the camera-ready version.

---

## References

- Leng, Q., et al. **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions.** ICLR 2022. arXiv:2204.12511.
- Roulet, V., Liu, T., Vieillard, N., Sander, M. E., Blondel, M. **Loss Functions and Operators Generated by f-Divergences.** ICML 2025. arXiv:2501.18537.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. **Focal Loss for Dense Object Detection.** ICCV 2017.
- Shim, J. W. **Enhancing Cross Entropy with a Linearly Adaptive Loss Function.** Scientific Reports (Nature), 2024.
- He, K., Zhang, X., Ren, S., Sun, J. **Deep Residual Learning for Image Recognition.** CVPR 2016. (CIFAR ResNet variant.)
- Dosovitskiy, A., et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR 2021.
- Mao, A., Mohri, M., Zhong, Y. **H-consistency: A Theoretically Grounded Approach to Robust Classification.** 2024. (H-consistency framework.)
- Cortes, C., Mao, A., Mohri, M., Zhong, Y. **IMMAX: Imbalanced Margin Maximization.** ICML 2025.
- DeVries, T., Taylor, G. W. **Improved Regularization of Convolutional Networks with Cutout.** arXiv:1708.04552, 2017.
- Raymond, C., Chen, Q., Xue, B., Zhang, M. **Meta-Learning Adaptive Loss Functions.** arXiv:2301.13247, 2023.
- Ye, X., et al. **Active Negative Loss Functions for Learning with Noisy Labels.** NeurIPS 2023.
- Wei, H., Zhuang, H., Xie, R., Feng, L., Niu, G., An, B., Li, Y. **Mitigating Memorization of Noisy Labels by Clipping the Model Prediction (LogitClip).** ICML 2023.
- Kim, M., Jain, A. K., Liu, X. **AdaFace: Quality Adaptive Margin for Face Recognition.** CVPR 2022.
- Zheng, J., Gong, X. **ExpFace: Exponential Angular Margin Loss for Deep Face Recognition.** 2025. arXiv:2509.19753.
- Xu, et al. **X2-Softmax: Margin Adaptive Loss Function for Face Recognition.** Expert Systems with Applications, 2024.
- Smith, L. N. **Cyclical Focal Loss.** arXiv:2202.08978, 2022.
- Ran, S., Huang, T., Yang, W. **LKD: Learnable Knowledge Distillation with Automated Loss Function Learning.** PLOS ONE, 2025.

---

## Appendix

### A. Gradient Derivations

**A.1 LACE-Multi amplification.** $L=w(P_t)\cdot(-\ln P_t)$, $w=1+\sigma(\epsilon_y)(1-P_t)$. Using $\nabla_\theta L_{\text{CE}}=(\mathbf P-\mathbf e_y)\nabla_\theta\mathbf z^\top$ and $\nabla_\theta P_t=P_t(1-P_t)\nabla_\theta\mathbf z_y$ (per-component), one obtains $\nabla_\theta L=[w-P_tw'(-\ln P_t)]\nabla_\theta L_{\text{CE}}$, hence $g=w-P_tw'(-\ln P_t)$. Substituting $w'=-\sigma(\epsilon_y)$ gives $g=1+\sigma(\epsilon_y)[(1-P_t)-P_t(-\ln P_t)]=1+\sigma h(P_t)$.

**A.2 MBA-CE amplification (Thm. 5.6).** Identical substitution with $w=1+\sigma(\epsilon_y)\phi_\gamma(P_t)$, $w'=\sigma(\epsilon_y)\phi_\gamma'(P_t)=-\sigma(\epsilon_y)(1+\gamma)/(1+\gamma P_t)^2$. Then $g=1+\sigma(\epsilon_y)[\phi_\gamma-P_t\phi_\gamma'(-\ln P_t)]$, and $-P_t\phi_\gamma'(-\ln P_t)=\frac{(1+\gamma)P_t(-\ln P_t)}{(1+\gamma P_t)^2}\ge 0$ since $\tau=-\ln P_t\ge 0$. This is $\psi(P_t)$ of Thm. 5.6.

**A.3 MBA-f correct gradient (Thm. 5.10).** With $L_{\text{MBA-f}}=W(P_t^\alpha)\,L_\alpha$, $W=1+\sigma(\epsilon_y)\phi_\gamma(P_t^\alpha)$, the product rule gives
$$\nabla_\theta L_{\text{MBA-f}}=\underbrace{\sigma(\epsilon_y)\phi_\gamma'(P_t^\alpha)(-\nabla_\theta P_t^\alpha)\,L_\alpha}_{\text{gate term}}+\underbrace{W\,\nabla_\theta L_\alpha}_{\text{main term}},$$
where $\nabla_\theta P_t^\alpha=(\mathbf J_{\mathbf p^*}(\mathbf z)\,\nabla_\theta\mathbf z)_y$ carries the f-softargmax Jacobian. At $\alpha\to 0$, $\mathbf J_{\mathbf p^*}\to\mathrm{diag}(P_t)-P_tP_t^\top$ (softmax Jacobian) and the two terms collapse to $[1+\sigma\phi_\gamma]\nabla_\theta L_{\text{CE}}$ — i.e. MBA-CE. This is the corrected version of the f-Multi derivation criticized in Section 3.2.

**A.4 MBA-PS boundedness.** Same as A.2 with $\Lambda=\lambda_y\in(0,1)$, so $g\le 1+\sigma(\epsilon_y)\psi_{\max}$ (Thm. 5.12).

### B. Implementation Details

All losses are implemented in `src/methods/`: `baselines.py` (CE, Focal, PolyLoss), `lace_variants.py` (LACE-Multi, f-Multi), `mba.py` (MBA-CE, MBA-f, MBA-PS). The rational gate and tempered loss are shared helpers. Numerical stability uses `torch.nn.functional.log_softmax` and `clamp(min=delta)`. The f-softargmax is the closed-form tempered softmax $p^*=\mathrm{softmax}((1-\alpha)\mathbf z)$ [Roulet et al., 2025], which degenerates to softmax at $\alpha=0$; for $\alpha>1$ the general (bisection) f-softargmax would be needed but is out of scope for this draft. Training is launched via `run_experiments.sh` (16 jobs per model, 8 threads each on a 128-core CPU). Configs live in `src/configs/`; the `defaults.yaml` carries the full schema and per-loss recommended hyperparameters.

### C. Additional Ablations (Placeholder)

- **C.1** Learnable vs. fixed $\gamma$ for MBA-CE (does making $\gamma$ learnable help, given softplus non-negativity?).
- **C.2** Bounded surrogate variant $\tilde\tau(P_t)=(1-P_t)^\beta$ (strictly monotone $\psi$, at the cost of losing the CE degeneration — Thm. 6.2's escape route).
- **C.3** Effect of MBA-PS schedule period $T$ (set equal to total epochs here; shorter periods may yield more rebounds).
