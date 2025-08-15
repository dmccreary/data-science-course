---
title: Bell Curve Emergence
description: A test to see if a local image will be used to generate a social media card.
image: formatting-tests/social-preview-test.png
hide:
  - toc           # Hide the table of contents on the right
---
# Bell Curve Emergence

<iframe src="main.html" height="450"  scrolling="no"></iframe>

## Sample iframe

You can include the following in your HTML to use this MicroSim

```html
<iframe src="DOC_ROOT/docs/sims/MICROSIM_NAME/main.html" height="500px" scrolling="no"></iframe>
```

[Run the MicroSim](main.html){ .md-button .md-button--primary }
[Edit this MicroSim](https://editor.p5js.org/dmccreary/sketches/P5_EDITOR_ID)

**Topics:** Normal distribution, Central Limit Theorem (CLT), sampling distribution of the mean, law of large numbers, z‑scores

**Description:**
Students watch a true bell curve emerge in real time by repeatedly sampling from a **non‑normal base distribution** (choose Uniform, Skewed, Exponential, or Bernoulli). Each iteration draws `n` observations, computes their **sample mean**, and drops a small dot into a **running histogram of sample means**. As more means accumulate and/or as `n` increases, the histogram approaches a **normal (Gaussian) curve**. Overlay a **theoretical normal** with mean `μ` (of the base distribution) and standard deviation `σ/√n` to make the convergence explicit. Optional shading shows central probability regions (e.g., ±1σ, ±2σ).

**Input Controls (in controls region):**

1.  **Base Distribution** (dropdown: Uniform\[0,1\], Skewed (Beta), Exponential(λ), Bernoulli(p))

2.  **Sample Size (n)** (slider: 1 → 100, default 10)

3.  **Samples per Tick** (slider: 1 → 200, default 50) --- controls animation speed

4.  **Bins** (slider: 10 → 80, default 51)

5.  **Show Theoretical Normal** (checkbox)

6.  **Shade ±σ, ±2σ** (checkbox)

7.  **Start/Pause** (button)

8.  **Reset** (button)

**On‑Canvas Readouts:**

-   Base `μ`, `σ`; current `n`; theoretical sampling SD `σ/√n`

-   Running count of sample means collected

-   Optional z‑score under cursor (hover on histogram to show area left of x)

**Learning Objectives:**

-   See **why** the sampling distribution of the mean approaches normality **regardless of the base distribution**.

-   Connect **sample size** to the **spread** of the sampling distribution (`σ/√n`).

-   Interpret **bell curve** parameters visually and relate to empirical rules (68--95--99.7%).

-   Distinguish between a **population distribution** (often non‑normal) and the **sampling distribution of the mean** (approximately normal for large `n`).

**Difficulty:** Intermediate

**Implementation Notes (p5.js, responsive width per standard rules):**

-   Two‑column layout: left = controls + small "base distribution preview"; right = main histogram of **sample means**.

-   Maintain an array `means[]`; on each tick, generate `k = Samples per Tick` sample means by drawing `n` IID values from the chosen base distribution, pushing their averages.

-   Histogram x‑axis spans a sensible range around the base mean (e.g., `μ ± 4σ`); update bin counts incrementally for performance.

-   When "Show Theoretical Normal" is checked, draw `N(μ, σ²/n)` scaled to histogram area.

-   For shading, fill under the curve between `μ±σ` and `μ±2σ`.

-   Accessibility: `describe(description, p5.LABEL)` with a clear string like:
    "Sampling distribution simulator that shows a bell curve emerging from sample means drawn from a chosen non‑normal population."

**Nice Extras (optional):**

-   Toggle between **"Sample Means"** vs **"Sample Sums"** (sums also go normal as `n` grows).

-   Snapshot button to **freeze and annotate** the current histogram with mean/SD labels.

-   A tiny **QQ‑plot** inset comparing sample means to a perfect normal line.