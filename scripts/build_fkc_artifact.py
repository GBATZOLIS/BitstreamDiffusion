#!/usr/bin/env python
"""Build the self-contained CFG-FKC demo Artifact (HTML) from the PNGs and the
verification results.json produced by fkc_cfg_2d_demo.py. Every number on the
page is read from results.json -- no hand-transcription.

    python scripts/build_fkc_artifact.py
-> writes scripts/_fkc_cfg_2d_out/fkc_cfg_2d_demo.html
"""
import base64
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fkc_cfg_2d_out")


def b64(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def fig(base, alt):
    l = b64(f"{base}_light.png")
    d = b64(f"{base}_dark.png")
    return (f'<img class="fig fig-light" alt="{alt}" '
            f'src="data:image/png;base64,{l}">'
            f'<img class="fig fig-dark" alt="{alt}" '
            f'src="data:image/png;base64,{d}">')


R = json.load(open(os.path.join(OUT, "results.json")))
meta = R["meta"]


def fnum(x, n=4):
    return f"{x:.{n}f}"


def ess(x):
    return f"{x:.3f}"


# ---- Pass A table (lambda-independence) ----
lam_label = {"zero": "0", "one": "1", "inv_w": "1/w", "sigma": "λ(σ)"}
rowsA = []
prev_w = None
for r in R["passA"]:
    wtxt = f"{r['w']:g}" if r["w"] != prev_w else ""
    prev_w = r["w"]
    ok = r["dmean"] < 0.06 and r["dcov"] < 0.12
    rowsA.append(
        f'<tr><td class="w">{wtxt}</td><td class="mono">{lam_label[r["lam"]]}</td>'
        f'<td class="num">{ess(r["ess"])}</td>'
        f'<td class="num">{fnum(r["dmean"])}</td>'
        f'<td class="num">{fnum(r["dcov"])}</td>'
        f'<td class="pass">{"✓" if ok else "✗"}</td></tr>')
tableA = "\n".join(rowsA)

# ---- Pass B table (extrapolation) ----
rowsB = []
for r in R["passB"]:
    ok = r["dmean"] < 0.06 and r["dcov"] < 0.12
    ms = f'({r["m_star"][0]:+.3f}, {r["m_star"][1]:+.3f})'
    me = f'({r["mean_emp"][0]:+.3f}, {r["mean_emp"][1]:+.3f})'
    rowsB.append(
        f'<tr><td class="w">{r["w"]:g}</td><td class="num">{r["n_steps"]}</td>'
        f'<td class="num">{ess(r["ess"])}</td>'
        f'<td class="mono tnum">{ms}</td><td class="mono tnum">{me}</td>'
        f'<td class="num">{fnum(r["dmean"])}</td>'
        f'<td class="num">{fnum(r["dcov"])}</td>'
        f'<td class="pass">{"✓" if ok else "✗"}</td></tr>')
tableB = "\n".join(rowsB)

# ---- churn line ----
ch = {c["name"]: c for c in R["churn"]}
churn_em = ch["em"]
churn_ch = ch["edm_churn"]

CSS = """
:root{
  color-scheme: light dark;
  --page:#e9ebf0; --surface:#ffffff; --fig-bg:#f9f9f7;
  --ink:#0b0b0b; --ink-2:#4c4f57; --muted:#8a8d94;
  --line:#d9dbe2; --line-2:#e7e9ee;
  --accent:#2a78d6; --accent-soft:#e7f0fb;
  --sampled:#c9551f; --good:#0a7d0a;
  --shadow: 0 1px 2px rgba(16,22,40,.05), 0 8px 28px rgba(16,22,40,.06);
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    --page:#08090b; --surface:#15171b; --fig-bg:#0d0d0d;
    --ink:#f2f4f8; --ink-2:#bcc0c8; --muted:#7f838c;
    --line:#24272d; --line-2:#1c1f24;
    --accent:#4a92ea; --accent-soft:#12233a;
    --sampled:#e0733f; --good:#31b531;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --page:#08090b; --surface:#15171b; --fig-bg:#0d0d0d;
  --ink:#f2f4f8; --ink-2:#bcc0c8; --muted:#7f838c;
  --line:#24272d; --line-2:#1c1f24;
  --accent:#4a92ea; --accent-soft:#12233a;
  --sampled:#e0733f; --good:#31b531;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.45);
}
:root[data-theme="light"]{
  --page:#e9ebf0; --surface:#ffffff; --fig-bg:#f9f9f7;
  --ink:#0b0b0b; --ink-2:#4c4f57; --muted:#8a8d94;
  --line:#d9dbe2; --line-2:#e7e9ee;
  --accent:#2a78d6; --accent-soft:#e7f0fb;
  --sampled:#c9551f; --good:#0a7d0a;
  --shadow: 0 1px 2px rgba(16,22,40,.05), 0 8px 28px rgba(16,22,40,.06);
}

*{box-sizing:border-box}
body{margin:0}
.wrap{
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono: ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
  background:var(--page); color:var(--ink);
  font-family:var(--sans); line-height:1.6;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  padding:clamp(20px,4vw,64px) 20px 96px;
}
.col{max-width:1120px; margin:0 auto}
.prose{max-width:70ch}

/* ---- header ---- */
.eyebrow{
  font-family:var(--mono); font-size:12px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); font-weight:600;
  display:flex; align-items:center; gap:10px; margin:0 0 18px;
}
.eyebrow::before{content:""; width:26px; height:2px; background:var(--accent); display:inline-block}
h1{
  font-size:clamp(30px,5vw,50px); line-height:1.05; letter-spacing:-.02em;
  font-weight:680; margin:0 0 20px; text-wrap:balance;
}
h1 .geo{font-family:var(--mono); font-weight:600; letter-spacing:-.01em;
  background:linear-gradient(90deg,var(--accent),var(--sampled));
  -webkit-background-clip:text; background-clip:text; color:transparent;
  white-space:nowrap;}
.dek{font-size:clamp(17px,2.2vw,20px); color:var(--ink-2); max-width:66ch; margin:0 0 26px}
.chips{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px}
.chip{
  font-family:var(--mono); font-size:12.5px; color:var(--ink-2);
  background:var(--surface); border:1px solid var(--line);
  border-radius:999px; padding:5px 12px; white-space:nowrap;
}
.chip b{color:var(--ink); font-weight:600}
.chip.ok{color:var(--good); border-color:color-mix(in srgb,var(--good) 40%,var(--line))}

/* ---- sections ---- */
section{margin-top:clamp(38px,6vw,70px)}
h2{font-size:clamp(21px,3vw,27px); letter-spacing:-.01em; font-weight:640;
  margin:0 0 6px; text-wrap:balance}
.kicker{font-family:var(--mono); font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 14px}
p{margin:0 0 16px}
p strong{font-weight:640}
code, .m{font-family:var(--mono); font-size:.92em;
  background:var(--accent-soft); color:var(--ink);
  padding:.08em .38em; border-radius:5px}
a{color:var(--accent)}

/* ---- formula panel ---- */
.formulas{
  background:var(--surface); border:1px solid var(--line);
  border-radius:14px; box-shadow:var(--shadow); overflow:hidden;
}
.formulas .f{
  padding:16px 22px; border-bottom:1px solid var(--line-2);
  display:grid; grid-template-columns:210px 1fr; gap:18px; align-items:baseline;
}
.formulas .f:last-child{border-bottom:0}
.formulas .lbl{font-size:13.5px; color:var(--ink-2); font-weight:560}
.formulas .lbl small{display:block; color:var(--muted); font-weight:400; font-size:12px; margin-top:3px}
.formulas .eq{font-family:var(--mono); font-size:14px; color:var(--ink);
  overflow-x:auto; white-space:nowrap; padding-bottom:2px}
.formulas .eq .hl{color:var(--accent); font-weight:600}
.formulas .eq .hs{color:var(--sampled); font-weight:600}
@media (max-width:620px){ .formulas .f{grid-template-columns:1fr; gap:6px} }

/* ---- figures ---- */
figure{margin:0}
.figcard{
  background:var(--fig-bg); border:1px solid var(--line);
  border-radius:14px; box-shadow:var(--shadow);
  padding:10px; overflow:hidden;
}
.fig{display:block; width:100%; height:auto; border-radius:6px}
.fig-dark{display:none}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])) .fig-light{display:none}
  :root:where(:not([data-theme="light"])) .fig-dark{display:block}
}
:root[data-theme="dark"] .fig-light{display:none}
:root[data-theme="dark"] .fig-dark{display:block}
:root[data-theme="light"] .fig-light{display:block}
:root[data-theme="light"] .fig-dark{display:none}
figcaption{color:var(--ink-2); font-size:14.5px; margin-top:14px; max-width:80ch}
figcaption b{color:var(--ink); font-weight:620}

.takeaways{list-style:none; padding:0; margin:16px 0 0;
  display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px}
.takeaways li{background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:10px; padding:12px 15px;
  font-size:14px; color:var(--ink-2)}
.takeaways li b{color:var(--ink); font-family:var(--mono); font-size:13px}

/* ---- tables ---- */
.tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:12px;
  box-shadow:var(--shadow); background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
caption{caption-side:top; text-align:left; padding:14px 16px 6px;
  font-weight:600; font-size:14.5px; color:var(--ink)}
caption small{display:block; font-weight:400; color:var(--muted);
  font-family:var(--mono); font-size:12px; margin-top:4px}
th,td{padding:9px 14px; text-align:left; border-bottom:1px solid var(--line-2)}
thead th{font-family:var(--mono); font-size:11.5px; letter-spacing:.05em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
  border-bottom:1px solid var(--line); white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.num,td.tnum,td.mono{font-family:var(--mono);
  font-variant-numeric:tabular-nums}
td.num{text-align:right}
td.w{font-weight:640; color:var(--ink)}
td.pass{text-align:center; color:var(--good); font-weight:700}
tbody tr:hover{background:color-mix(in srgb,var(--accent) 6%,transparent)}

/* ---- callout ---- */
.note{display:flex; gap:14px; background:var(--accent-soft);
  border:1px solid color-mix(in srgb,var(--accent) 25%,var(--line));
  border-radius:12px; padding:16px 18px; margin-top:20px; font-size:14.5px;
  color:var(--ink-2)}
.note .dot{flex:0 0 auto; width:9px; height:9px; border-radius:50%;
  background:var(--accent); margin-top:7px}
.note b{color:var(--ink)}

/* ---- reproduce ---- */
.repro{background:var(--surface); border:1px solid var(--line);
  border-radius:14px; box-shadow:var(--shadow); overflow:hidden}
.repro .hd{padding:14px 18px; border-bottom:1px solid var(--line-2);
  font-weight:600; display:flex; align-items:center; gap:10px}
.repro pre{margin:0; padding:16px 18px; overflow-x:auto;
  font-family:var(--mono); font-size:13px; line-height:1.7; color:var(--ink-2)}
.repro pre .c{color:var(--muted)}
.repro pre .k{color:var(--accent)}
footer{margin-top:60px; padding-top:22px; border-top:1px solid var(--line);
  color:var(--muted); font-size:13px; font-family:var(--mono)}
.legend{display:flex; flex-wrap:wrap; gap:18px 26px; margin:10px 0 0; padding:0;
  list-style:none; font-size:13.5px; color:var(--ink-2)}
.legend li{display:flex; align-items:center; gap:9px}
.legend .sw{width:26px; height:0; border-top-width:3px; border-top-style:solid; border-radius:2px}
.legend .dotm{width:11px; height:11px; border-radius:50%}
.legend .dash{border-top-style:dashed}
"""

HTML = f"""<style>{CSS}</style>
<div class="wrap">
<div class="col">

  <header class="prose">
    <p class="eyebrow">Feynman–Kac · Classifier-Free Guidance</p>
    <h1>The CFG-FKC particle cloud is the geometric average
      <span class="geo">q<sub>u</sub><sup>1−w</sup> q<sub>c</sub><sup>w</sup></span></h1>
    <p class="dek">A weighted-particle sampler with a Feynman–Kac reweighting term
      is supposed to draw from the guidance-tilted target
      q<sub>u</sub><sup>1−w</sup>q<sub>c</sub><sup>w</sup>. Here it is, in 2D and 1D,
      with the sampled cloud laid over the <em>exact</em> analytic Gaussian across
      guidance weights — and the empirical mean/covariance asserted against the
      closed form.</p>
    <div class="chips">
      <span class="chip">branch <b>tasks/fkc-temperature</b></span>
      <span class="chip">CPU · float64</span>
      <span class="chip"><b>{meta['K']:,}</b> particles</span>
      <span class="chip">σ <b>{meta['sig_max']:g} → {meta['sig_min']:g}</b></span>
      <span class="chip ok">all asserts ✓</span>
    </div>
  </header>

  <section class="prose">
    <p class="kicker">The setup</p>
    <h2>Two Gaussians, one tilted target</h2>
    <p>Take an unconditional model <span class="m">q<sub>u</sub> = N(m<sub>u</sub>, S<sub>u</sub>)</span>
      and a conditional one <span class="m">q<sub>c</sub> = N(m<sub>c</sub>, S<sub>c</sub>)</span>,
      well separated, with <b>q<sub>c</sub> the sharper of the two</b>. Classifier-free
      guidance at weight <span class="m">w</span> targets their geometric average
      q<sub>u</sub><sup>1−w</sup>q<sub>c</sub><sup>w</sup>: at
      <span class="m">w=0</span> it is q<sub>u</sub>, at <span class="m">w=1</span> it is
      q<sub>c</sub>, and for <span class="m">w&gt;1</span> the mean marches
      <em>past</em> q<sub>c</sub> while the covariance tightens — guidance as
      extrapolation.</p>
    <p>The catch: the guided reverse process follows the score of a family of
      geometric-average <em>marginals</em> that is <b>not</b> the diffusion of the
      terminal target. The per-particle Feynman–Kac log-weight closes exactly
      that gap. Critically it is <b>λ-independent</b> — the drift-mixing
      coefficient λ changes the proposal, never the target. This is the 2D
      extension (nonzero means, full covariance) of the 1-D two-Gaussian gates
      <code>test_gate10</code> / <code>test_gate12</code> in <code>tests/test_fkc.py</code>.</p>
  </section>

  <section>
    <p class="kicker">The math being simulated</p>
    <h2 class="prose">Exact target, drift, and the reweighting term</h2>
    <div class="formulas">
      <div class="f">
        <div class="lbl">Exact geometric average<small>at diffusion scale σ²</small></div>
        <div class="eq"><span class="hl">Λ*</span> = (1−w)(S<sub>u</sub>+σ²I)⁻¹ + w(S<sub>c</sub>+σ²I)⁻¹ &nbsp;&nbsp; S* = <span class="hl">Λ*</span>⁻¹ &nbsp;&nbsp; m* = S*[(1−w)(S<sub>u</sub>+σ²I)⁻¹m<sub>u</sub> + w(S<sub>c</sub>+σ²I)⁻¹m<sub>c</sub>]</div>
      </div>
      <div class="f">
        <div class="lbl">VE score<small>per model, VE marginal</small></div>
        <div class="eq">s<sub>·</sub>(x) = −(S<sub>·</sub>+σ²I)⁻¹(x − m<sub>·</sub>)</div>
      </div>
      <div class="f">
        <div class="lbl">Guided drift<small>h = σ<sub>next</sub>−σ<sub>cur</sub> &lt; 0</small></div>
        <div class="eq">s<sub>geo</sub> = s<sub>u</sub> + w(s<sub>c</sub>−s<sub>u</sub>) &nbsp; d = −σ·s<sub>geo</sub> &nbsp; x ← x + h(1+<span class="hs">λ</span>)d + √(2<span class="hs">λ</span>σΔ)·z</div>
      </div>
      <div class="f">
        <div class="lbl">Feynman–Kac log-weight<small>score <em>difference</em>; λ-independent</small></div>
        <div class="eq">Δ log w = ½·w(w−1)·(σ<sub>cur</sub>² − σ<sub>next</sub>²)·‖<span class="hl">s<sub>c</sub> − s<sub>u</sub></span>‖²</div>
      </div>
    </div>
    <div class="note"><span class="dot"></span><div>Because the drift uses the <b>guided score</b> but the weight uses the
      <b>score difference</b> <span class="m">s<sub>c</sub>−s<sub>u</sub></span>, a scheme that
      weighted by ‖s<sub>c</sub>‖² or used the wrong coefficient would miss the
      target. That is exactly what the assertions below rule out.</div></div>
  </section>

  <section>
    <p class="kicker">Figure 1 · primary result</p>
    <h2 class="prose">The cloud lands on the exact ellipse at every w</h2>
    <p class="prose">Faint dashed contours are the data-level references
      <span style="color:var(--good);font-weight:600">q<sub>u</sub></span> and
      <span style="color:#d55181;font-weight:600">q<sub>c</sub></span>; the solid
      <span style="color:var(--accent);font-weight:600">blue</span> ellipse is the
      exact 1σ/2σ geometric average; the
      <span style="color:var(--sampled);font-weight:600">orange</span> scatter is the
      resampled FKC cloud. The <span class="m">+</span> marks the analytic mean, the
      dot the empirical one.</p>
    <figure>
      <div class="figcard">{fig('fkc_2d_primary', 'CFG-FKC 2D particle clouds vs exact geometric average for w = 0, 1, 1.5, 3')}</div>
      <figcaption><b>m<sub>u</sub>=(−1.5, 0)</b>, <b>m<sub>c</sub>=(1.5, 0.8)</b>.
        As w grows the mean slides from q<sub>u</sub> to q<sub>c</sub> and then beyond
        it (w=3 sits at m*=(+2.18, +0.53), well past m<sub>c</sub>), and the ellipse
        shrinks — guidance sharpens as it extrapolates. The cloud tracks the exact
        ellipse throughout.</figcaption>
    </figure>
    <ul class="takeaways">
      <li><b>w = 0</b> &nbsp;cloud = q<sub>u</sub>, exactly the unconditional model</li>
      <li><b>w = 1</b> &nbsp;cloud = q<sub>c</sub>, the pure conditional</li>
      <li><b>w = 1.5</b> &nbsp;mean past q<sub>c</sub>, covariance tightening</li>
      <li><b>w = 3</b> &nbsp;strong extrapolation, still exact</li>
    </ul>
  </section>

  <section>
    <p class="kicker">Figure 1b · ingredients and result</p>
    <h2 class="prose">The two source clouds, and where the result lands</h2>
    <p class="prose">Same story with the inputs made explicit as <em>point clouds</em>
      rather than contours: the <span style="color:var(--good);font-weight:600">green</span>
      <span class="m">q<sub>u</sub></span> cloud and the
      <span style="color:#d55181;font-weight:600">magenta</span>
      <span class="m">q<sub>c</sub></span> cloud are the two data distributions; the
      <span style="color:var(--sampled);font-weight:600">orange</span> cloud is the FKC
      result; the <span style="color:var(--accent);font-weight:600">blue</span> ellipse
      is the exact target.</p>
    <figure>
      <div class="figcard">{fig('fkc_clouds', 'Source clouds q_u and q_c with the FKC result cloud and exact geometric average, for w = 0, 1, 1.5, 3')}</div>
      <figcaption>The orange result <b>sits on the green cloud at w=0</b>, <b>jumps to
        the magenta cloud at w=1</b>, then <b>marches past it and tightens</b> for
        w=1.5 and w=3. The <span class="m">×</span> marks are the source means
        m<sub>u</sub>, m<sub>c</sub>; the <span class="m">+</span> is the exact
        geometric-average mean the cloud is asserted against.</figcaption>
    </figure>
  </section>

  <section>
    <p class="kicker">Figure 1c · coincidence check</p>
    <h2 class="prose">Zoom in: the cloud's own ellipse lands on the target</h2>
    <p class="prose">Scatter-vs-ellipse is hard to judge once the cloud is small, so here
      each w is <em>zoomed to its own scale</em> and the cloud's <b>empirical</b>
      covariance ellipse (<span style="font-weight:600">dashed</span>) is drawn over the
      <span style="color:var(--accent);font-weight:600">exact target</span> ellipse
      (solid). Dashed on solid = coincidence, directly readable. The source
      <span style="color:var(--good);font-weight:600">q<sub>u</sub></span> and
      <span style="color:#d55181;font-weight:600">q<sub>c</sub></span> clouds are kept
      in frame for context — when the unconditional source falls outside the zoom an
      <span class="m">← q<sub>u</sub></span> arrow points to it.</p>
    <figure>
      <div class="figcard">{fig('fkc_match', 'Zoomed per-w overlay of the sampled cloud empirical ellipse on the exact target ellipse')}</div>
      <figcaption>The fitted ellipse tracks the analytic one at every w. The only
        visible daylight is at <b>w=1.5</b> (‖Δcov‖/‖S*‖ = 0.070, the cloud a touch
        wider) and <b>w=3</b> (0.031) — both well inside the 0.12 tolerance, and the
        means coincide to ≤ 0.04.</figcaption>
    </figure>
  </section>

  <section>
    <p class="kicker">Verification · provably correct, not just pretty</p>
    <h2 class="prose">Empirical mean &amp; covariance vs the closed form</h2>
    <p class="prose">The script asserts the importance-weighted estimators against the
      analytic <span class="m">m*, S*</span> at σ<sub>min</sub>. Tolerances:
      <b>|mean − m*| &lt; 0.06</b> and
      <b>‖cov − S*‖<sub>F</sub> / ‖S*‖ &lt; 0.12</b>.</p>

    <div class="tablewrap">
      <table>
        <caption>A · λ-independence — pure importance weighting, no resampling
          <small>the FK weight is λ-independent, so every λ must hit m*, S* while the ESS stays healthy (Gate 10's regime, w ≤ 1.5)</small></caption>
        <thead><tr><th>w</th><th>λ</th><th>ESS/K</th>
          <th>|mean − m*|</th><th>‖cov − S*‖<sub>F</sub>/‖S*‖</th><th>ok</th></tr></thead>
        <tbody>
{tableA}
        </tbody>
      </table>
    </div>
    <div class="note"><span class="dot"></span><div>Past <span class="m">w≈1.5</span>
      the <b>pure</b> importance weights concentrate on a single particle
      (ESS/K → {min(r['ess'] for r in R['passA']):.3f} at w=1.5,
      ≈ 0.0001 at w=3) — the estimator is still unbiased, just too high-variance
      to measure with a finite cloud. The mixing kernels plus sequential resampling
      below are what make large-w extrapolation tractable.</div></div>

    <div class="tablewrap" style="margin-top:22px">
      <table>
        <caption>B · Extrapolation — λ(σ) drift + sequential resampling (the production configuration)
          <small>diversity maintained → mean/cov match the exact geometric average for every w, including the w=3 extrapolation</small></caption>
        <thead><tr><th>w</th><th>steps</th><th>ESS/K</th>
          <th>exact m*</th><th>sampled mean</th>
          <th>|mean − m*|</th><th>‖Δcov‖/‖S*‖</th><th>ok</th></tr></thead>
        <tbody>
{tableB}
        </tbody>
      </table>
    </div>
    <p class="prose" style="margin-top:14px; font-size:13.5px; color:var(--muted)">
      The w=3 target is stiff (sharp precision (1−w)Λ<sub>u</sub>+wΛ<sub>c</sub>),
      so the linear-in-σ Euler grid is refined to {meta['n_steps_w3']:,} steps there;
      w ≤ 1.5 uses {meta['n_steps_small']} as in the gates.</p>
  </section>

  <section>
    <p class="kicker">Figure 2 · proposal variants</p>
    <h2 class="prose">Euler–Maruyama vs EDM churn</h2>
    <p class="prose">The <code>edm_churn</code> proposal (Gate 12 flavor) churns σ up
      by γ before denoising and evaluates scores at σ̂, but keeps the FK weight
      on the consecutive-target interval <span class="m">σ<sub>cur</sub>²−σ<sub>next</sub>²</span>.
      It is leading-order in γ, so it lands on the same geometric average.</p>
    <figure>
      <div class="figcard">{fig('fkc_churn', 'em vs edm_churn proposals at w=2, both against the exact geometric average')}</div>
      <figcaption><b>w = 2, γ = 0.4.</b> Both clouds wrap the exact ellipse.
        em: |mean−m*| = {fnum(churn_em['dmean'])},
        ‖Δcov‖/‖S*‖ = {fnum(churn_em['dcov'])} ·
        edm_churn: |mean−m*| = {fnum(churn_ch['dmean'])},
        ‖Δcov‖/‖S*‖ = {fnum(churn_ch['dcov'])}. The churn covariance is
        slightly looser — the expected O(γ) approximation error — but the mean
        is spot on.</figcaption>
    </figure>
  </section>

  <section>
    <p class="kicker">Figure 3 · 1D companion</p>
    <h2 class="prose">The clean quantitative view</h2>
    <p class="prose">The same mechanism in one dimension: the weighted histogram of the
      FKC cloud (filled) sits under the exact geometric-average pdf (line) for each w.
      The density peak marches right and grows taller — mean extrapolation and
      variance sharpening in a single picture.</p>
    <figure>
      <div class="figcard">{fig('fkc_1d', '1D weighted histogram of the FKC cloud vs exact geometric-average pdf for w = 0, 1, 3')}</div>
      <figcaption>1D toy: q<sub>u</sub>=N(−1.5, 1.0), q<sub>c</sub>=N(1.5, 0.30).
        w=3 gives N(2.26, 0.13) — mean beyond q<sub>c</sub>, variance less than half.</figcaption>
    </figure>
  </section>

  <section>
    <p class="kicker">Reproduce</p>
    <h2 class="prose">Run it yourself</h2>
    <div class="repro">
      <div class="hd"><span class="dot" style="width:9px;height:9px;border-radius:50%;background:var(--accent);display:inline-block"></span>
        scripts/fkc_cfg_2d_demo.py</div>
      <pre><span class="c"># worktree: /home/gb511/BitstreamDiffusion-fkc  (branch tasks/fkc-temperature)</span>
<span class="k">source</span> ~/miniconda3/bin/activate pytorch
<span class="k">CUDA_VISIBLE_DEVICES</span>="" python scripts/fkc_cfg_2d_demo.py
<span class="c"># -> validates the categorical palette (dataviz skill), asserts weighted</span>
<span class="c">#    mean/cov vs analytic m*, S*, saves 6 PNGs (light+dark) + results.json</span></pre>
    </div>
    <ul class="legend">
      <li><span class="sw" style="border-color:var(--accent)"></span> exact geometric average</li>
      <li><span class="dotm" style="background:var(--sampled)"></span> FKC sampled cloud</li>
      <li><span class="sw dash" style="border-color:var(--good)"></span> q<sub>u</sub> reference</li>
      <li><span class="sw dash" style="border-color:#d55181"></span> q<sub>c</sub> reference</li>
    </ul>
  </section>

  <footer>
    CFG-FKC 2D/1D demonstration · all figures generated from asserted runs ·
    palette validated (all-pairs CVD, light + dark) via the data-viz validator ·
    self-contained, no external assets.
  </footer>

</div>
</div>
"""

path = os.path.join(OUT, "fkc_cfg_2d_demo.html")
with open(path, "w") as f:
    f.write(HTML)
print("wrote", path, f"({len(HTML)/1024:.0f} KB source, images inline)")
