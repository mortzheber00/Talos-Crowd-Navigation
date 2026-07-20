"""Generate the evaluation slide deck with figures inlined as data URIs."""
import base64, os, html

WEBP = "/home/sebas/.claude/jobs/73529903/tmp/webp"
GIFS = "/home/sebas/Documents/university/TUM/Master/SS26/LR/talos_crowd_navigation/.claude/worktrees/evaluation/experiments/figures/gifs"
OUT = "/home/sebas/.claude/jobs/73529903/deck.html"

def img(name):
    with open(os.path.join(WEBP, name + ".webp"), "rb") as f:
        return "data:image/webp;base64," + base64.b64encode(f.read()).decode()

def gif(name):
    with open(os.path.join(GIFS, "nav_" + name + ".gif"), "rb") as f:
        return "data:image/gif;base64," + base64.b64encode(f.read()).decode()

def gif_raw(fname):
    with open(os.path.join(GIFS, fname), "rb") as f:
        return "data:image/gif;base64," + base64.b64encode(f.read()).decode()

GIF_SCENS = [("empty", "Empty (control)"), ("head_on", "Head-on"),
             ("crossing", "Crossing"), ("corridor", "Corridor"),
             ("circle", "Circle"), ("random", "Random crowd")]

SLIDES = [
 dict(n=1, arc="setup", qid="", head="Crowd navigation for a Talos humanoid",
      sub="An evaluation of what the ORCA-on-a-biped stack actually does",
      body="""<p>MuJoCo · Talos · ORCA crowd · LIP-MPC + TSID walking controller.
      <strong>800+ simulation runs</strong> across eight questions.</p>
      <p class="muted">Lara Heidtmann · Florian Bohnes · Sebastián Serna · Moritz Heber · Nicolas Guenther</p>""",
      fig=None, take=None),

 dict(n=2, arc="setup", qid="", head="ORCA and a biped live in different worlds",
      sub="The project is a timescale and representation bridge",
      body="""<div class="split">
      <div><h4>ORCA</h4><ul><li>2D holonomic velocity planner</li>
      <li>assumes velocity changes <em>instantly</em></li>
      <li>agents are disks</li></ul></div>
      <div><h4>A biped</h4><ul><li>moves in discrete 0.8 s footsteps</li>
      <li>dynamically committed 1–2 steps ahead</li>
      <li>falls if pushed too hard</li></ul></div></div>
      <p>Everything interesting in this project lives at that seam. So the question is not
      &ldquo;does ORCA work&rdquo; — it is a published algorithm — but
      <strong>what does putting ORCA on top of a biped cost, and where does the coupling break?</strong></p>""",
      fig=None, take=None),

 dict(n=3, arc="setup", qid="", head="Three layers, three rates",
      sub="ROS topics at every boundary above the whole-body controller",
      body="""<pre class="arch">goal ─▶ <b>ORCA</b>  ~30 Hz            ─▶ v_cmd  (2D velocity)
     ─▶ <b>Reactive footstep planner</b>  1.25 Hz  ─▶ next FootStep
     ─▶ <b>LIP-MPC</b> 10 Hz ─▶ swing spline ─▶ <b>TSID</b> 300 Hz
     ─▶ MuJoCo</pre>
      <p>Pedestrians are kinematic ORCA disks (r = 0.3 m, v<sub>max</sub> = 1.3 m/s) that
      reciprocally avoid each other <em>and the robot</em>. The robot walks from the origin to (4, 0).</p>""",
      fig=None, take=None),

 dict(n=4, arc="setup", qid="method", head="What we asked",
      sub="The project was built demo-first; these are the questions it can answer",
      body="""<table class="q"><tbody>
      <tr><td class="qid">Q-A</td><td>Does each layer meet its real-time budget?</td><td class="muted">measurement</td></tr>
      <tr><td class="qid">Q-B</td><td>Does the footstep bridge execute the velocity command?</td><td class="muted">mechanism</td></tr>
      <tr><td class="qid">Q-B2</td><td>Should ORCA plan at a speed the walker can deliver?</td><td class="muted">intervention</td></tr>
      <tr><td class="qid">Q-C</td><td>Does the system navigate safely across scenarios?</td><td class="muted">validation</td></tr>
      <tr><td class="qid">Q-D</td><td>How much avoidance should the robot take on itself?</td><td class="muted">trade-off</td></tr>
      <tr><td class="qid">Q-E</td><td>At what crowd density does it break?</td><td class="muted">limits</td></tr>
      <tr><td class="qid">Q-F</td><td>What does onboard perception cost vs ground truth?</td><td class="muted">ablation</td></tr>
      <tr class="added"><td class="qid">Q-G</td><td>Is the robot avoiding the crowd, or the crowd avoiding the robot?</td><td class="muted">attribution</td></tr>
      <tr class="added"><td class="qid">Q-G2</td><td>How much crowd cooperation does it minimally need?</td><td class="muted">dose-response</td></tr>
      <tr class="added"><td class="qid">Q-H</td><td>Robot's share vs the crowd's — whose avoidance is it?</td><td class="muted">reciprocity</td></tr>
      </tbody></table>
      <p class="note">Q-G and its follow-ups (highlighted) were added <em>after</em> Q-C came back at 100 %
      everywhere. A result that uniform is a reason to check the setup, not to celebrate.</p>""",
      fig=None, take=None),

 dict(n=5, arc="setup", qid="method", head="How it was measured",
      sub="Headless sweeps, 10 seeds per condition, metrics to disk",
      body="""<div class="split">
      <div><h4>Setup</h4><ul>
      <li>800+ runs, headless, 10 at a time</li>
      <li>each run its own <code>ROS_DOMAIN_ID</code> — without it, parallel runs share a DDS
      domain and one run's ORCA steers on another's odometry</li>
      <li>10 seeds per condition; simulator otherwise deterministic (verified)</li>
      <li>Wilson 95 % intervals for rates, Wilcoxon signed-rank for paired tests</li></ul></div>
      <div><h4>Two decisions that change what the numbers mean</h4><ul>
      <li><strong>Clearance is always ground truth</strong>, never what the robot perceived —
      otherwise an undetected pedestrian silently vanishes from the safety statistics.</li>
      <li><strong>Samples after a fall are excluded</strong> from collision counts: a collapsed
      robot drifting into someone is not an avoidance failure.</li></ul></div></div>
      <p style="margin-top:16px">Every one of the 827 runs, robot path overlaid and coloured by outcome —
      the whole evaluation in one frame:</p>""",
      ensemble=True, fig=None, take=None),

 dict(n=6, arc="the puzzle", qid="Q-C", head="It works. Every criterion, met.",
      sub="53 runs · six scenarios · an empty-world control",
      body="""<p>100 % collision-free everywhere, zero falls, ≥5 moving pedestrians, planning far
      inside real time. The bar was 80 %.</p>""",
      fig="qc_scenarios", fig2="qc_trajectories",
      take=("ok", "But look at the trajectories: the robot walks <strong>nearly straight</strong> through "
            "every crowd — path ratio never exceeds 1.06. And clearance is 0.348 m in every scenario, "
            "identical to three decimals.")),

 dict(n=7, arc="the puzzle", qid="Q-C", head="The same six, moving",
      sub="Synchronised top-down runs — all reset together",
      body="""<p>The trajectories above, animated. Robot in <span style="color:#0d7a4f;
      font-weight:600">green</span>, pedestrians in grey, goal the red star. Every clip runs on the same
      18 s clock (the slowest scenario's finish), so they loop in lockstep.</p>""",
      gifs=True,
      take=("ok", "Watch the robot: it barely deviates in any of them. That is the observation Q-G is "
            "about to explain.")),

 dict(n=8, arc="the puzzle", qid="Q-A", head="The avoidance planner is free",
      sub="Measured one run at a time on an idle machine",
      body="""<p>The whole navigation stack — ORCA + footstep planner + MPC — is <b class="num">≈13 ms</b>
      p95 against the project's 50 ms bar. <strong>Criterion met comfortably.</strong></p>
      <p class="note">Honest caveats: TSID (2.33 ms) plus the MuJoCo step (1.17 ms) is 3.5 ms against a
      3.33 ms period, so the 300 Hz loop is marginally over in sum — that is why the sim runs ~3× slower
      than realtime. Perception's 1115 ms maximum is a first-call renderer allocation, not a stall.</p>""",
      fig="qa_timing_budget",
      take=("ok", "ORCA is <b class='num'>1.8 %</b> of its budget. Everything expensive is the whole-body "
            "controller or the perception front-end. The interesting engineering was never the solver.")),

 dict(n=8, arc="the puzzle", qid="Q-B", head="The bridge delivers half the speed, two steps late",
      sub="Commanded vs achieved velocity, 20 runs",
      body="""<p>ORCA asks for 1.2 m/s. The footstep planner clamps a step to
      1.4·step_x = 0.35 m per 0.8 s — <b class="num">0.44 m/s</b>. The command saturates permanently.</p>
      <table class="data"><tbody>
      <tr><td>forward speed gain</td><td class="num">0.50</td><td class="muted">half the command is delivered</td></tr>
      <tr><td>lateral (avoidance) gain</td><td class="num">1.08</td><td class="muted">avoidance <em>is</em> executed</td></tr>
      <tr><td>lateral lag</td><td class="num">1.57 s</td><td class="muted">≈ 2 footsteps late</td></tr>
      </tbody></table>
      <p class="note">The obvious story is wrong: the clamp scales the whole displacement vector, so
      lateral <em>should</em> be attenuated too. It is not — the forward component carries almost all the
      magnitude and is what saturates.</p>""",
      fig="qb_command_tracking",
      take=("warn", "Not a tracking bug — a <strong>specification mismatch</strong>. ORCA is asking for a "
            "speed that does not exist on this robot. Remember the 2-step lag; it returns once we remove "
            "the crowd's cooperation.")),

 dict(n=9, arc="the puzzle", qid="Q-B2", head="Asking for more speed delivers nothing",
      sub="80 runs sweeping ORCA's speed setpoint",
      body="""<table class="data head"><thead><tr><th>desired_speed</th><th>achieved</th><th>gain</th>
      <th>t<sub>goal</sub></th><th>path ratio</th></tr></thead><tbody>
      <tr><td class="num">0.30</td><td class="num">0.299</td><td class="num">1.02</td><td class="num">14.5 s</td><td class="num">1.075</td></tr>
      <tr><td class="num">0.45</td><td class="num">0.387</td><td class="num">0.93</td><td class="num">10.7 s</td><td class="num">1.024</td></tr>
      <tr class="best"><td class="num">0.60</td><td class="num">0.403</td><td class="num">0.76</td><td class="num">10.2 s</td><td class="num">1.016</td></tr>
      <tr><td class="num">1.20 <span class="muted">(shipped)</span></td><td class="num">0.408</td><td class="num">0.45</td><td class="num">10.3 s</td><td class="num">1.048</td></tr>
      </tbody></table>
      <p>Achieved speed saturates at <b class="num">≈0.40 m/s</b>. Between 0.6 and 1.2 the robot asks for
      twice the speed and arrives <em>no sooner</em>.</p>""",
      fig="qb2_speed_matching",
      take=("fix", "Set <code>desired_speed ≈ 0.6 m/s</code>. Free, and it makes the velocity command mean something.")),

 dict(n=10, arc="the puzzle", qid="Q-D · Q-E", head="Nothing we tune changes safety. Nothing breaks it either.",
      sub="110 runs of parameters · 70 runs of density",
      body="""<div class="split">
      <div><h4>Q-D — the safety knobs are inert</h4>
      <p>Minimum clearance is <b class="num">0.348 m</b> in <em>all twelve</em> cells of
      responsibility × safety margin. Margin 0.1 m or 0.4 m; responsibility 0.0 or 1.0 — it does not move.</p>
      <p class="muted">That constant is exactly what the <em>pedestrians'</em> ORCA produces:
      0.35 + 0.15 + 0.3 + 0.15 = 0.95 m centres → 0.35 m surface gap.</p>
      <p>What responsibility buys is <em>efficiency</em>: 16 s and 1.45× path at 0.0, ~9.3 s and ~1.0× above 0.2.</p></div>
      <div><h4>Q-E — no density limit exists</h4>
      <p>100 % collision-free at every crowd size from 0 to <b class="num">24</b> pedestrians. Tripling the
      crowd from 8 to 24 costs 1.5 s and 4 % of path length.</p>
      <p class="muted">A planner actually solving a harder problem would show <em>something</em> —
      longer detours, closer calls, occasional failures.</p></div></div>""",
      fig="qd_responsibility", fig2="qe_density",
      take=("warn", "A safety parameter that changes nothing about safety, and a density sweep with no "
            "breaking point. Two puzzles with one explanation.")),

 dict(n=11, arc="the resolution", qid="Q-G", head="The crowd is doing the avoiding",
      sub="Same scenarios, same seeds — pedestrians simply cannot see the robot",
      body="""<table class="data head"><thead><tr><th>condition</th><th>cooperative</th><th>blind</th>
      <th>clearance</th></tr></thead><tbody>
      <tr><td>head_on ×2</td><td class="num ok">100 %</td><td class="num bad">0 %</td><td class="num">+0.348 → <b class="bad">−0.544</b></td></tr>
      <tr><td>crossing ×4</td><td class="num ok">100 %</td><td class="num bad">0 %</td><td class="num">+0.349 → <b class="bad">−0.524</b></td></tr>
      <tr><td>random ×8</td><td class="num ok">100 %</td><td class="num bad">10 %</td><td class="num">+0.348 → <b class="bad">−0.487</b></td></tr>
      <tr><td>random ×16</td><td class="num ok">100 %</td><td class="num bad">0 %</td><td class="num">+0.348 → <b class="bad">−0.558</b></td></tr>
      </tbody></table>
      <p>Paired Wilcoxon over 40 pairs: <b class="num">W = 0, p = 1.8 × 10⁻¹²</b>. W = 0 means
      <em>every single pair</em> moved the same way. A clearance of −0.52 m against radii summing to
      0.6 m puts the disks nearly concentric — the robot walks <strong>through</strong> the pedestrian.</p>
      <p class="note"><strong>The robot is not idle.</strong> Path ratio rises from ≈1.0 to 1.26–2.14 and
      time-to-goal roughly doubles: its ORCA layer is steering hard. It walks at 0.4 m/s and executes each
      command 1.57 s late, while pedestrians close at 1.3 m/s — covering ~2 m inside its reaction lag.
      <strong>The avoidance is out-run, not under-tuned.</strong></p>""",
      fig="qg_cooperation",
      take=("bad", "This reinterprets everything: Q-C's 100 %, Q-D's frozen clearance, Q-E's missing "
            "density limit, Q-F's p = 0.76. Four puzzles, one cause — but <em>how much</em> cooperation? "
            "Next slide.")),

 dict(n=12, arc="the resolution", qid="Q-G2", head="But it only needs a little cooperation",
      sub="150 runs — the dose-response between cooperative and blind",
      body="""<p>Q-G is a binary. Q-G2 fills the middle: the crowd stays visible, but each
      pedestrian takes a smaller share of the avoidance (<code>ped_responsibility</code> 0.5 → 0.0).</p>
      <table class="data head"><thead><tr><th>ped_resp</th><th>head_on</th><th>crossing</th><th>random ×8</th></tr></thead><tbody>
      <tr><td class="num">0.0</td><td class="num ok">100 %</td><td class="num ok">100 %</td><td class="num bad">60 %</td></tr>
      <tr class="best"><td class="num">0.1</td><td class="num ok">100 %</td><td class="num ok">100 %</td><td class="num ok">100 %</td></tr>
      <tr><td class="num">0.3 <span class="muted">(shipped)</span></td><td class="num ok">100 %</td><td class="num ok">100 %</td><td class="num ok">100 %</td></tr>
      <tr><td class="num">0.5</td><td class="num ok">100 %</td><td class="num ok">100 %</td><td class="num ok">100 %</td></tr>
      </tbody></table>
      <p><strong>A sharp knee at 0.1.</strong> Once each pedestrian takes even 10 % of the avoidance,
      success is 100 %. And note: zero responsibility (crowd sees the robot but won't yield) is far milder
      than <em>blind</em> — the robot's own ORCA carries it as long as the crowd merely acknowledges it.</p>""",
      fig="qg2_ped_responsibility",
      take=("ok", "The honest refinement: the robot's avoidance is <em>sufficient for any crowd that can "
            "see it</em>, and useless against one that cannot. Q-G's 0 % is the blind extreme, not the "
            "whole story. The shipped 0.3 sits safely past the knee.")),

 dict(n=13, arc="the resolution", qid="Q-H", head="Whose share is it — robot or crowd?",
      sub="200 runs crossing the robot's share with the crowd's",
      body="""<p>Reference ORCA expects a pair's shares to sum to ≈1.0; the shipped default sums to only
      0.2 + 0.3 = 0.5. So does the <em>total</em> matter, or <em>who</em> provides it?</p>
      <p><strong>The failure is a single column, not a diagonal.</strong> Every cell where the crowd takes
      even 10 % is 100 %. The robot's own share is a weak floor: in the fully-non-yielding column it lifts
      success from 30 % to 60 %, but <em>saturates</em> — robot 0.5 and 1.0 are also stuck at 60 %,
      because it is out-run and cannot reach 100 % alone.</p>
      <p class="note">Panel 3 is the test: if the reciprocity <em>total</em> predicted success the four
      curves would collapse onto one. They don't — each has its knee at the same <em>pedestrian</em>
      share, not the same total. A total of 1.0 gives 60 % (robot 1.0 + ped 0.0) or 100 % (robot 0.5 +
      ped 0.5). The split matters, not the sum — because the two agents are not symmetric.</p>""",
      fig="qh_reciprocity",
      take=("ok", "The shipped (0.2, 0.3) sits deep in the safe interior. Q-H says the robot's 0.2 is "
            "barely adding to what the crowd's 0.3 already handles — it could be raised for cheap "
            "insurance toward the non-yielding corner.")),

 dict(n=14, arc="the resolution", qid="Q-F", head="Perception costs the goal, provably not safety",
      sub="60 runs, paired by scenario and seed",
      body="""<p>Collision-free success falls from 100 % to <b class="num bad">10 %</b> and time-to-goal
      roughly quadruples. Clearance does not move at all — <b class="num">p = 0.76</b> over 30 pairs.</p>
      <p>The camera tracks <em>the entire crowd</em> (2.0/2, 4.0/4, 8.0/8). It is not a detection failure.
      It carries a true 0.30 m pedestrian as a <b class="num bad">3–15 m</b> disk.</p>
      <pre class="code"><span class="cm"># talos.py — both guards take a max; nothing bounds it above</span>
if all(clip[name]):
    radius = max(radius, prev['r'])          <span class="cm"># hold outright</span>
else:
    radius = max(radius, 0.99 * prev['r'])   <span class="cm"># ≤1 %/frame release</span></pre>
      <p>One bad frame latches and the estimate ratchets upward for the whole run. ORCA then sees a
      neighbour overlapping the robot, takes its already-colliding branch, and commands a full-speed
      escape — <code>|v_cmd|</code> sits at the 1.5 m/s cap for 97 % of the run.</p>""",
      fig="qf_perception",
      take=("fix", "Fix is a ceiling: <code>radius = min(radius, 0.6)</code>. The "
            "&ldquo;never under-estimate&rdquo; instinct is right; it just needs a bound.")),

 dict(n=15, arc="the resolution", qid="", head="What we would change",
      sub="In priority order",
      body="""<ol class="rec">
      <li><strong>The bridge is too slow for the crowd's speed.</strong> Either slow the crowd to a speed
      the walker can respond to, or shorten the reaction path — react at the MPC rate rather than once per
      0.8 s step, and shrink the one-step commitment. <em>This is the real research problem the project
      uncovered.</em></li>
      <li><strong>Set <code>desired_speed ≈ 0.6 m/s</code></strong> (Q-B2). Free.</li>
      <li><strong>Bound the perceived radius</strong> (Q-F). One clamp turns a 10–30× over-estimate into a
      working perception path.</li>
      <li><strong>Give pedestrians contact</strong> so a collision has a physical consequence and the
      balance controller is actually tested by one.</li></ol>
      <h4>Limitations we are not hiding</h4>
      <ul class="lim">
      <li>Pedestrians are ghosts — in the blind runs the robot passes <em>through</em> people without ever
      losing balance. With contact enabled the blind condition would look worse still.</li>
      <li>The shipped crowd avoids the robot using <em>ground-truth</em> robot state — more cooperative and
      better informed than any real crowd. Q-G quantifies how much that flatters us: all of it.</li>
      <li>The blind condition is one extreme and the shipped cooperative crowd the other; Q-G2 measures
      the middle, but a clean robot-only isolation (per-neighbour responsibility) was left unbuilt.</li>
      <li>Kinematic disks, not gait-driven humans. One robot, one goal, no static obstacles.</li></ul>""",
      fig=None, take=None),
]

# number slides by position, so inserting one never desyncs the badges/nav
for _i, _s in enumerate(SLIDES):
    _s["n"] = _i + 1

def slide_html(s):
    parts = [f'<section class="slide" id="s{s["n"]}">']
    parts.append('<div class="rail"><span class="arc">%s</span><span class="num-badge">%02d</span></div>'
                 % (html.escape(s["arc"]), s["n"]))
    parts.append('<div class="content">')
    if s["qid"]:
        parts.append(f'<div class="eyebrow">{html.escape(s["qid"])}</div>')
    parts.append(f'<h2>{s["head"]}</h2>')
    parts.append(f'<p class="sub">{s["sub"]}</p>')
    parts.append(f'<div class="body">{s["body"]}</div>')
    if s.get("gifs"):
        cells = "".join(
            f'<figure class="gcell"><img src="{gif(scen)}" alt="{html.escape(title)}">'
            f'</figure>' for scen, title in GIF_SCENS)
        parts.append(f'<div class="gifgrid">{cells}</div>')
    if s.get("ensemble"):
        parts.append(f'<figure><img src="{gif_raw("ensemble_all.gif")}" '
                     f'alt="827 runs overlaid, coloured by outcome"></figure>')
    for key in ("fig", "fig2"):
        if s.get(key):
            parts.append(f'<figure><img src="{img(s[key])}" alt="{html.escape(s["qid"] or s["head"])} figure" loading="lazy"></figure>')
    if s.get("take"):
        kind, text = s["take"]
        parts.append(f'<div class="take {kind}">{text}</div>')
    parts.append('</div></section>')
    return "\n".join(parts)

nav = "\n".join(
    f'<a href="#s{s["n"]}"><span class="n">{s["n"]:02d}</span>{html.escape(s["qid"] or "·")}</a>'
    for s in SLIDES)

doc = f"""<title>Crowd Navigation for Talos — Evaluation</title>
<style>
:root {{
  --ground:#f2f4f5; --surface:#ffffff; --ink:#12171b; --ink2:#5b666e;
  --accent:#2a78d6; --alert:#cc3b3a; --good:#0d7a4f; --amber:#a8690b;
  --rule:#dde2e5; --plate:#ffffff;
  --serif: Palatino,"Palatino Linotype","Iowan Old Style","Book Antiqua",Georgia,serif;
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono: ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --ground:#13181c; --surface:#1a2126; --ink:#eef2f4; --ink2:#95a2ab;
    --accent:#5c9ee6; --alert:#e8706e; --good:#3fae7f; --amber:#d69a3c;
    --rule:#2b343a; --plate:#f6f7f8;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#13181c; --surface:#1a2126; --ink:#eef2f4; --ink2:#95a2ab;
  --accent:#5c9ee6; --alert:#e8706e; --good:#3fae7f; --amber:#d69a3c;
  --rule:#2b343a; --plate:#f6f7f8;
}}
:root[data-theme="light"] {{
  --ground:#f2f4f5; --surface:#ffffff; --ink:#12171b; --ink2:#5b666e;
  --accent:#2a78d6; --alert:#cc3b3a; --good:#0d7a4f; --amber:#a8690b;
  --rule:#dde2e5; --plate:#ffffff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 20px 96px; }}

header.top {{ position:sticky; top:0; z-index:20; background:color-mix(in srgb, var(--ground) 88%, transparent);
  backdrop-filter:blur(10px); border-bottom:1px solid var(--rule); }}
.topin {{ max-width:1180px; margin:0 auto; padding:10px 20px; display:flex; align-items:center;
  gap:18px; flex-wrap:wrap; }}
.brand {{ font-family:var(--serif); font-size:15px; letter-spacing:.01em; white-space:nowrap; }}
.brand b {{ font-weight:600; }}
nav {{ display:flex; gap:2px; flex-wrap:wrap; overflow-x:auto; }}
nav a {{ display:flex; align-items:baseline; gap:5px; padding:4px 8px; border-radius:5px;
  text-decoration:none; color:var(--ink2); font-size:12px; font-family:var(--mono);
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
nav a:hover {{ background:var(--surface); color:var(--accent); }}
nav a .n {{ opacity:.5; font-size:10px; }}
nav a:focus-visible, .wrap a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

.slide {{ display:grid; grid-template-columns:76px 1fr; gap:0; background:var(--surface);
  border:1px solid var(--rule); border-radius:10px; margin:26px 0; overflow:hidden; }}
.rail {{ border-right:1px solid var(--rule); padding:26px 0 20px; display:flex; flex-direction:column;
  align-items:center; gap:14px; }}
.rail .arc {{ writing-mode:vertical-rl; transform:rotate(180deg); font-size:10.5px;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ink2); }}
.rail .num-badge {{ font-family:var(--mono); font-size:12px; color:var(--ink2);
  font-variant-numeric:tabular-nums; }}
#s11 .rail {{ background:color-mix(in srgb, var(--alert) 9%, transparent); }}
#s11 .rail .arc {{ color:var(--alert); font-weight:600; }}
.content {{ padding:30px 34px 34px; min-width:0; }}

.eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.1em; color:var(--accent);
  text-transform:uppercase; margin-bottom:8px; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:clamp(24px,3.1vw,33px); line-height:1.2;
  margin:0 0 6px; text-wrap:balance; letter-spacing:-.005em; }}
.sub {{ margin:0 0 20px; color:var(--ink2); font-size:15px; }}
.body p {{ margin:0 0 13px; max-width:74ch; }}
.body ul, .body ol {{ margin:0 0 13px; padding-left:20px; }}
.body li {{ margin-bottom:6px; }}
h4 {{ font-family:var(--sans); font-size:12px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink2); margin:0 0 9px; font-weight:600; }}
.muted {{ color:var(--ink2); }}
.note {{ font-size:14.5px; color:var(--ink2); border-left:2px solid var(--rule); padding-left:13px; }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.bad {{ color:var(--alert); }} .ok {{ color:var(--good); }}
code {{ font-family:var(--mono); font-size:.9em; background:color-mix(in srgb, var(--ink) 7%, transparent);
  padding:1px 5px; border-radius:4px; }}

.split {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:26px; margin-bottom:14px; }}
pre.arch, pre.code {{ font-family:var(--mono); font-size:13px; line-height:1.65; overflow-x:auto;
  background:color-mix(in srgb, var(--ink) 5%, transparent); border:1px solid var(--rule);
  border-radius:7px; padding:14px 16px; margin:0 0 14px; }}
pre .cm {{ color:var(--ink2); }}

table {{ border-collapse:collapse; width:100%; margin:0 0 15px; font-size:14.5px; }}
table.q td {{ padding:7px 12px 7px 0; border-bottom:1px solid var(--rule); vertical-align:top; }}
table.q .qid {{ font-family:var(--mono); color:var(--accent); white-space:nowrap; width:56px; font-size:13px; }}
table.q tr.added td {{ background:color-mix(in srgb, var(--alert) 7%, transparent); }}
table.q tr.added .qid {{ color:var(--alert); font-weight:600; }}
table.data {{ font-variant-numeric:tabular-nums; }}
table.data td, table.data th {{ padding:7px 14px 7px 0; border-bottom:1px solid var(--rule); text-align:left; }}
table.data th {{ font-size:11.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--ink2); font-weight:600; }}
table.data tr.best td {{ background:color-mix(in srgb, var(--good) 10%, transparent); font-weight:600; }}
.tablewrap {{ overflow-x:auto; }}

figure {{ margin:20px 0 0; background:var(--plate); border:1px solid var(--rule); border-radius:8px;
  padding:9px; overflow-x:auto; }}
figure img {{ display:block; width:100%; max-width:100%; height:auto; border-radius:3px; }}
.gifgrid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:18px; }}
.gifgrid .gcell {{ margin:0; padding:6px; background:var(--plate); border:1px solid var(--rule);
  border-radius:8px; overflow:hidden; }}
.gifgrid .gcell img {{ width:100%; height:auto; display:block; border-radius:3px; }}
@media (max-width:720px) {{ .gifgrid {{ grid-template-columns:repeat(2,1fr); }} }}

.take {{ margin-top:20px; padding:13px 17px; border-radius:7px; font-size:15px;
  border-left:3px solid var(--accent); background:color-mix(in srgb, var(--accent) 8%, transparent); }}
.take.warn {{ border-color:var(--amber); background:color-mix(in srgb, var(--amber) 10%, transparent); }}
.take.bad {{ border-color:var(--alert); background:color-mix(in srgb, var(--alert) 10%, transparent); }}
.take.fix {{ border-color:var(--good); background:color-mix(in srgb, var(--good) 10%, transparent); }}
.take.ok {{ border-color:var(--accent); }}
ol.rec li {{ margin-bottom:10px; }}
ul.lim li {{ color:var(--ink2); }}

footer {{ color:var(--ink2); font-size:13.5px; text-align:center; padding:34px 20px 0;
  border-top:1px solid var(--rule); margin-top:34px; }}
@media (max-width:640px) {{
  .slide {{ grid-template-columns:1fr; }}
  .rail {{ flex-direction:row; justify-content:space-between; padding:11px 18px;
    border-right:0; border-bottom:1px solid var(--rule); }}
  .rail .arc {{ writing-mode:horizontal-tb; transform:none; }}
  .content {{ padding:22px 20px 26px; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ scroll-behavior:auto !important; }} }}
html {{ scroll-behavior:smooth; }}
</style>

<header class="top"><div class="topin">
<span class="brand"><b>Crowd Navigation for Talos</b> — Evaluation</span>
<nav>{nav}</nav>
</div></header>

<div class="wrap">
{chr(10).join(slide_html(s) for s in SLIDES)}
<footer>800+ simulation runs · MuJoCo + ORCA + LIP-MPC/TSID · figures and tables regenerate from
<code>experiments/analyze.py</code></footer>
</div>
"""

with open(OUT, "w") as f:
    f.write(doc)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
