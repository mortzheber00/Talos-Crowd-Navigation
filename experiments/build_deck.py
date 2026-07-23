"""Generate the evaluation slide deck with figures inlined as data URIs.

Paths are derived from this file's location so the deck rebuilds from any
checkout. WEBP holds convert(1)-compressed copies of experiments/figures/*.png
(regenerate them with experiments/regenerate.sh). Override the output path with
the DECK_OUT environment variable."""
import base64, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
WEBP = os.path.join(FIGS, "webp")
GIFS = os.path.join(FIGS, "gifs")
OUT = os.environ.get("DECK_OUT", os.path.join(HERE, "deck.html"))

def _datauri(path, mime):
    """Inline a file as a data URI, or None if it has not been generated yet
    (partial-data previews skip the missing figure instead of crashing)."""
    if not os.path.exists(path):
        print(f"  [deck] missing, skipped: {os.path.relpath(path, HERE)}")
        return None
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def img(name):
    return _datauri(os.path.join(WEBP, name + ".webp"), "image/webp")

def gif(name):
    return _datauri(os.path.join(GIFS, "nav_" + name + ".gif"), "image/gif")

def gif_raw(fname):
    return _datauri(os.path.join(GIFS, fname), "image/gif")

GIF_SCENS = [("empty", "Empty (control)"), ("head_on", "Head-on"),
             ("crossing", "Crossing"), ("corridor", "Corridor"),
             ("circle", "Circle"), ("random", "Random")]

SLIDES = [
 dict(arc="method", qid="Methodology",
      head="How the crowd-navigation stack was evaluated",
      sub="Headless sweeps in MuJoCo — ground truth vs. onboard perception, metrics straight to disk",
      body="""<div class="split">
      <div><h4>What we ran</h4><ul>
      <li>Six repeatable crowd scenarios, robot walking origin&nbsp;&rarr;&nbsp;(4,&nbsp;0)&nbsp;m.</li>
      <li><b class="num">750</b> runs behind slides&nbsp;1&amp;&nbsp;3: every scenario under
      <strong>ground truth</strong> and the <strong>onboard depth camera</strong> (375 each),
      random-heavy since it is the only scenario whose layout the seed changes.</li>
      <li>A <b class="num">360</b>-run responsibility sweep behind slide&nbsp;4 (the failure boundary).</li>
      <li>Headless, 10 runs at once; each with its own <code>ROS_DOMAIN_ID</code> — without it,
      parallel runs share a DDS domain and one run's ORCA steers on another's odometry.</li>
      <li>Wilson 95&nbsp;% intervals for rates; Wilcoxon signed-rank for paired tests.</li></ul></div>
      <div><h4>Two decisions that fix what the numbers mean</h4><ul>
      <li><strong>Clearance is always ground truth</strong>, never what the robot perceived — else an
      undetected pedestrian silently vanishes from the safety statistics.</li>
      <li><strong>Post-fall samples are excluded</strong> from collisions — a collapsed robot drifting
      into someone is not an avoidance failure.</li>
      <li>Ground truth vs. perception is a <em>topic-name choice</em>, not a separate code path: the ORCA
      node just subscribes to a different obstacle topic.</li></ul></div></div>
      <p style="margin-top:14px">Every scenario run overlaid, robot path coloured by outcome —
      <strong>onboard perception</strong> on top, <strong>ground truth</strong> below. Same crowds, two
      sensing conditions, the whole evaluation in two frames:</p>""",
      ensembles=[("ensemble_perception.gif",
                  "Onboard depth camera — 375 runs overlaid"),
                 ("ensemble_all.gif",
                  "Ground truth — 375 runs overlaid")],
      ensemble=False, fig=None, fig2=None, take=None, gifs=False),

 dict(arc="scenarios", qid="The scenarios",
      head="Six crowds, from an empty hall to a packed square",
      sub="Kinematic ORCA pedestrians (r = 0.3 m, up to 1.3 m/s) that reciprocally avoid each other and the robot",
      body="""<div class="tablewrap"><table class="data"><tbody>
      <tr><td><strong>Empty</strong></td><td class="muted">no pedestrians — the gait/​control floor</td></tr>
      <tr><td><strong>Head-on</strong></td><td class="muted">two pedestrians walking straight at the robot</td></tr>
      <tr><td><strong>Crossing</strong></td><td class="muted">six pedestrians cutting across the robot's route</td></tr>
      <tr><td><strong>Corridor</strong></td><td class="muted">two opposing files at <em>mixed</em> speeds — some faster than the robot can walk, some at its pace</td></tr>
      <tr><td><strong>Circle</strong></td><td class="muted">eight pedestrians crossing to opposite points — a near-symmetric squeeze</td></tr>
      <tr><td><strong>Random</strong></td><td class="muted">ten pedestrians, a share of them routed onto the robot's own goal so the destination is contested</td></tr>
      </tbody></table></div>
      <p>All six on a shared 18&nbsp;s clock, robot in <span style="color:#0d7a4f;font-weight:600">green</span>,
      pedestrians grey, goal the red star:</p>""",
      gifs=True, ensemble=False, fig=None, fig2=None, take=None),

 dict(arc="results", qid="Collision results",
      head="Collision-free almost everywhere — perception included",
      sub="Two questions per scenario, ground truth vs. the onboard depth camera: did it stay safe, and did it arrive in time?",
      body="""<p><strong>Safety is solved.</strong> <b class="num">Zero collisions</b> across all 1&thinsp;500
      scenario runs — every scenario, both sensors, 100&nbsp;% collision-free (left). At nominal cooperation
      the crowd always yields, so the robot simply never hits anyone.</p>
      <p><strong>The cost is liveness — reaching the goal in time (right).</strong> Trivial on the simple
      crowds; the depth camera shaves at most a run in twenty. But on the contested-goal <strong>Random</strong>
      square, arrivals within 45&nbsp;s drop from <b class="num">89&nbsp;%</b> on ground truth to
      <b class="num bad">56&nbsp;%</b> with the camera — a dense crowd converging on the goal is where merged
      camera tracks make the robot hesitate. The failure is a <em>time</em> cost, not a safety one.</p>""",
      fig="slide3_collisions", ensemble=False, gifs=False, fig2=None,
      take=("warn", "Safe everywhere, but slow in the dense contested crowd. Slide&nbsp;4 opens that scenario "
            "up — and shows where safety <em>itself</em> finally breaks.")),

 dict(arc="failure", qid="Where it fails",
      head="It fails when the crowd stops yielding — not when the robot does",
      sub="Robot's avoidance share × crowd's share, zoomed into the low-cooperation corner (Random, 24 peds)",
      body="""<p>The nominal crowd is collision-free because <em>both</em> sides give way. To find the edge we
      shrink each side's ORCA share toward zero and watch where safety breaks. Everything is already
      collision-free once each side takes even ~0.1, so the whole gradient lives in
      <code>[0,&nbsp;0.20]</code> — this sweep zooms there at 0.04 resolution.</p>
      <p><strong>The failure is a column, not a diagonal.</strong> Collisions appear only as the
      <em>crowd's</em> share &rarr; 0, and the robot's own share barely moves that line: a fully-committed
      robot is still out-run by a crowd that refuses to yield. <strong>Safety follows the crowd; the robot's
      share buys liveness (time-to-goal), not clearance.</strong></p>
      <p class="note">This is the bridge from slide&nbsp;3: the 100&nbsp;%s there are not the robot solving a
      hard problem alone — they hold because the crowd cooperates. Take that away and the safety margin goes
      with it.</p>""",
      fig="slide4_failure_boundary", ensemble=False, gifs=False, fig2=None,
      take=None),
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
            f'<figure class="gcell"><img src="{uri}" alt="{html.escape(title)}">'
            f'<figcaption>{html.escape(title)}</figcaption></figure>'
            for scen, title in GIF_SCENS if (uri := gif(scen)))
        parts.append(f'<div class="gifgrid">{cells}</div>')
    # a stack of labelled ensemble GIFs (e.g. perception above ground truth)
    for fname, cap in s.get("ensembles", []):
        uri = gif_raw(fname)
        if uri:
            parts.append(f'<figure><img src="{uri}" alt="{html.escape(cap)}">'
                         f'<figcaption>{cap}</figcaption></figure>')
    if s.get("ensemble") and (uri := gif_raw("ensemble_all.gif")):
        parts.append(f'<figure><img src="{uri}" '
                     f'alt="ground-truth runs overlaid, coloured by outcome"></figure>')
    if s.get("ensemble_file") and (uri := gif_raw(s["ensemble_file"])):
        cap = s.get("ensemble_caption", "")
        parts.append(f'<figure><img src="{uri}" '
                     f'alt="perception runs overlaid">'
                     + (f'<figcaption>{cap}</figcaption>' if cap else '')
                     + '</figure>')
    for key in ("fig", "fig2"):
        if s.get(key) and (uri := img(s[key])):
            parts.append(f'<figure><img src="{uri}" alt="{html.escape(s["qid"] or s["head"])} figure" loading="lazy"></figure>')
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
/* dense reference tables: the methodology matrix carries a lot of small facts,
   so it gets tighter rows and a smaller face than the result tables */
table.cfg td {{ padding:5px 10px 5px 0; font-size:13.5px; border-bottom:1px solid var(--rule); }}
table.cfg td:nth-child(odd) {{ color:var(--ink2); }}
table.matrix {{ font-size:12.5px; }}
table.matrix th, table.matrix td {{ padding:6px 10px 6px 0; vertical-align:top;
  border-bottom:1px solid var(--rule); }}
table.matrix th {{ font-size:10.5px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--ink2); white-space:nowrap; }}
table.matrix .qid {{ font-family:var(--mono); color:var(--accent); font-weight:600;
  white-space:nowrap; }}
table.matrix tr.added .qid {{ color:var(--alert); }}
table.matrix tr.added {{ background:color-mix(in srgb, var(--alert) 5%, transparent); }}
table.matrix code {{ font-size:11.5px; background:none; padding:0; color:var(--ink); }}

figure {{ margin:20px 0 0; background:var(--plate); border:1px solid var(--rule); border-radius:8px;
  padding:9px; overflow-x:auto; }}
figure img {{ display:block; width:100%; max-width:100%; height:auto; border-radius:3px; }}
figcaption {{ font-size:12.5px; color:var(--ink2); padding:8px 4px 2px; }}
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
<footer>MuJoCo + ORCA + LIP-MPC/TSID · six scenarios, ground truth vs. onboard perception ·
figures regenerate from <code>experiments/analyze.py</code></footer>
</div>
"""

with open(OUT, "w") as f:
    f.write(doc)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
