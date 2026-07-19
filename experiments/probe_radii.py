"""Print the perceived obstacle radii over a short run, against the true 0.3 m."""
import numpy as np
import bullet_sims.walking as w

w.RENDER = False
w.PEDESTRIAN_SENSOR_DATA = True
w.SCENARIO = "random"
w.N_PEDS = 12
w.SEED = 1
w.WALK_DURATION = 45.0

_orig = w._run_loop
def patched(simulator, crowd, robot, *a, **kw):
    real_update = robot.update_obstacles
    hist = []
    def spy():
        real_update()
        obs = robot.obstacles2d()
        if obs:
            hist.append([o['r'] for o in obs])
        if len(hist) % 30 == 1 and obs:
            rs = np.array([o['r'] for o in obs])
            com = simulator.computeComPosition(robot.robot.id())[:2]
            gaps = [np.hypot(*(com - np.array([o['x'], o['y']]))) - 0.3 - o['r']
                    for o in obs]
            k = int(np.argmin(gaps))
            print(f"[probe] n={len(rs)} r_max={rs.max():.2f} | worst believed gap="
                  f"{min(gaps):+.2f} m from track r={obs[k]['r']:.2f} "
                  f"at d={np.hypot(*(com - np.array([obs[k]['x'], obs[k]['y']]))):.2f} m "
                  f"stale={obs[k].get('stale')}", flush=True)
    robot.update_obstacles = spy
    try:
        return _orig(simulator, crowd, robot, *a, **kw)
    finally:
        if hist:
            allr = np.concatenate([np.array(h) for h in hist])
            print(f"\n[probe] OVERALL perceived radius: min={allr.min():.2f} "
                  f"med={np.median(allr):.2f} p95={np.percentile(allr,95):.2f} "
                  f"max={allr.max():.2f}  |  true = 0.30 m", flush=True)
w._run_loop = patched
w.main()
