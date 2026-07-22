## Schnelle Überprüfung

```bash
ros2 launch bullet_sims depth_perception.launch.py
```
### Oder separat:

Starte in **drei separaten Terminals**:

### Terminal 1: Roboter
```bash
ros2 run bullet_sims stand_and_perceive
```

### Terminal 2: depth_image_proc
```bash
ros2 launch bullet_sims depth_image_proc.launch.py
```

# Überprüfe die Frequenz der Nachrichten
ros2 topic hz /depth_image_rect
ros2 topic hz /depth_camera_info
ros2 topic hz /depth/points  # Sollte Pointcloud-Frequenz zeigen

# Inspiziere den Node
ros2 node info /point_cloud_xyz

## Mögliche Probleme & Lösungen

### 1. **Topics existieren nicht** 
Wenn `/depth_image_rect` oder `/depth_camera_info` nicht existieren:
```bash
# Überprüfe, ob der Roboter wirklich publiziert
ros2 topic list | head -20
```

**Lösung:** In `talos.py`, überprüfe dass `publish_depth_image()` in der main-loop aufgerufen wird.

### 2. **Topics existieren, aber Pointcloud wird nicht generiert**
Das Problem ist wahrscheinlich die **message_filters Synchronisierung**.

**Überprüfe die Timestamps:**
```bash
# Schaue dir die Header-Timestamps an
ros2 topic echo --once /depth_image_rect
ros2 topic echo --once /depth_camera_info
# Die `stamp` Werte sollten identisch sein!
```

**Wenn Timestamps unterschiedlich sind:**
- Das ist das Problem! Die Nachrichten müssen **exakt denselben Timestamp** haben
- Das sollte durch die aktuelle Implementierung behoben sein (beide nutzen `self.ros_node.get_clock().now()`)

### 3. **Node startet, gibt keine Fehler, aber publiziert nicht**
Starte den Node mit DEBUG-Logging:
```bash
ros2 run depth_image_proc point_cloud_xyz_node --ros-args --log-level debug
```

### 4. **Alternative: Andere depth_image_proc Node verwenden**
Wenn `point_cloud_xyz_node` nicht funktioniert, probiere `point_cloud_xyz_radial_node`:
```bash
ros2 run depth_image_proc point_cloud_xyz_radial_node \
  --ros-args \
  -r depth_image_rect:=/depth_image_rect \
  -r depth_camera_info:=/depth_camera_info \
  -r points:=/depth/points
```

### 5. **Überprüfe den Depth-Wert Bereich**
Das Tiefenbild muss **Meter-Einheiten** haben (nicht Pixel oder raw sensor values):
```python
import rclpy
from sensor_msgs.msg import Image
import numpy as np

def callback(msg):
    data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    print(f"Depth min: {data.min()}, max: {data.max()}")  # Should be in meters

rclpy.init()
node = rclpy.create_node('test')
node.create_subscription(Image, '/depth_image_rect', callback, 1)
rclpy.spin(node)
```

## Vollständiger manueller Test

```bash
# Terminal 1: Roboter
ros2 run bullet_sims walking

# Terminal 2: point_cloud_xyz direkt mit Remappings
ros2 run depth_image_proc point_cloud_xyz_node \
  -r depth_image_rect:=/depth_image_rect \
  -r depth_camera_info:=/depth_camera_info \
  -r points:=/depth/points

# Terminal 3: Überprüfe Topics
ros2 topic list
ros2 topic echo /depth/points --once
```

## Wenn nichts hilft

Überprüfe die launch.py Syntax:
```bash
python3 /workspaces/talos_crowd_navigation/src/bullet_sims/launch/depth_image_proc.launch.py
```

Und verifiziere, dass `depth_image_proc` installiert ist:
```bash
dpkg -l | grep depth_image_proc
# oder
ros2 pkg list | grep depth
```

## Logs ansehen
```bash
# Wenn Nodes fehlschlagen, schau die Logs an
tail -f ~/.ros/log/latest/*/stderr.log
```

# Segmentation-basierte 2D-Hindernis-Wahrnehmung

## Überblick

Eine schlanke Pipeline für lokale ORCA-Navigation, die die bestehende MuJoCo-Tiefenkamera um ein deckungsgleiches Segmentation-Rendering erweitert. Pro Zylinder werden Position (x, y), Radius r und Geschwindigkeit (vx, vy) in Weltkoordinaten in Echtzeit extrahiert — ohne 3D-Clustering, nur NumPy-Geometrie auf Pixel-Masken.

## Architektur

```
MuJoCo Scene (TALOS + 5 Crowd-Zylinder als Mocap)
    ↓
Renderer: one update_scene() → depth + seg (pixel-aligned)
    ↓
Talos.update_obstacles() [30 Hz]
  • Segmentation-Maskierung pro Geom-ID
  • Pinhole-Rückprojektion (K-Matrix)
  • Welt-Transformation (MuJoCo→ROS Optical Frame)
  • Schwerpunkt + Radius + EMA-Geschwindigkeit
    ↓
obstacles2d() → [{id, x, y, r, vx, vy, stale, t}]
    ↓
RViz Marker + ORCA (Zukunft)
```

## Core-Mathematik

### Segmentation-Maskierung
```python
mask = (seg[...,0] == geom_id) & (seg[...,1] == mjOBJ_GEOM) & isfinite(depth)
```

### Pinhole-Rückprojektion (Optical Frame)
```python
X = (u - cx) * z / fx   # +X: rechts
Y = (v - cy) * z / fy   # +Y: unten
Z = z                   # Tiefenwert
```

### Frame-Konversion: MuJoCo → ROS
```python
# MuJoCo Kamera: blickt entlang -Z, +Y oben
# ROS Optical: blickt entlang +Z, +Y unten
R_w_opt = R_wc @ diag([1, -1, -1])
pts_w = (R_w_opt @ [X, Y, Z].T).T + cam_pos
```

### Frontflächen-Bias-Korrektur (pro Kamera)
```python
# WICHTIG: pro Kamera, nicht auf der gemergten Wolke (siehe "Warum NICHT naiv
# mergen"). Kamera sieht nur den vorderen Bogen; Centroid liegt ~2r/π vor der
# echten Achse entlang des Sichtstrahls.
ray    = normalized(centroid - cam_pos_xy)            # cam_pos DIESER Kamera
r_est  = (perc(lateral, 97.5) - perc(lateral, 2.5)) / 2   # robuster Chord
center = centroid + (2/π) * r_est * ray
cover  = perc(||xy - center||, 97.5)                  # robuste Abdeckung
r_view = max(r_est, cover)                            # Radius aus DIESER Sicht
# Fusion + Padding + zeitlicher Guard: siehe Datenfluss unten
```

### Woher kommt der Faktor $\frac{2r}{\pi}$? 
Das lässt sich sauber mit einer kleinen Integration herleiten.
Nehmen wir an, der echte Mittelpunkt des Zylinders liegt im Ursprung $(0,0)$ und die Kamera schaut von der positiven X-Achse genau darauf.

Die Kamera sieht nun einen Halbkreis-Bogen mit dem Radius $r$.
Dieser Bogen lässt sich über den Winkel $\theta$ von $-\frac{\pi}{2}$ bis $+\frac{\pi}{2}$ beschreiben.Für jeden Punkt auf diesem Bogen gilt für die X-Koordinate:$$x = r \cdot \cos(\theta)$$

Der Schwerpunkt $\bar{x}$ einer Kurve ist definiert als die Summe aller X-Werte multipliziert mit dem Linienelement $ds$, geteilt durch die Gesamtlänge der Kurve $L$.

Die Gesamtlänge des Halbkreises ($L$):
$$L = r \cdot \pi$$

Das Integral für die X-Koordinaten:
Da ein kleines Stück des Bogens $ds = r \cdot d\theta$ ist, integrieren wir über den sichtbaren Bereich:
$$\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} x \, 
ds = \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} (r \cdot \cos(\theta)) \cdot (r \, d\theta) = r^2 \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos(\theta) \, 
d\theta$$Integrieren von $\cos(\theta)$:

Das Integral von $\cos(\theta)$ von $-\frac{\pi}{2}$ bis $\frac{\pi}{2}$ ist genau $2$ (denn $\sin(\frac{\pi}{2}) - \sin(-\frac{\pi}{2}) = 1 - (-1) = 2$).

Damit ergibt sich für den Zähler:$$2r^2$$

Berechnung des Schwerpunkts ($\bar{x}$):
Jetzt teilen wir das Ergebnis durch die Gesamtlänge $L$:$$\bar{x} = \frac{2r^2}{r \cdot \pi} = \frac{2r}{\pi}$$

### Geschwindigkeit mit EMA (= Exponential Moving Average)
```python
# Finite Differenz über echte Sim-Zeit (nicht assume 30 Hz)
dt = t_now - t_last
v_raw = np.clip((center_now - center_last) / dt, -3, 3)  # 3 m/s Clamp

# Exponential Moving Average (α = 0.3)
v_smooth = α * v_raw + (1 - α) * v_prev
```

## Fehlerquellen & Robustness-Guards

| Problem | Grund | Lösung |
|---------|-------|--------|
| **Frontflächen-Bias** | Nur vorderer Bogen sichtbar, Schwerpunkt sitzt vorn | 2r/π-Korrektion entlang Sichtstrahl (pro Kamera) |
| **Naht-Inflation** | Merge zweier Blickwinkel → breiter Bogen → `center` daneben → Radius ~2× | Pro Kamera schätzen, dann fusionieren (nicht `vstack`-fitten) |
| **FOV-Randclipping** | Seitlicher Zylinderumriss abgeschnitten → verzerrter Chord | Geclippte Sicht verwerfen, solange eine saubere Sicht existiert |
| **Occluder-Sichel-Spike** | Verdeckungs-Fragment → Streupunkte weit von Achse | Robuste Perzentile (2.5/97.5) statt roher min/max |
| **Verdeckungs-Kollaps / Spike-Ratchet** | Dünne Maske → r springt hoch *oder* runter | Symmetrisches ±1%/Frame-Clamp (starrer Körper) |
| **Wiederaufnahme-Sprung** | Track re-emerges an anderem Ort → v_raw explodiert | `v_raw = clip(v_raw, ±3 m/s)` |
| **Sensor-Rauschen** | Einzelne Pixel verschieben Centroid | EMA (Position α=0.5, Velocity α=0.15) dämpft |
| **Leere Maske** | Vollständige Verdeckung | `stale=True`, letzten Wert halten, nach 1 s droppen |

## Performance (Verifizierung headless, `MUJOCO_GL=egl`)

| Metrik | Statisch | Bewegt |
|--------|----------|---------|
| **Positionsfehler** | Ø 4.7 cm | Ø 8.8 cm (RMSE 10.7 cm) |
| **Radius** | 0.265–0.274 m (Ziel: 0.275) | Ø 0.256 m |
| **Geschwindigkeit** | 0 m/s (korrekt) | Ø 0.11 m/s Error bei Ø 0.36 m/s GT |
| **Seg/Depth-Alignment** | 0/9.1M Pixel inkonsistent | 0/28.6M Pixel |
| **Update-Rate** | 30 Hz (ROS) | 30 Hz |

→ **Aussage:** Robot-tauglich für ORCA (~5–11 cm Fehler ist akzeptabel für lokale Navigation mit ~25 cm Hindernissen).

## Warum Fehler trotz Ground Truth?

Ground Truth gibt die echte Zylinderachse im Weltframe. Der **Sensor kann die Achse direkt nicht messen** — er misst nur Oberflächenpunkte. Das ist geometrischer Fehler, nicht Mess-Rauschen:

```
True axis:        Sensor sees (front arc):
    ◯ (x,y)       | ◯  
   d=0.5m         | visible surface
                  → centroid_meas ≠ axis
                  → bias ≈ 2r/π in direction-to-camera
```

Die Korrektur mit `(2/π)*r_est*ray` fängt ~80 % dieses Bias ab. Der Rest (~1–2 cm) kommt von:
- Radius-Unsicherheit (wir schätzen r aus Sehne, nicht echte r)
- Pixelisierungs-Rauschen (Schwerpunkt auf diskretem Gitter)
- Bodenebenen-Projektion (kleine Fehler beim z→0 setzen)

## Implementierungs-Highlights

### MujocoWrapper
- **`get_depth_and_seg_image()`**: Ein Renderer, ein `update_scene()`, zwei `render()`-Aufrufe mit Mode-Switch → garantiert pixel-deckungsgleich
- **`get_camera_pose()`**: Kamera-Weltpose aus `data.cam_xpos`/`cam_xmat`
- **`getGeomId(name)`**: Stabile Name→ID-Auflösung pro Frame (recompile-sicher)

### CrowdEnv (neues Modul)
- 5 Zylinder als **Mocap-Bodies** (`body.mocap=True`, `contype=0`)
- Scripted motion: Mixtur aus linearen Pendeln (Sinus) + Kreisbahnen (2–5 m entfernt)
- `ground_truth(dt)` liefert exakte Referenzwerte zur Validierung

### Talos
- **`set_obstacle_geoms(names)`**: Tracker-Initialisierung
- **`update_obstacles()`**: Kern-Pipeline (segmentation → unprojection → tracking)
- **`obstacles2d()`**: Accessor für ORCA (gibt `[{id, x, y, r, vx, vy, stale}]` zurück)
- **`publish_obstacle_markers(ground_truth)`**: RViz-Overlay (Schätzung vs. Ground Truth)

## Integration in stand_and_perceive.py

```python
# Init
crowd = CrowdEnv(simulator)
robot.set_obstacle_geoms(crowd.geom_names)

# Main loop
while running:
    # Physics step (1 kHz)
    crowd.update(simulator.simTime())
    simulator.step()
    
    # Perception (30 Hz)
    if time_to_publish:
        robot.update_obstacles()
        robot.publish_obstacle_markers(crowd.ground_truth())
        # → obstacles2d() ready für ORCA
```

## Erweiterung (2026-07-16): 4-Kamera-Rundumsicht (360°)

Die oben beschriebene Pipeline lief ursprünglich auf der einen Front-RGB-D-Kamera
(`rgbd_depth_cam`, deckt sich mit dem URDF-`rgbd_optical_frame`). Für ORCA reicht
ein ~90°-Sichtfeld nach vorne aber nicht — ein Fußgänger seitlich oder hinter dem
Roboter bleibt unsichtbar. Die Pipeline wurde daher um vier zusätzliche
Weitwinkel-Kameras erweitert, die zusammen 360° abdecken.

### Montage

```python
percep_fovy, percep_w, percep_h = 90.0, 320, 240   # fovy=90° @ 4:3 -> ~106° horizontal
percep_radial = 0.15                                # [m] Versatz entlang Blickrichtung

for k in range(4):
    yaw = k * np.pi / 2.0                           # 0, 90, 180, 270 deg
    sim.addCameraToLink(
        'head_2_link', name=f'percep_cam_{k}',
        pos=[percep_radial*cos(yaw), percep_radial*sin(yaw), camera_pos[2]],
        orientation=None, yaw=yaw,
        fovy=percep_fovy, width=percep_w, height=percep_h)
```

- Alle 4 Kameras hängen am selben Link (`head_2_link`) wie die Frontkamera, sind
  aber per `yaw` um die Kopf-Hochachse gedreht — `addCameraToLink` rotiert dafür
  die Standard-"schau nach vorne"-Orientierung um `Rz(yaw)`, bevor sie ins
  Body-Frame konvertiert wird.
- **106° statt 90° Rasterabstand**: Das horizontale FOV ist bei fovy=90° und
  4:3-Bildformat größer als der 90°-Versatz zwischen den Kameras → benachbarte
  Sichtfelder überlappen sich an den Nähten. Das ist bewusst so gewählt (siehe
  Clip-Guard unten).
- **0.15 m radialer Versatz**: Eine Kamera exakt im Kopfzentrum rendert 0 Pixel,
  weil das Kopf-Mesh die Sicht komplett verschluckt (nur die Frontkamera
  funktioniert dort, weil das Mesh nach vorne offen ist). Jede Surround-Kamera
  wird deshalb 0.15 m entlang ihrer eigenen Blickrichtung nach außen versetzt.
  Verifiziert: 6/6 Test-Fußgänger sichtbar, 6-8 cm Positionsfehler.
- Geringere Auflösung als die Frontkamera (320×240 statt der RGB-D-Auflösung),
  weil bei 30 Hz pro Frame jetzt 4× depth+seg gerendert wird — Kompromiss
  zwischen Sichtfeld und Renderkosten.
- Jede Kamera bekommt ihre eigene Pinhole-Intrinsik `K` (`_pinhole_K(fovy, w, h)`),
  da FOV/Auflösung von der Frontkamera abweichen.
- `rgbd_depth_cam` bleibt separat bestehen und beliefert weiterhin den echten
  ROS-Depth-Topic/Pointcloud; die 4 Surround-Kameras (`self._percep_cams`)
  werden ausschließlich intern von `update_obstacles()` genutzt und erzeugen
  keinen eigenen ROS-Topic.

### Datenfluss mit 4 Kameras (pro Perceptions-Tick, ~30 Hz)

```
for cam in [percep_cam_0 .. percep_cam_3]:
    depth, seg   = get_depth_and_seg_image(camera=cam)   # 1 Renderer, 1 update_scene()
    cam_pos, R_wc = get_camera_pose(camera=cam)           # DIESE Kamera-Pose
    R_w_opt = R_wc @ diag([1,-1,-1])                       # MuJoCo -> ROS Optical

    for jedes getrackte Geom:
        mask = segmentation-Maske für Geom-ID
        (u, v, Z) -> Pinhole-Rückprojektion mit cam-eigenem K
        pts_world = (R_w_opt @ [X,Y,Z]).T + cam_pos
        pts[geom_name].append(pts_world[:, :2])   # je Kamera SEPARAT gehalten
        cams_xy[geom_name].append(cam_pos[:2])    # + Mount DIESER Kamera

# nach der Kamera-Schleife, pro Geom: NICHT alle Punkte mergen und einmal fitten,
# sondern pro (sauberer, unclipped) Kamera schätzen -> dann fusionieren:
per_view = [front_arc_estimate(p, cxy) for p, cxy in saubere Sichten]
center   = punktzahl-gewichteter Mittelwert der per-view centers
r_cover  = max(per-view radii)          # jeder ~r-begrenzt -> keine Inflation
radius   = 1.15 * r_cover                # Padding
radius   = clip(radius, 0.99*prev_r, 1.01*prev_r)   # ±1%/Frame (starrer Körper)
```

### Warum NICHT naiv mergen: Radius-Inflation an den Nähten

Der naheliegende Ansatz — alle 4 Punktwolken per `np.vstack` in einen Topf werfen
und **einmal** die Front-Arc-Schätzung darauf rechnen — **funktioniert nicht**.
Er blähte den geschätzten Radius an Kamera-Nähten auf bis zu ~2× GT auf (0.3 m →
0.43–0.56 m). Der Grund ist geometrisch:

Die Front-Arc-Korrektur (2r/π entlang *eines* Sichtstrahls) setzt voraus, dass die
Punkte **einen** vorderen Bogen aus **einer** Blickrichtung bilden. Ein Fußgänger
an der Naht wird aber von zwei ~90° auseinander stehenden Kameras gesehen; die
gemergte Wolke spannt einen **breiteren Bogen**, als eine einzelne Sicht je zeigen
könnte. Dann liegt der Centroid nicht mehr 2r/π vor der Achse, der eine Strahl
passt zu keiner der beiden Sichten, `center` landet **neben** der echten Achse,
und der `max(||xy − center||)`-Term greift einen weit entfernten Punkt → Radius
explodiert. (Zusätzlich nutzte der naive Code `cam_pos` nur der *letzten* Kamera.)

**Der Fix: pro Kamera schätzen, dann fusionieren.**

1. **Pro Kamera** wird `_front_arc_estimate(pts_cam, cam_xy)` gerechnet — dort
   gilt das Ein-Blickrichtung-Modell exakt, mit der *eigenen* Kamerapose. Ergebnis
   je Sicht: ein Center und ein Radius.
2. **Fusion**: punktzahl-gewichteter Mittelwert der Center; der Radius ist das
   `max` der Einzelradien. Da jeder Einzelradius durch die Geometrie auf ~r
   begrenzt ist, ist das `max` sicher (nie Unterschätzung) **ohne** aufzublähen.
3. **Geclippte Sichten fliegen raus**, solange eine saubere (unclipped, ≥
   `MIN_OBSTACLE_PIXELS`) Sicht existiert: ein am Bildrand abgeschnittener Bogen
   liefert eine verzerrte Chord/Center-Schätzung. Das ist der typische Naht-Fall
   `clip=[True, False]` → nur die Kamera nutzen, die den ganzen Bogen sieht.
4. **Robuste Statistik** in der Schätzung: Chord und Abdeckung über Perzentile
   (2.5/97.5) statt roher min/max, damit Streupunkte einer Occluder-Sichel den
   Radius nicht hochziehen.
5. **Zeitlicher Guard**: der Radius darf sich pro Frame nur um ±1 % ändern
   (symmetrisch). Ein starrer Körper ändert seinen Radius nicht sprunghaft, also
   kann ein Ein-Frame-Ausreißer ihn weder kollabieren noch aufblähen. (Der frühere
   nur-Schrumpf-Guard `≥ 0.99·prev` *ratchete* Spikes ~0.5 s lang fest.)

**Weiterhin gilt:** Jede Kamera transformiert unabhängig korrekt ins Weltframe
(eigenes `K`, eigene Pose — kein gemeinsamer Fehler-Zwischenschritt), der 0.15 m
Montageversatz ist gegenüber Fußgänger-Abständen vernachlässigbar, und
`MIN_OBSTACLE_PIXELS` wird über alle 4 Kameras **summiert** (ein Track gilt erst
als verdeckt, wenn ihn *keine* Sicht mehr ausreichend zeigt).

Verifiziert (headless, `random`): max. Radius 0.561 → **0.425 m**, Anteil > 0.36 m
von **28.9 % → 0.5 %**, Ausreißer > 0.45 m von 80 → **0**, Naht-Fälle (2 Kameras)
im Mittel 0.383 → **0.308 m**; Roboter erreicht weiter das Ziel. Nahe Fußgänger
(kollisionsrelevant) lesen ≥ GT; nur ferne minimal darunter, wachsen aber bei
1 %/Frame rechtzeitig ein.

## Vor- und Nachteile

### ✅ Vorteile
- **Direkt**: Keine 3D-Clustering, keine Rausch-Akkumulation
- **Schnell**: 30 Hz auf CPU
- **Robust**: Alle Lookups per Name (recompile-sicher)
- **Präzise**: Segmentation pixelweise
- **Einfach**: K-Matrix ist kalibriert, EMA-Parameter explizit
- **Prüfbar**: Ground Truth sofort verfügbar

### ❌ Nachteile
- **Frontflächen-Bias**: ~1–5 cm systematischer Fehler (geometrisch)
- **FOV-Grenzen**: Randclipped Zylinder unterestimiert r
- **Verdeckung**: Sofort stale; verzögerte Wiederaufnahme
- **Bodenebenen-Annahme**: Funktioniert nur auf flacher Ebene
- **Rauschen-empfindlich**: Pixel-Rauschen → Centroid-Drift

## Nächste Schritte: ORCA-Integration

```python
# In einer künftigen orca.py:
obstacles = robot.obstacles2d()
for obs in obstacles:
    if obs['stale']:
        continue  # ignoring occluded tracks
    # Feed to RVO2 / ORCA solver:
    # position = (obs['x'], obs['y'])
    # radius = obs['r']
    # velocity = (obs['vx'], obs['vy'])
```

---