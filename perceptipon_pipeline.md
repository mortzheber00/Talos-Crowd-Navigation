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

### Frontflächen-Bias-Korrektur
```python
# Kamera sieht nur vordere Bogenoberfläche
# Centroid liegt ~2r/π vor der echten Achse
ray = normalized(centroid - cam_pos_xy)
r_est = (lateral_max - lateral_min) / 2
center_true = centroid + (2/π) * r_est * ray
radius = 1.10 * max(r_est, max_distance_to_center)
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
| **Frontflächen-Bias** | Nur vorderer Bogen sichtbar, Schwerpunkt sitzt vorn | 2r/π-Korrektion entlang Sichtstrahl |
| **FOV-Randclipping** | Seitlicher Zylinderumriss wird abgeschnitten | Radius-Clipping: `radius = max(radius, prev_r)` |
| **Verdeckungs-Kollaps** | Okklusion → dünne Maske → r_est→0 | Schrumpf-Limit: `radius >= 0.99 * prev_r` |
| **Wiederaufnahme-Sprung** | Track re-emerges an anderem Ort → v_raw explodiert | `v_raw = clip(v_raw, ±3 m/s)` |
| **Sensor-Rauschen** | Einzelne Pixel verschieben Centroid | EMA mit α=0.3 dämpft |
| **Leere Maske** | Vollständige Verdeckung | `stale=True`, letzten Wert halten |

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