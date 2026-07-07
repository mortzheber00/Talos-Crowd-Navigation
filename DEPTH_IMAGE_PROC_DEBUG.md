# Debug-Anleitung: depth_image_proc funktioniert nicht

## Schnelle Überprüfung

Starte in **drei separaten Terminals**:

### Terminal 1: Roboter
```bash
ros2 run bullet_sims walking
# oder: ros2 launch bullet_sims t23.launch.py
```

### Terminal 2: depth_image_proc
```bash
ros2 launch bullet_sims depth_image_proc.launch.py
```

### Terminal 3: Debug
```bash
# Überprüfe, welche Topics publiziert werden
ros2 topic list | grep -iE 'depth|camera'

# Überprüfe die Frequenz der Nachrichten
ros2 topic hz /depth_image_rect
ros2 topic hz /depth_camera_info
ros2 topic hz /depth/points  # Sollte Pointcloud-Frequenz zeigen

# Inspiziere den Node
ros2 node info /point_cloud_xyz
```

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
