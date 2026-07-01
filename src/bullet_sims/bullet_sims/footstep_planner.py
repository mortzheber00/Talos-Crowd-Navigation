import numpy as np
import pinocchio as pin
from enum import Enum
import time
import pybullet as pb
import pybullet_data
import bullet_sims.talos_conf as conf



class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        
    def poseInWorld(self):
        return self.pose
        
    def plot(self, simulation):
        # Draw footprint polygon in the world using the provided simulation API
        # Expecting `footprint` as 3 x n numpy array in the foot frame and
        # `self.pose` as a pin.SE3 object.
        try:
            verts = self.footprint
            # transform vertices into world frame
            world_pts = []
            for i in range(verts.shape[1]):
                v = verts[:, i]
                p = self.pose.translation + self.pose.rotation.dot(v)
                world_pts.append(p.tolist())

            # choose color by side
            color = [0, 1, 0] if self.side == Side.LEFT else [1, 0, 0]

            # draw edges
            for i in range(len(world_pts)):
                a = world_pts[i]
                b = world_pts[(i + 1) % len(world_pts)]
                try:
                    simulation.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=2.0)
                except Exception:
                    # fallback for alternative API names
                    try:
                        simulation.addDebugLine(a, b, color)
                    except Exception:
                        pass

            # display side label at centroid
            centroid = np.mean(np.array(world_pts), axis=0).tolist()
            label = 'L' if self.side == Side.LEFT else 'R'
            try:
                simulation.addUserDebugText(label, centroid, textColorRGB=color, textSize=1.2)
            except Exception:
                try:
                    simulation.addDebugText(label, centroid, color)
                except Exception:
                    pass

            # mark target position with small cross (no dependency on sphere API)
            cross_size = 0.02
            cx, cy, cz = centroid
            cross_pts = [
                ([cx - cross_size, cy, cz], [cx + cross_size, cy, cz]),
                ([cx, cy - cross_size, cz], [cx, cy + cross_size, cz]),
                ([cx, cy, cz - cross_size], [cx, cy, cz + cross_size])
            ]
            for a, b in cross_pts:
                try:
                    simulation.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=2.0)
                except Exception:
                    try:
                        simulation.addDebugLine(a, b, color)
                    except Exception:
                        pass
        except Exception:
            # If anything fails, don't crash the planner visualization
            return None

        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        self.initial_support = None
        self.initial_swing = None
        
    def planLine(self, T_support_w, T_swing_w, support_side, swing_side, no_steps):
        """Plan a straight-line sequence. 
        Returns ONLY the actual future steps. The initial stance is stored internally
        for visualization purposes.
        """
        dx = self.conf.step_size_x
        dy = 2 * self.conf.step_size_y

        lfxp = abs(self.conf.lfxp)
        lfxn = -abs(self.conf.lfxn)
        lfyp = abs(self.conf.lfyp)
        lfyn = -abs(self.conf.lfyn)

        footprint = np.array([
            [lfxp,  lfxp,  lfxn,  lfxn], 
            [lfyp,  lfyn,  lfyn,  lfyp], 
            [0.0,   0.0,   0.0,   0.0]   
        ])
        
        # === INITIALLAYOUT FÜR DIE VISUALISIERUNG SPEICHERN ===
        # Diese landen nicht in der Rückgabeliste!
        self.initial_support = FootStep(pin.SE3(T_support_w), footprint, support_side)
        self.initial_swing = FootStep(pin.SE3(T_swing_w), footprint, swing_side)

        steps = []
        
        # === ERSTER ECHTER SCHRITT (Index 0) ===
        # Basiert kinematisch auf dem Fuß, der jetzt gerade in der Luft ist (T_swing_w).
        current_side = swing_side
        translation_update = np.array([dx, 0.0, 0.0])
        
        new_translation = T_swing_w.translation + T_swing_w.rotation.dot(translation_update)
        new_rotation = T_swing_w.rotation.copy()
        
        step_0 = FootStep(pin.SE3(new_rotation, new_translation), footprint, current_side)
        steps.append(step_0)

        # === ALLE WEITEREN SCHRITTE (Ab Index 1 bis no_steps - 1) ===
        for i in range(1, no_steps):
            # Seite wechseln basierend auf dem direkt vorherigen Schritt
            current_side = other_foot_id(steps[-1].side)
            
            if i == no_steps - 1:
                # Letzter Schritt: Schließt nur parallel auf (kein dx nach vorne)
                translation_update = np.array([0.0, dy if current_side == Side.LEFT else -dy, 0.0])
            else:
                # Normaler Schritt nach vorne (dx) und Spurwechsel (dy)
                translation_update = np.array([dx, dy if current_side == Side.LEFT else -dy, 0.0])
            
            parent_pose = steps[-1].pose
            new_translation = parent_pose.translation + parent_pose.rotation.dot(translation_update)
            new_rotation = parent_pose.rotation.copy()
            
            step = FootStep(pin.SE3(new_rotation, new_translation), footprint, current_side)
            steps.append(step)
                                
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        """Zeichnet die Dummy-Startpositionen und danach den echten Plan."""
        # Standkonfiguration zeichnen (falls vorhanden)
        if hasattr(self, 'initial_support'):
            self.initial_support.plot(simulation)
        if hasattr(self, 'initial_swing'):
            self.initial_swing.plot(simulation)
            
        # Die echten geplanten Schritte zeichnen
        for step in self.steps:
            step.plot(simulation)

            
    #>>>>TODO: Generate a plan and plot it in pybullet.
    #>>>>TODO: Check that the plan looks as expected

# Eine kleine Mock-Konfigurationsklasse zum Testen
class DummyConfig:
    step_size_x = 0.2
    step_size_y = 0.1  # Abstand zur Mitte (Soll-Spurbreite ist 2 * step_size_y)
    lfxp = 0.08   # Fußlänge vorne
    lfxn = -0.04  # Fußlänge hinten
    lfyp = 0.04   # Fußbreite links
    lfyn = -0.04  # Fußbreite rechts
    # foot_scaling = 1.
    # lfxp = foot_scaling*0.12                # foot length in positive x direction
    # lfxn = foot_scaling*0.08                # foot length in negative x direction
    # lfyp = foot_scaling*0.065               # foot length in positive y direction
    # lfyn = foot_scaling*0.065               # foot length in negative y direction

def main():
    # PyBullet GUI starten
    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    
    # Eine Ebene laden, damit wir eine visuelle Referenz haben
    planeId = pb.loadURDF("plane.urdf")

    # 1. Konfiguration initialisieren
    config = conf
    planner = FootStepPlanner(config)

    # 2. Startpose definieren (Hier leicht gedreht um 45 Grad um die Z-Achse, zum Testen der Rotation!)
    # Falls du flach starten willst: R = np.eye(3)
    R_rotated = pin.utils.rotate('z', np.deg2rad(45.0)) 
    T_start = pin.SE3(R_rotated, np.array([0.0, 0.0, 0.005])) # Minimal über dem Boden

    # 3. Plan erstellen (8 Schritte, startend mit dem LINKEN Fuß)
    planner.planLine(T_start, support_side=Side.LEFT, no_steps=8)

    # 4. In PyBullet plotten
    # Wir übergeben das `pb`-Modul selbst, da es die `addUserDebugLine`-Methoden besitzt
    planner.plot(pb)

    print("Schrittplan erfolgreich gezeichnet! Schließe das GUI-Fenster zum Beenden.")
    
    # Simulationsschleife am Leben erhalten
    try:
        while pb.isConnected():
            pb.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pb.disconnect()

if __name__=='__main__':
    main()
