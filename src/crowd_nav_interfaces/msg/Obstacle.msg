# One tracked 2D dynamic obstacle (disk) in the world ground plane, SI units.
# Produced either by the onboard depth-camera tracker (Talos.update_obstacles)
# or from simulation ground truth (PedestrianSimulator.states).

string name        # stable track key (e.g. MuJoCo geom name); may be empty
int32 id           # numeric id (geom id for perceived, pedestrian id for truth)
float64 x          # world position x [m]
float64 y          # world position y [m]
float64 radius     # disk radius [m]
float64 vx         # world velocity x [m/s]
float64 vy         # world velocity y [m/s]
bool stale         # perceived only: track was occluded/out of view on the
                   # last update and still carries its previous estimate
