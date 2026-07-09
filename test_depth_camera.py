#!/usr/bin/env python3
"""
Test depth camera: spawn robot, attach depth camera to head, measure depth to a test box.
Run with: MUJOCO_GL=egl python3 test_depth_camera.py (GPU) or MUJOCO_GL=osmesa (CPU)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Force headless GL if running in container without DISPLAY
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, "src/simulator")
from simulator.mujoco_wrapper import MujocoWrapper


def test_depth_camera_simple():
    """Simplest test: minimal scene (no URDF), depth camera, test object."""
    print("=== Test 1: Minimal scene ===")

    sim = MujocoWrapper(render=True, sim_rate=100)  # with viewer

    # Add a simple camera holder at origin (head_link)
    head = sim._spec.worldbody.add_body(name="head_link", pos=[0, 0, 1.0])

    # Add a test box 1 meter in front of the camera (along +X)
    box_body = sim._spec.worldbody.add_body(name="test_box", pos=[1.0, 0, 1.0])
    import mujoco
    box_geom = box_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.2, 0.2],  # half-extents: 10cm deep, 40cm tall/wide
        rgba=[1, 0, 0, 0.8]  # red for visibility
    )

    # Recompile to include new bodies
    sim._dirty = True
    sim._model = None
    sim._ensure_compiled()

    # Attach depth camera to head_link
    print("Adding depth camera to 'head_link'...")
    sim.addCameraToLink("head_link", name="test_cam", fovy=58.0, width=640, height=480)

    # Run a few steps and read depth
    print("Stepping simulation...")
    for step in range(30):
        sim.step()

    print("Reading depth image...")
    depth = sim.get_depth_image()

    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth dtype: {depth.dtype}")

    # Find pixels that hit the box (non-zero, not background)
    valid = depth[depth > 0]
    if valid.size > 0:
        print(f"  Valid pixels: {valid.size} / {depth.size}")
        print(f"  Depth range: {valid.min():.3f} - {valid.max():.3f} m")
        print(f"  Expected: ~0.95 m (1.0 m box position - 0.05 m camera offset)")

        # Center region (where the box should be)
        h, w = depth.shape
        center_region = depth[h//3:2*h//3, w//3:2*w//3]
        center_valid = center_region[center_region > 0]
        if center_valid.size > 0:
            print(f"  Center region mean: {center_valid.mean():.3f} m (std {center_valid.std():.3f})")

    # Show the depth image
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(depth, cmap="turbo")
    plt.colorbar(label="Depth (m)")
    plt.title("Depth Image (raw)")

    plt.subplot(1, 3, 2)
    depth_clamped = np.clip(depth, 0, 5)  # clip to [0, 5m] for visibility
    plt.imshow(depth_clamped, cmap="turbo")
    plt.colorbar(label="Depth (m, clipped to 5m)")
    plt.title("Depth (clipped)")

    plt.subplot(1, 3, 3)
    plt.hist(valid, bins=50, edgecolor='k')
    plt.xlabel("Depth (m)")
    plt.ylabel("Pixel count")
    plt.title("Depth histogram")

    plt.tight_layout()
    plt.savefig("/tmp/depth_test_minimal.png", dpi=100)
    print(f"  Saved to /tmp/depth_test_minimal.png")
    plt.show()

    sim.disconnect()
    print("Test 1 complete.\n")


def test_depth_camera_with_urdf():
    """Test with a real robot URDF (if available) and measure floor distance."""
    # Check if talos URDF exists
    talos_urdf = "src/simulator/talos.urdf"
    if not os.path.exists(talos_urdf):
        print("=== Test 2: Skipped (URDF not found) ===\n")
        return

    print("=== Test 2: With Talos URDF ===")

    sim = MujocoWrapper(render=True, sim_rate=100)

    # Load robot
    print("Loading URDF...")
    body_id = sim.createBodyFromUrdf(
        talos_urdf,
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1],
        use_fixed_base=False
    )

    # Try to attach camera to head_link (or fallback to root)
    try:
        print("Attaching camera to 'head_link'...")
        sim.addCameraToLink("head_link", fovy=58.0, width=640, height=480)
    except ValueError as e:
        print(f"  {e}")
        print("  Attempting to find a suitable link...")
        # Fallback: list available links
        rec = sim._bodies[body_id]
        if rec.link_bodies:
            link_name = rec.link_bodies[0]
            print(f"  Using first joint link: '{link_name}'")
            sim.addCameraToLink(link_name, fovy=58.0, width=640, height=480)
        else:
            print("  No joint links found, skipping URDF test.")
            sim.disconnect()
            return

    # Step and measure
    print("Stepping simulation...")
    for step in range(100):
        sim.step()

    print("Reading depth...")
    depth = sim.get_depth_image()

    valid = depth[depth > 0]
    if valid.size > 0:
        print(f"  Valid pixels: {valid.size} / {depth.size}")
        print(f"  Depth range: {valid.min():.3f} - {valid.max():.3f} m")
        print(f"  Mean depth: {valid.mean():.3f} m (std {valid.std():.3f})")

    # Show histogram
    plt.figure(figsize=(8, 4))
    plt.hist(valid, bins=50, edgecolor='k')
    plt.xlabel("Depth (m)")
    plt.ylabel("Pixel count")
    plt.title("Depth histogram (URDF test)")
    plt.savefig("/tmp/depth_test_urdf.png", dpi=100)
    print(f"  Saved to /tmp/depth_test_urdf.png")
    plt.show()

    sim.disconnect()
    print("Test 2 complete.\n")


def interactive_test():
    """Interactive test: walk through depth camera with arrow keys / wasd."""
    print("=== Interactive Test ===")
    print("Controls: WASD = move box, R = reset, Q = quit (in MuJoCo viewer)")

    sim = MujocoWrapper(render=True, sim_rate=100)

    head = sim._spec.worldbody.add_body(name="head_link", pos=[0, 0, 1.0])
    box_body = sim._spec.worldbody.add_body(name="test_box", pos=[0.8, 0, 1.0])
    import mujoco
    box_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.2, 0.2], rgba=[0, 1, 0, 0.8])

    sim._dirty = True
    sim._model = None
    sim._ensure_compiled()

    sim.addCameraToLink("head_link", width=640, height=480)

    print("Running. Press Ctrl+C to exit.")
    try:
        step_count = 0
        while sim.isRunning():
            sim.step()
            step_count += 1

            # Read depth every 30 steps
            if step_count % 30 == 0:
                depth = sim.get_depth_image()
                valid = depth[depth > 0]
                if valid.size > 0:
                    print(f"  Step {step_count}: depth mean={valid.mean():.3f}m, std={valid.std():.3f}m")
    except KeyboardInterrupt:
        print("\nStopped.")

    sim.disconnect()
    print("Interactive test complete.\n")


if __name__ == "__main__":
    print("Testing depth camera integration with MuJoCo\n")
    print("=" * 60)

    test_depth_camera_simple()
    test_depth_camera_with_urdf()
    # Uncomment to run interactive test (requires X11/display):
    # interactive_test()

    print("=" * 60)
    print("All tests complete. Check /tmp/depth_test_*.png for results.")
