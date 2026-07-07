#!/usr/bin/env python3
"""
Script to inspect which topics the point_cloud_xyz_node expects.
Run this while the node is running in another terminal.
"""
import subprocess
import time
import sys

def get_node_subscriptions(node_name):
    """Get subscriptions from a running node"""
    try:
        result = subprocess.run(
            ['ros2', 'node', 'info', node_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

def list_all_topics():
    """List all published topics"""
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        topics = result.stdout.strip().split('\n')
        depth_topics = [t for t in topics if 'depth' in t.lower() or 'camera' in t.lower()]
        print("=== Topics with 'depth' or 'camera' ===")
        for topic in depth_topics:
            print(f"  {topic}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    print("Waiting 2 seconds for nodes to start...")
    time.sleep(2)

    print("\n=== Checking point_cloud_xyz node ===")
    get_node_subscriptions('/point_cloud_xyz')

    print("\n=== All relevant topics ===")
    list_all_topics()
