#!/bin/bash

# Script to debug depth_image_proc connectivity
# Run in a terminal after starting the robot and depth_image_proc

echo "=== ROS Topic Check ==="
echo ""
echo "1. All available topics:"
ros2 topic list

echo ""
echo "2. Topics containing 'depth' or 'camera':"
ros2 topic list | grep -iE 'depth|camera'

echo ""
echo "3. Check point_cloud_xyz node info:"
ros2 node list | grep -i point_cloud
if ros2 node list | grep -q point_cloud_xyz; then
    echo "  Found point_cloud_xyz, checking subscriptions..."
    ros2 node info /point_cloud_xyz 2>/dev/null | grep -A 20 "Subscriptions"
fi

echo ""
echo "4. Check depth image frequency:"
ros2 topic hz /depth_image_rect 2>/dev/null || echo "  Topic /depth_image_rect not found"

echo ""
echo "5. Check camera info frequency:"
ros2 topic hz /depth_camera_info 2>/dev/null || echo "  Topic /depth_camera_info not found"

echo ""
echo "6. Check if points topic exists:"
if ros2 topic list | grep -q "points"; then
    echo "  ✓ Found points topic"
    ros2 topic hz /depth/points 2>/dev/null || echo "    (but no data flowing)"
else
    echo "  ✗ No points topic found - depth_image_proc may not be working"
fi
