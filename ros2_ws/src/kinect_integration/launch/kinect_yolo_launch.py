import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '-c', 'source /home/pi/venv_freenect/bin/activate && ros2 run kinect_integration kinect_yolo_node'],
            output='screen'
        ),
    ])
