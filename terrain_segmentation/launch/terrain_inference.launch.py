from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='terrain_segmentation',
            executable='segmentation_node',
            name='segmentation_node',
            parameters=[{
                'target_width': 848,
                'target_height': 840,
            }]
        )
    ])
