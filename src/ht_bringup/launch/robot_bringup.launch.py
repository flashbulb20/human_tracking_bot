import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # 1. 카메라 노드 (v4l2_camera 사용 시)
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera',
            parameters=[{
                'video_device': '/dev/video0',
                'image_size': [320, 240],
                'frame_rate': 10.0,
                'io_method': 'mmap',
                'use_v4l2_buffer': True
            }],
            output='screen'
        ),

        # 2. 모터 드라이버
        Node(
            package='ht_hardware',
            executable='motor_driver',
            name='motor_driver',
            output='screen'
        ),

        # 3. YOLO 감지기 (카메라 켜지고 3초 뒤 실행)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='ht_vision',
                    executable='yolo_detector_pose',
                    name='yolo_detector_pose',
                    output='screen'
                )
            ]
        ),

        # 4. 추적 컨트롤러
        Node(
            package='ht_control',
            executable='tracking_controller',
            name='tracking_controller',
            output='screen'
        )
    ])