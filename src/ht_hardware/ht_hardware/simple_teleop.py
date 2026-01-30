import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# 설정: 이동 속도
SPEED = 0.6
TURN_SPEED = 0.8

msg = """
---------------------------
   Simple Key Teleop
---------------------------
    w
 a  s  d

w: Forward
s: Backward
a: Left Turn
d: Right Turn
Space: Stop
CTRL-C to quit
---------------------------
"""

class SimpleTeleop(Node):
    def __init__(self):
        super().__init__('simple_teleop')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

    def send_vel(self, linear, angular):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.publisher_.publish(twist)

def get_key():
    # 터미널 세팅을 변경하여 키 입력을 즉시 받음
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    rclpy.init()
    node = SimpleTeleop()
    print(msg)

    try:
        while True:
            key = get_key()
            
            # 키보드 입력 매핑
            if key == 'w':
                node.send_vel(SPEED, 0.0)
            elif key == 's':
                node.send_vel(-SPEED, 0.0)
            elif key == 'a':
                node.send_vel(0.0, TURN_SPEED)  # 좌회전
            elif key == 'd':
                node.send_vel(0.0, -TURN_SPEED) # 우회전
            elif key == ' ': # 스페이스바 정지
                node.send_vel(0.0, 0.0)
            elif key == '\x03': # Ctrl+C
                break
            else:
                node.send_vel(0.0, 0.0) # 다른 키 누르면 정지

    except KeyboardInterrupt:
        pass
    finally:
        node.send_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()