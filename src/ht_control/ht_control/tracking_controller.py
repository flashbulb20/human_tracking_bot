import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time

class TrackingController(Node):
    def __init__(self):
        super().__init__('tracking_controller')
        
        self.target_sub = self.create_subscription(
            Point, '/target_point', self.control_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- [튜닝 파라미터 수정] ---
        
        # 1. 목표 면적 (Segmentation은 박스보다 작으므로 값을 낮춤)
        self.target_area = 0.06 
        
        # 2. 속도 게인 (반응성)
        self.linear_k = 2.0    # 전진 반응을 더 민감하게
        self.angular_k = 2.5 
        
        # 3. 최소 기동 속도 (Deadzone 보정 - 매우 중요!)
        self.min_speed = 0.18
        
        # 최대 속도
        self.max_linear_speed = 0.6
        self.max_angular_speed = 1.5
        # --------------------------

        self.last_msg_time = time.time()
        self.create_timer(0.1, self.safety_stop_check)
        self.get_logger().info("Tracking Controller (Seg Tuned) Started!")

    def control_callback(self, msg):
        self.last_msg_time = time.time()
        twist = Twist()

        if msg.z <= 0.0:
            self.stop_robot()
            return

        # [1] 회전 제어
        error_x = 0.5 - msg.x
        if abs(error_x) < 0.05: error_x = 0.0
        twist.angular.z = error_x * self.angular_k

        # [2] 거리 제어 (Linear)
        current_area = msg.y
        error_dist = self.target_area - current_area
        
        # 너무 멀리 있는 노이즈(0.005 이하)는 무시
        if current_area < 0.005:
            error_dist = 0.0
        
        # P제어 계산
        raw_linear = error_dist * self.linear_k

        # --- [최소 속도 보정] ---
        if raw_linear > 0.01:  # 전진해야 할 때
            twist.linear.x = max(raw_linear, self.min_speed)
        elif raw_linear < -0.01: # 후진해야 할 때
            twist.linear.x = min(raw_linear, -self.min_speed)
        else:
            twist.linear.x = 0.0
            
        # 속도 제한 (Clamp)
        twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)

        # 디버깅: 터미널에서 현재 상태 확인 가능
        # self.get_logger().info(f"Area: {current_area:.4f} -> Cmd: {twist.linear.x:.2f}")

        self.cmd_vel_pub.publish(twist)

    def safety_stop_check(self):
        if time.time() - self.last_msg_time > 0.5:
            self.stop_robot()

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = TrackingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()