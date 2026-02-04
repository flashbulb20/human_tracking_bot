import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time

class TrackingController(Node):
    def __init__(self):
        super().__init__('tracking_controller')
        
        # 딜레이 최소화를 위해 큐 사이즈 1 유지
        self.target_sub = self.create_subscription(
            Point, 
            '/target_point', 
            self.control_callback, 
            1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- [가변 게인 튜닝] ---
        # 1. 목표 거리
        self.target_area = 0.5 
        
        # 2. 속도 게인
        self.linear_k = 2.5
        
        self.angular_k = 2.0 
        self.angular_d = 0.1  # D게인 유지

        # 3. 속도 제한
        self.min_speed = 0.2
        self.max_linear_speed = 1.0
        self.max_angular_speed = 1.0
        # -----------------------

        # PD 제어 변수
        self.last_error_x = 0.0
        self.last_time = time.time()
        
        self.last_msg_time = time.time()
        self.create_timer(0.1, self.safety_stop_check)
        self.get_logger().info("Tracking Controller (Adaptive Gain Mode) Started!")

    def control_callback(self, msg):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0: return

        self.last_msg_time = current_time
        twist = Twist()

        if msg.z == 2.0:
            twist.linear.x = 0.0
            twist.angular.z = 0.4 # 천천히 회전
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Searching Mode: Rotating...")
            return

        if msg.z <= 0.0:
            self.stop_robot()
            return

        # --- [1. 가변 회전 게인 계산] ---
        # 거리가 가까울수록(면적이 클수록) 회전 민감도를 낮춤
        current_area = msg.y
        adaptive_factor = max(0.2, 1.0 - current_area) # 최소 0.2배는 보장
        
        # 실제 적용될 회전 게인
        real_angular_k = self.angular_k * adaptive_factor

        # --- [2. PD 회전 제어] ---
        error_x = 0.5 - msg.x
        derivative = (error_x - self.last_error_x) / dt
        
        # 적응형 게인 적용
        angular_output = (error_x * real_angular_k) + (derivative * self.angular_d)
        twist.angular.z = angular_output
        
        self.last_error_x = error_x
        self.last_time = current_time

        # --- [3. 거리 제어] ---
        error_dist = self.target_area - current_area
        
        # 노이즈 무시
        if current_area < 0.005:
            error_dist = 0.0
        
        raw_linear = error_dist * self.linear_k

        # 최소 속도 및 제한
        if raw_linear > 0.01:
            twist.linear.x = max(raw_linear, self.min_speed)
        elif raw_linear < -0.01:
            twist.linear.x = min(raw_linear, -self.min_speed)
        else:
            twist.linear.x = 0.0
            
        twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)

        self.cmd_vel_pub.publish(twist)
        
        # 디버깅: 현재 적용된 회전 게인이 얼마인지 확인해보세요!
        # self.get_logger().info(f"Area: {current_area:.2f} | Adaptive K: {real_angular_k:.2f}")

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