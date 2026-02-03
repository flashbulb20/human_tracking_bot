import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time

class TrackingController(Node):
    def __init__(self):
        super().__init__('tracking_controller')
        
        # [핵심 변경 1] Queue Size를 1로 줄여서 딜레이 최소화 (가장 최신 데이터만 처리)
        self.target_sub = self.create_subscription(
            Point, 
            '/target_point', 
            self.control_callback, 
            1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- [튜닝 파라미터] ---
        self.target_area = 0.5
        
        # P게인 (반응성): 오차에 비례해서 속도를 냄
        self.linear_k = 2.5
        self.angular_k = 1.0
        
        # [핵심 변경 2] D게인 (브레이크): 오차가 줄어드는 속도를 감지해 미리 감속
        # 이 값이 크면 회전이 묵직해지고(진동 감소), 작으면 가벼워짐(진동 발생)
        self.angular_d = 0.1 

        self.min_speed = 0.2
        self.max_linear_speed = 1.0
        self.max_angular_speed = 1.0
        
        # PD 제어를 위한 이전 오차 저장 변수
        self.last_error_x = 0.0
        self.last_time = time.time()
        
        self.last_msg_time = time.time()
        self.create_timer(0.1, self.safety_stop_check)
        self.get_logger().info("Tracking Controller (PD Control + Low Latency) Started!")

    def control_callback(self, msg):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0: return # 0으로 나누기 방지

        self.last_msg_time = current_time
        twist = Twist()

        if msg.z <= 0.0:
            self.stop_robot()
            return

        # --- [PD 회전 제어] ---
        # 1. 오차 계산 (목표 - 현재)
        error_x = 0.5 - msg.x
        
        # 2. 미분(D) 계산 (오차의 변화량 / 시간)
        # 오차가 빠르게 줄어들면 derivative는 음수(-)가 되어 출력을 깎아먹음 (브레이크 효과)
        derivative = (error_x - self.last_error_x) / dt
        
        # 3. PD 출력 계산
        angular_output = (error_x * self.angular_k) + (derivative * self.angular_d)
        
        twist.angular.z = angular_output
        
        # 다음 계산을 위해 현재 상태 저장
        self.last_error_x = error_x
        self.last_time = current_time

        # --- [거리 제어 (기존 유지)] ---
        current_area = msg.y
        error_dist = self.target_area - current_area
        
        if current_area < 0.005:
            error_dist = 0.0
        
        raw_linear = error_dist * self.linear_k

        if raw_linear > 0.01:
            twist.linear.x = max(raw_linear, self.min_speed)
        elif raw_linear < -0.01:
            twist.linear.x = min(raw_linear, -self.min_speed)
        else:
            twist.linear.x = 0.0
            
        # 속도 제한
        twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)

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