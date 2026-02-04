import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gpiozero import Motor
import time

class MotorDriver(Node):
    def __init__(self):
        super().__init__('motor_driver')
        
        # --- 핀 번호 설정 (BCM) ---
        # Front Left
        self.fl_in1, self.fl_in2, self.fl_en = 5, 6, 12
        # Front Right
        self.fr_in1, self.fr_in2, self.fr_en = 13, 19, 18
        # Rear Left
        self.rl_in1, self.rl_in2, self.rl_en = 20, 21, 16
        # Rear Right
        self.rr_in1, self.rr_in2, self.rr_en = 23, 24, 25
        # -------------------------

        self.get_logger().info('Initializing Motors with Safety & Turn Boost...')

        try:
            # PWM 모드로 모터 설정
            self.motor_fl = Motor(forward=self.fl_in1, backward=self.fl_in2, enable=self.fl_en, pwm=True)
            self.motor_rl = Motor(forward=self.rl_in1, backward=self.rl_in2, enable=self.rl_en, pwm=True)
            self.motor_fr = Motor(forward=self.fr_in1, backward=self.fr_in2, enable=self.fr_en, pwm=True)
            self.motor_rr = Motor(forward=self.rr_in1, backward=self.rr_in2, enable=self.rr_en, pwm=True)
        except Exception as e:
            self.get_logger().error(f'Motor Init Failed: {e}')

        self.subscription = self.create_subscription(Twist, 'cmd_vel', self.listener_callback, 10)
        
        # [안전 장치] 마지막 명령 시간 기록
        self.last_cmd_time = time.time()
        self.create_timer(0.1, self.safety_stop_callback) # 0.1초마다 검사

        # [튜닝 파라미터]
        self.max_speed = 1.0       # 최대 속도 (0.0 ~ 1.0)
        self.turn_gain = 1.5       # 회전 민감도 (회전이 안되면 이 값을 2.0~3.0으로 높이세요!)

    def listener_callback(self, msg):
        self.last_cmd_time = time.time() # 명령 수신 시간 갱신
        
        linear = msg.linear.x
        angular = msg.angular.z

        # 회전 힘 보정 (제자리 회전 시 힘을 더 줌)
        angular_speed = angular * self.turn_gain

        # 왼쪽/오른쪽 바퀴 속도 계산
        left_speed = linear - angular_speed
        right_speed = linear + angular_speed

        # 디버깅용 로그 (터미널에서 값 확인 가능)
        # self.get_logger().info(f'L: {left_speed:.2f}, R: {right_speed:.2f}')

        self.set_side_speed('left', left_speed)
        self.set_side_speed('right', right_speed)

    def safety_stop_callback(self):
        # 0.5초 이상 명령이 없으면 강제 정지 (누르고 있을 때만 가게 하기 위함)
        if time.time() - self.last_cmd_time > 0.5:
            self.stop_all()

    def set_side_speed(self, side, speed):
        speed = max(min(speed, self.max_speed), -self.max_speed) # -1 ~ 1 제한
        
        if side == 'left':
            self.control_motor(self.motor_fl, speed)
            self.control_motor(self.motor_rl, speed)
        elif side == 'right':
            self.control_motor(self.motor_fr, speed)
            self.control_motor(self.motor_rr, speed)

    def control_motor(self, motor, speed):
        if speed > 0.05: # 데드존 (너무 작은 값은 무시)
            motor.forward(speed)
        elif speed < -0.05:
            motor.backward(-speed)
        else:
            motor.stop()

    def stop_all(self):
        self.motor_fl.stop()
        self.motor_rl.stop()
        self.motor_fr.stop()
        self.motor_rr.stop()

def main(args=None):
    rclpy.init(args=args)
    node = MotorDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()