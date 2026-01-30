import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gpiozero import Motor

class MotorDriver(Node):
    def __init__(self):
        super().__init__('motor_driver')
        
        # ---------------------------------------------------------
        # 1. 핀 번호 설정 (사용자 제공 BCM 번호)
        # ---------------------------------------------------------
        # Front Left (FL)
        self.fl_in1 = 5
        self.fl_in2 = 6
        self.fl_en  = 12
        
        # Front Right (FR)
        self.fr_in1 = 13
        self.fr_in2 = 19
        self.fr_en  = 18

        # Rear Left (RL)
        self.rl_in1 = 20
        self.rl_in2 = 21
        self.rl_en  = 16

        # Rear Right (RR)
        self.rr_in1 = 23
        self.rr_in2 = 24
        self.rr_en  = 25
        # ---------------------------------------------------------

        self.get_logger().info('Initializing 4WD Motors...')

        # 2. GPIOZero 모터 객체 생성 (PWM 모드 활성화)
        try:
            # 왼쪽 바퀴들
            self.motor_fl = Motor(forward=self.fl_in1, backward=self.fl_in2, enable=self.fl_en, pwm=True)
            self.motor_rl = Motor(forward=self.rl_in1, backward=self.rl_in2, enable=self.rl_en, pwm=True)
            
            # 오른쪽 바퀴들
            self.motor_fr = Motor(forward=self.fr_in1, backward=self.fr_in2, enable=self.fr_en, pwm=True)
            self.motor_rr = Motor(forward=self.rr_in1, backward=self.rr_in2, enable=self.rr_en, pwm=True)
            
            self.get_logger().info('All 4 motors initialized successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize motors: {e}')

        # 3. Subscriber: cmd_vel 토픽 구독
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10)
        
        # 4. 로봇 파라미터 (튜닝 필요)
        # 바퀴 간격이 넓을수록 회전 시 모터 속도 차이가 커집니다.
        self.wheel_separation = 0.15 # 예: 15cm
        self.max_speed_scale = 1.0   # 모터 최대 속도 제한 (0.0 ~ 1.0)

    def listener_callback(self, msg):
        """Twist 메시지를 받아 모터 PWM으로 변환"""
        linear_x = msg.linear.x   # 전진/후진 속도
        angular_z = msg.angular.z # 회전 속도

        # 차동 구동 계산 (Differential Drive Kinematics)
        left_speed = linear_x - (angular_z * self.wheel_separation / 2.0)
        right_speed = linear_x + (angular_z * self.wheel_separation / 2.0)

        # 4륜 구동이므로 앞뒤 바퀴에 같은 속도 명령 전달
        self.set_side_speed('left', left_speed)
        self.set_side_speed('right', right_speed)

    def set_side_speed(self, side, speed):
        """왼쪽 또는 오른쪽 면의 모터 속도 설정"""
        # 속도를 -1.0 ~ 1.0 사이로 제한 (Clamp)
        speed = max(min(speed, 1.0), -1.0)
        
        # 전체 속도 스케일링 (너무 빠르면 줄이기 위함)
        speed = speed * self.max_speed_scale

        if side == 'left':
            self.control_motor(self.motor_fl, speed)
            self.control_motor(self.motor_rl, speed)
        elif side == 'right':
            self.control_motor(self.motor_fr, speed)
            self.control_motor(self.motor_rr, speed)

    def control_motor(self, motor, speed):
        """개별 모터 구동 함수"""
        if speed > 0:
            motor.forward(speed)
        elif speed < 0:
            motor.backward(-speed) # backward는 양수 값을 받음
        else:
            motor.stop()

def main(args=None):
    rclpy.init(args=args)
    motor_driver = MotorDriver()
    
    try:
        rclpy.spin(motor_driver)
    except KeyboardInterrupt:
        pass
    finally:
        # 안전하게 모터 정지 및 리소스 해제
        motor_driver.motor_fl.close()
        motor_driver.motor_rl.close()
        motor_driver.motor_fr.close()
        motor_driver.motor_rr.close()
        motor_driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()