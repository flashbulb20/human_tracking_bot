import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading  # 별도 쓰레드 사용을 위해 추가
import sys

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # 1. 모델 로드
        self.model = YOLO('yolov8n.pt')
        
        # 2. ROS 설정
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/yolo_result', 10)
        self.target_pub = self.create_publisher(Point, '/target_point', 10)

        # 3. 추적 변수
        self.target_id = None
        
        # 4. 사용자 입력을 받는 별도 쓰레드 실행
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("YOLO Detector Started. Waiting for input in terminal...")

    def input_loop(self):
        """터미널에서 지속적으로 입력을 받는 함수"""
        print("\n" + "="*40)
        print("  [Target Selection Command]")
        print("  - Enter Number: Track that ID (e.g., 1)")
        print("  - Enter -1: Clear/Reset Target")
        print("  (Check ID numbers in RQT Image View)")
        print("="*40 + "\n")

        while rclpy.ok():
            try:
                # 사용자 입력 대기 (여기서 멈춰 있어도 영상처리는 계속됨)
                user_input = input("Enter Target ID >> ")
                
                val = int(user_input)
                if val == -1:
                    self.target_id = None
                    self.get_logger().info("Target Cleared.")
                else:
                    self.target_id = val
                    self.get_logger().info(f"Target Set to ID: {self.target_id}")
            except ValueError:
                print("Invalid input! Please enter an integer.")
            except EOFError:
                break

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        # YOLO 추적 실행
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker="bytetrack.yaml")
        
        target_found = False
        target_msg = Point()
        target_msg.z = 0.0  # 기본값: 없음

        for result in results:
            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = box
                
                # 시각화: 기본 박스 (초록색)
                color = (0, 255, 0)
                thickness = 2
                text = f"ID: {track_id}"

                # 사용자가 지정한 타겟인지 확인
                if self.target_id is not None and track_id == self.target_id:
                    target_found = True
                    color = (0, 0, 255)  # 타겟은 빨간색
                    thickness = 4
                    text = f"TARGET {track_id}"
                    
                    # 좌표 계산 및 발행 데이터 준비
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    target_msg.x = float(cx)
                    target_msg.y = float(cy)
                    target_msg.z = 1.0  # 발견됨

                # 그리기 (RQT 송출용)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 타겟 데이터 발행
        self.target_pub.publish(target_msg)
        
        # 결과 이미지 발행 (RQT용)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

        # cv2.imshow는 삭제됨

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()