import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage # CompressedImage 추가
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading
import numpy as np

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # 1. 모델 로드 (Segmentation)
        self.model = YOLO('yolo26n-seg.pt') 
        
        # 2. ROS 설정
        self.bridge = CvBridge()
        
        # [변경점 1] CompressedImage 구독
        # qos_profile은 기본 10이나, best_effort를 쓰면 더 빠를 수 있음 (여기선 기본 유지)
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/image_raw/compressed', # v4l2_camera가 발행하는 압축 토픽
            self.image_callback, 
            10)
            
        self.image_pub = self.create_publisher(Image, '/yolo_result', 10)
        self.target_pub = self.create_publisher(Point, '/target_point', 10)

        self.target_id = None
        
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("YOLO Compressed Segmentation Tracker Started.")

    def input_loop(self):
        print("\n" + "="*40)
        print("  [Segmentation Tracker (Compressed)]")
        print("  Enter ID to track (e.g., 1)")
        print("  Enter -1 to clear")
        print("="*40 + "\n")
        while rclpy.ok():
            try:
                user_input = input("Enter Target ID >> ")
                val = int(user_input)
                if val == -1:
                    self.target_id = None
                    self.get_logger().info("Target Cleared.")
                else:
                    self.target_id = val
                    self.get_logger().info(f"Target Set to ID: {self.target_id}")
            except:
                pass

    def image_callback(self, msg):
        try:
            # [변경점 2] Compressed Image 디코딩 (cv_bridge 없이 직접 수행 - 더 빠름)
            # 1. byte array를 numpy array로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            # 2. 이미지로 디코딩 (IMREAD_COLOR)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Decoding failed: {e}")
            return

        height, width, _ = frame.shape
        
        # 추적 실행
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker="bytetrack.yaml", retina_masks=False)
        
        target_msg = Point()
        target_msg.z = -1.0 

        for result in results:
            if result.boxes.id is None or result.masks is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            masks = result.masks.xy 

            for box, track_id, mask_poly in zip(boxes, ids, masks):
                track_id = int(track_id)
                x1, y1, x2, y2 = box
                color = (0, 255, 0)
                text = f"ID: {track_id}"

                if self.target_id is not None and track_id == self.target_id:
                    color = (0, 0, 255)
                    
                    if len(mask_poly) > 0:
                        centroid = np.mean(mask_poly, axis=0)
                        cx = centroid[0] / width
                    else:
                        cx = ((x1 + x2) / 2) / width

                    mask_area_pixel = cv2.contourArea(mask_poly.astype(np.float32))
                    mask_area_ratio = mask_area_pixel / (width * height)

                    target_msg.x = float(cx)
                    target_msg.y = float(mask_area_ratio)
                    target_msg.z = 1.0
                    
                    text = f"TARGET {track_id} | Area: {mask_area_ratio:.3f}"
                    
                    cv2.fillPoly(frame, [mask_poly.astype(np.int32)], (0, 0, 100))

                cv2.polylines(frame, [mask_poly.astype(np.int32)], True, color, 2)
                cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.target_pub.publish(target_msg)
        # 결과 이미지는 RQT 확인용이므로 일반 Image 메시지로 발행 (cv_bridge 필요)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

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