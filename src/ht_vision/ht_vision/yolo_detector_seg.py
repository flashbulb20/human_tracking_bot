import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading
import numpy as np
import time

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Segmentation 모델 사용
        self.model = YOLO('yolo26n-seg.pt') 
        
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/image_raw/compressed', 
            self.image_callback, 
            10)
            
        self.image_pub = self.create_publisher(Image, '/yolo_result', 10)
        self.target_pub = self.create_publisher(Point, '/target_point', 10)

        # --- [Re-ID 관련 변수] ---
        self.target_id = None
        self.target_hist = None  # 타겟의 옷 색깔(히스토그램) 저장소
        self.last_seen_time = 0  # 타겟을 마지막으로 본 시간
        # -----------------------
        
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("YOLO Re-ID Tracker (Histogram) Started.")

    def input_loop(self):
        print("\n" + "="*40)
        print("  [Re-ID Tracker]")
        print("  Enter ID to track (e.g., 1)")
        print("  Enter -1 to clear")
        print("="*40 + "\n")
        while rclpy.ok():
            try:
                user_input = input("Enter Target ID >> ")
                val = int(user_input)
                if val == -1:
                    self.target_id = None
                    self.target_hist = None
                    self.get_logger().info("Target Cleared.")
                else:
                    self.target_id = val
                    self.target_hist = None # ID가 바뀌면 히스토그램 초기화
                    self.get_logger().info(f"Target Set to ID: {self.target_id}")
            except:
                pass

    def calc_histogram(self, image, mask=None):
        """이미지(또는 마스크 영역)의 HSV 색상 히스토그램을 계산"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Hue(색상)와 Saturation(채도)만 사용 (명도 V는 조명에 민감해서 제외)
        # H: 0~180, S: 0~256 -> 30x32 bins로 압축
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        
        # 정규화 (크기가 달라도 비교 가능하게 0~1로 만듦)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def compare_histograms(self, hist1, hist2):
        """두 히스토그램의 유사도 비교 (Correlation 기법)"""
        if hist1 is None or hist2 is None:
            return 0.0
        # 결과값: 1.0(완벽 일치) ~ 0.0(완전 다름)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return

        height, width, _ = frame.shape
        
        # 추적 실행
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker="bytetrack.yaml", retina_masks=False)
        
        target_msg = Point()
        target_msg.z = -1.0 
        
        current_target_found = False
        potential_candidates = [] # 타겟을 놓쳤을 때 후보군

        for result in results:
            if result.boxes.id is None or result.masks is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            masks = result.masks.xy 

            for box, track_id, mask_poly in zip(boxes, ids, masks):
                track_id = int(track_id)
                x1, y1, x2, y2 = box.astype(int)
                
                # 안전장치: 좌표가 이미지 범위를 벗어나지 않게
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # 현재 사람의 히스토그램 추출 (Re-ID용)
                # 1. 사람 영역만 자르기 (ROI)
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0: continue
                
                # 2. 마스크가 있으면 마스크 내부만, 없으면 박스 전체 사용
                roi_hist = self.calc_histogram(person_roi)

                color = (0, 255, 0)
                text = f"ID: {track_id}"
                similarity = 0.0

                # --- [타겟 추적 및 재식별 로직] ---
                
                # Case A: 현재 보고 있는 사람이 타겟임
                if self.target_id is not None and track_id == self.target_id:
                    current_target_found = True
                    self.last_seen_time = time.time()
                    
                    # [중요] 타겟의 현재 모습을 계속 업데이트 (조명 변화 대응)
                    # 급격한 변화를 막기 위해 기존 정보 90% + 새 정보 10% 반영
                    if self.target_hist is None:
                        self.target_hist = roi_hist
                    else:
                        self.target_hist = 0.9 * self.target_hist + 0.1 * roi_hist
                        
                    # 좌표 및 면적 계산 (이전과 동일)
                    if len(mask_poly) > 0:
                        cx = np.mean(mask_poly, axis=0)[0] / width
                    else:
                        cx = ((x1 + x2) / 2) / width
                    
                    mask_area_ratio = cv2.contourArea(mask_poly.astype(np.float32)) / (width * height)
                    
                    target_msg.x = float(cx)
                    target_msg.y = float(mask_area_ratio)
                    target_msg.z = 1.0
                    
                    color = (0, 0, 255)
                    text = f"TARGET {track_id}"
                    cv2.fillPoly(frame, [mask_poly.astype(np.int32)], (0, 0, 100))

                # Case B: 타겟이 아님 -> 후보군으로 등록
                elif self.target_id is not None and self.target_hist is not None:
                    # 저장된 타겟 모습과 얼마나 비슷한지 비교
                    similarity = self.compare_histograms(self.target_hist, roi_hist)
                    potential_candidates.append((track_id, similarity))
                    
                    # 디버깅용 유사도 표시 (0.8 이상이면 매우 비슷)
                    if similarity > 0.6:
                        text += f" ({similarity:.2f})"

                # 그리기
                cv2.polylines(frame, [mask_poly.astype(np.int32)], True, color, 2)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- [타겟 재식별 (Re-ID) 결정] ---
        # 타겟이 화면에 없고(Lost), 타겟을 놓친지 5초 이내라면 수색
        if not current_target_found and self.target_id is not None:
            if time.time() - self.last_seen_time < 5.0: 
                best_match_id = None
                max_sim = 0.0
                
                for pid, sim in potential_candidates:
                    if sim > max_sim:
                        max_sim = sim
                        best_match_id = pid
                
                # 유사도가 0.75 (75%) 이상이면 동일 인물로 간주하고 ID 변경!
                if max_sim > 0.75:
                    self.get_logger().warn(f"Re-ID Success! ID {self.target_id} -> {best_match_id} (Sim: {max_sim:.2f})")
                    self.target_id = best_match_id
                    # 갱신된 ID로 즉시 타겟 메시지 생성은 다음 프레임부터 됨

            elif time.time() - self.last_seen_time > 10.0:
                # 10초 이상 못 찾으면 포기
                cv2.putText(frame, "Target Lost Completely...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.target_pub.publish(target_msg)
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