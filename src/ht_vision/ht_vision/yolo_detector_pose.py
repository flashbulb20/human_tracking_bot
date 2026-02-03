import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import time

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Pose 모델 (관절 추정)
        self.model = YOLO('yolov8n-pose.pt') 
        
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/image_raw/compressed', 
            self.image_callback, 
            10)
            
        self.image_pub = self.create_publisher(Image, '/yolo_result', 10)
        self.target_pub = self.create_publisher(Point, '/target_point', 10)

        # --- [추적 상태 변수] ---
        self.target_id = None
        self.target_hist = None
        self.last_seen_time = 0
        
        # --- [타겟 선정(Locking) 관련 변수] ---
        self.lock_candidate_id = None # 현재 조건을 만족 중인 후보 ID
        self.lock_start_time = 0.0    # 조건을 만족하기 시작한 시간
        self.LOCK_DURATION = 1.5      # 유지해야 하는 시간 (초)
        # -----------------------------------

        self.get_logger().info("Robust Tracker Started. (Center + Hand Raise + 1.5s Hold)")

    def calc_histogram(self, image):
        """Re-ID용 히스토그램 (중앙부 50%)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        center_hsv = hsv[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        if center_hsv.size == 0: center_hsv = hsv
        
        hist = cv2.calcHist([center_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None: return 0.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def is_hand_raised(self, keypoints):
        """손 들기 감지 (Trigger)"""
        if keypoints.conf is None: return False
        kpts = keypoints.xy[0].cpu().numpy()
        confs = keypoints.conf[0].cpu().numpy()

        NOSE = 0
        L_WRIST, R_WRIST = 9, 10

        if confs[NOSE] < 0.5: return False 
        
        ref_y = kpts[NOSE][1]
        left_raised = (confs[L_WRIST] > 0.5) and (kpts[L_WRIST][1] < ref_y)
        right_raised = (confs[R_WRIST] > 0.5) and (kpts[R_WRIST][1] < ref_y)
        
        return left_raised or right_raised

    def get_torso_center(self, keypoints, box):
        """몸통 중심 반환 (안정적 추적용)"""
        if keypoints.conf is None: return None
        kpts = keypoints.xy[0].cpu().numpy()
        confs = keypoints.conf[0].cpu().numpy()
        
        # 어깨(5,6), 골반(11,12)
        torso_indices = [5, 6, 11, 12]
        valid_points = []
        for i in torso_indices:
            if confs[i] > 0.5: valid_points.append(kpts[i])
        
        if len(valid_points) >= 2:
            return np.mean(valid_points, axis=0)[0]
        else:
            x1, _, x2, _ = box
            return (x1 + x2) / 2

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return

        height, width, _ = frame.shape
        
        # 추적 실행
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker="bytetrack.yaml")
        
        target_msg = Point()
        target_msg.z = -1.0 
        
        current_target_found = False
        potential_candidates = [] 
        
        # 이번 프레임에서 조건을 만족하는 후보가 있는지 확인용 플래그
        frame_candidate_found = False 

        for result in results:
            if result.boxes.id is None: continue
            
            keypoints = result.keypoints
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for i, (box, track_id) in enumerate(zip(boxes, ids)):
                track_id = int(track_id)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Re-ID용 히스토그램 추출
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0: continue
                roi_hist = self.calc_histogram(person_roi)

                color = (0, 255, 0)
                text = f"ID: {track_id}"
                thickness = 2
                
                # 박스 중심 계산 (조건 확인용)
                box_cx = ((x1 + x2) / 2) / width

                # =========================================================
                # [로직 1] 타겟 선정 (Locking Process)
                # =========================================================
                if self.target_id is None:
                    is_candidate = False
                    
                    # 1. 손을 들었는가?
                    if keypoints is not None and len(keypoints) > i:
                        if self.is_hand_raised(keypoints[i]):
                            # 2. 화면 중앙에 있는가? (0.35 ~ 0.65)
                            if 0.35 < box_cx < 0.65:
                                is_candidate = True
                    
                    if is_candidate:
                        frame_candidate_found = True
                        
                        if self.lock_candidate_id == track_id:
                            elapsed = time.time() - self.lock_start_time
                            remain = max(0.0, self.LOCK_DURATION - elapsed)
                            
                            # 시각화
                            bar_width = int((x2 - x1) * (elapsed / self.LOCK_DURATION))
                            cv2.rectangle(frame, (x1, y1-25), (x1 + bar_width, y1-15), (0, 255, 255), -1) 
                            cv2.rectangle(frame, (x1, y1-25), (x2, y1-15), (255, 255, 255), 1)
                            
                            text = f"Hold.. {remain:.1f}s"
                            color = (0, 255, 255)

                            # 3. 시간 충족 -> 타겟 확정!
                            if elapsed >= self.LOCK_DURATION:
                                self.target_id = track_id
                                self.target_hist = roi_hist
                                self.lock_candidate_id = None
                                
                                # ▼▼▼ [핵심 수정: 즉시 추적 상태로 전환] ▼▼▼
                                self.last_seen_time = time.time()  # 시간 갱신 (중요!)
                                current_target_found = True        # 찾았다고 표시 (중요!)
                                
                                # 시각적 피드백도 바로 빨간색으로 변경
                                color = (0, 0, 255)
                                text = f"TARGET {track_id}"
                                thickness = 4
                                
                                self.get_logger().info(f"Target LOCKED! ID: {track_id}")
                                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                                
                        else:
                            self.lock_candidate_id = track_id
                            self.lock_start_time = time.time()
                    
                    # 후보였는데 조건을 깼거나, 다른 사람인 경우 -> 타이머 초기화는 루프 밖에서 처리

                # =========================================================
                # [로직 2] 타겟 추적 (Tracking)
                # =========================================================
                elif self.target_id is not None:
                    if track_id == self.target_id:
                        current_target_found = True
                        self.last_seen_time = time.time()
                        
                        if self.target_hist is None: self.target_hist = roi_hist
                        else: self.target_hist = 0.9 * self.target_hist + 0.1 * roi_hist
                        
                        # 몸통 중심 추적
                        cx_pixel = self.get_torso_center(keypoints[i], box)
                        cx = cx_pixel / width 
                        
                        box_area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
                        
                        target_msg.x = float(cx)
                        target_msg.y = float(box_area_ratio)
                        target_msg.z = 1.0
                        
                        color = (0, 0, 255)
                        text = f"TARGET {track_id}"
                        thickness = 4
                        
                    elif self.target_hist is not None:
                        sim = self.compare_histograms(self.target_hist, roi_hist)
                        potential_candidates.append((track_id, sim))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # [타겟 선정 초기화 로직]
        # 이번 프레임에서 아무도 조건을 만족하지 못했다면 후보 취소
        if self.target_id is None and not frame_candidate_found:
            self.lock_candidate_id = None
            self.lock_start_time = 0.0

        # [로직 3] 타겟 놓침 -> Re-ID
        if self.target_id is not None and not current_target_found:
            best_match_id = None
            max_sim = 0.0
            for pid, sim in potential_candidates:
                if sim > max_sim:
                    max_sim = sim
                    best_match_id = pid
            
            if max_sim > 0.75:
                self.get_logger().warn(f"Re-ID: Switched {self.target_id} -> {best_match_id}")
                self.target_id = best_match_id
            elif time.time() - self.last_seen_time > 5.0:
                self.target_id = None
                self.target_hist = None
                cv2.putText(frame, "TARGET LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

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