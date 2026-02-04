import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import time

# --- [1] 칼만 필터---
class SimpleKalmanFilter:
    def __init__(self):
        # 상태 변수 4개 (x, y, vx, vy), 측정 변수 2개 (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        # 공분산 행렬 초기화 (튜닝 영역)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03 # 값이 클수록 예측을 불신
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 # 값이 클수록 측정을 불신 (YOLO 흔들림 보정)
        
        self.is_initialized = False

    def update(self, x, y):
        """YOLO 측정값으로 보정"""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.is_initialized:
            # 초기값 설정
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.is_initialized = True
        
        # 1. Predict (다음 상태 예측)
        self.kf.predict()
        # 2. Correct (측정값 반영)
        return self.kf.correct(measurement)

    def predict(self):
        """측정값이 없을 때 예측값만 반환"""
        if not self.is_initialized:
            return 0.5, 0.5 # 중앙값
        
        prediction = self.kf.predict()
        return prediction[0][0], prediction[1][0] # x, y 반환

# -----------------------------------------------------------

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        self.model = YOLO('yolov8n-pose.pt') 
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 1)
        self.image_pub = self.create_publisher(Image, '/yolo_result', 10)
        self.target_pub = self.create_publisher(Point, '/target_point', 1) # 큐 사이즈 1

        # 상태 변수
        self.target_id = None
        self.target_hist = None
        self.last_seen_time = 0
        
        # Locking 변수
        self.lock_candidate_id = None
        self.lock_start_time = 0.0
        self.LOCK_DURATION = 1.5 

        self.kf = SimpleKalmanFilter()
        self.last_known_pos = (0.5, 0.0) # 마지막 위치 저장 (x, area)
        self.search_mode = False         # 수색 모드 플래그

        self.get_logger().info("YOLO Tracker with Kalman Filter & Search Mode Started!")

    def calc_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        center_hsv = hsv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        if center_hsv.size == 0: center_hsv = hsv
        hist = cv2.calcHist([center_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None: return 0.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def is_hand_raised(self, keypoints):
        if keypoints.conf is None: return False
        kpts = keypoints.xy[0].cpu().numpy()
        confs = keypoints.conf[0].cpu().numpy()
        NOSE, L_WRIST, R_WRIST = 0, 9, 10
        if confs[NOSE] < 0.5: return False 
        ref_y = kpts[NOSE][1]
        return ((confs[L_WRIST] > 0.5 and kpts[L_WRIST][1] < ref_y) or 
                (confs[R_WRIST] > 0.5 and kpts[R_WRIST][1] < ref_y))

    def get_torso_center(self, keypoints, box):
        if keypoints.conf is None: return None
        kpts = keypoints.xy[0].cpu().numpy()
        confs = keypoints.conf[0].cpu().numpy()
        torso_indices = [5, 6, 11, 12]
        valid_points = [kpts[i] for i in torso_indices if confs[i] > 0.5]
        if len(valid_points) >= 2: return np.mean(valid_points, axis=0)[0]
        else: return (box[0] + box[2]) / 2

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return

        height, width, _ = frame.shape
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker="bytetrack.yaml")
        
        target_msg = Point()
        target_msg.z = -1.0 
        
        current_target_found = False
        potential_candidates = [] 
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
                
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0: continue
                roi_hist = self.calc_histogram(person_roi)

                color = (0, 255, 0)
                text = f"ID: {track_id}"
                thickness = 2
                
                box_cx = ((x1 + x2) / 2) / width

                # [로직 1] 타겟 선정
                if self.target_id is None:
                    is_candidate = False
                    if keypoints is not None and len(keypoints) > i:
                        if self.is_hand_raised(keypoints[i]):
                            if 0.35 < box_cx < 0.65:
                                is_candidate = True
                    
                    if is_candidate:
                        frame_candidate_found = True
                        if self.lock_candidate_id == track_id:
                            elapsed = time.time() - self.lock_start_time
                            remain = max(0.0, self.LOCK_DURATION - elapsed)
                            bar_width = int((x2 - x1) * (elapsed / self.LOCK_DURATION))
                            cv2.rectangle(frame, (x1, y1-25), (x1 + bar_width, y1-15), (0, 255, 255), -1) 
                            cv2.rectangle(frame, (x1, y1-25), (x2, y1-15), (255, 255, 255), 1)
                            
                            if elapsed >= self.LOCK_DURATION:
                                self.target_id = track_id
                                self.target_hist = roi_hist
                                self.lock_candidate_id = None
                                
                                # [추적 시작]
                                self.last_seen_time = time.time()
                                current_target_found = True
                                self.kf = SimpleKalmanFilter() # KF 초기화
                                self.search_mode = False
                                
                                color = (0, 0, 255)
                                text = f"TARGET {track_id}"
                                thickness = 4
                                self.get_logger().info(f"Target LOCKED! ID: {track_id}")
                        else:
                            self.lock_candidate_id = track_id
                            self.lock_start_time = time.time()

                # [로직 2] 타겟 추적
                elif self.target_id is not None:
                    if track_id == self.target_id:
                        current_target_found = True
                        self.last_seen_time = time.time()
                        self.search_mode = False # 찾았으니 수색 종료
                        
                        if self.target_hist is None: self.target_hist = roi_hist
                        else: self.target_hist = 0.9 * self.target_hist + 0.1 * roi_hist
                        
                        cx_pixel = self.get_torso_center(keypoints[i], box)
                        cx = cx_pixel / width 
                        
                        # --- [KF Update] 실측값으로 칼만 필터 보정 ---
                        self.kf.update(cx, 0) # x축만 예측
                        # ----------------------------------------

                        box_area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
                        self.last_known_pos = (cx, box_area_ratio) # 마지막 정보 저장

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

        # 타겟 후보 초기화
        if self.target_id is None and not frame_candidate_found:
            self.lock_candidate_id = None
            self.lock_start_time = 0.0

        # [로직 3] 타겟 소실 처리 (Kalman Filter Prediction & Search Mode)
        if self.target_id is not None and not current_target_found:
            
            # A. Re-ID 시도
            best_match_id = None
            max_sim = 0.0
            for pid, sim in potential_candidates:
                if sim > max_sim:
                    max_sim = sim
                    best_match_id = pid
            
            if max_sim > 0.75:
                self.get_logger().warn(f"Re-ID: Switched {self.target_id} -> {best_match_id}")
                self.target_id = best_match_id
                self.kf = SimpleKalmanFilter() # ID 바뀌면 KF 재설정
                self.last_seen_time = time.time()
            
            else:
                elapsed_lost = time.time() - self.last_seen_time
                
                # B. Coasting Phase (0 ~ 2초): 칼만 필터 예측값 사용
                if elapsed_lost < 2.0:
                    pred_x, _ = self.kf.predict()
                    
                    # 예측값을 유효 범위(0~1)로 클램핑
                    pred_x = max(0.0, min(pred_x, 1.0))
                    
                    target_msg.x = float(pred_x)
                    target_msg.y = float(self.last_known_pos[1]) # 거리(면적)는 마지막 값 유지
                    target_msg.z = 1.0 # "보고 있다"고 속임
                    
                    cv2.putText(frame, f"PREDICTING... ({elapsed_lost:.1f}s)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                # C. Search Phase (2 ~ 6초): 제자리 회전하며 찾기
                elif elapsed_lost < 6.0:
                    self.search_mode = True
                    target_msg.z = 2.0 # [약속] 2.0은 '수색 중' 신호
                    
                    cv2.putText(frame, "SEARCHING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                # D. Give Up (6초 초과): 타겟 해제
                else:
                    self.target_id = None
                    self.target_hist = None
                    self.search_mode = False
                    cv2.putText(frame, "TARGET LOST COMPLETELY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.target_pub.publish(target_msg)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

if __name__ == '__main__':
    main()