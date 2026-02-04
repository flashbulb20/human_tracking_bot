# 🤖 Human Tracking Robot (HT-Bot) with ROS 2 & YOLOv8
Raspberry Pi 5와 ROS 2 Jazzy를 기반으로 한 지능형 사람 추종 로봇입니다. 단순한 Bounding Box 추적을 넘어, **Pose Estimation(관절 추정)**과 **Kalman Filter(경로 예측)**를 결합하여 끊김 없는 주행 성능을 제공합니다.

## ✨ 주요 기능 (Key Features)
이 프로젝트는 일반적인 추적 로봇과 차별화된 고급 기능들을 포함하고 있습니다.

1. 🎯 스마트 타겟 선정 (Smart Locking)
제스처 인식: 로봇 정면에서 손을 들고 1.5초간 유지하면 타겟으로 자동 등록됩니다.

안정적 추적: 옷자락이 흔들려도 영향받지 않도록, 사람의 **몸통(Torso: 어깨와 골반의 중심)**을 추적 좌표로 사용합니다.

2. 🧠 강건한 추적 알고리즘 (Robust Tracking)
Kalman Filter: 타겟이 기둥 뒤로 사라지거나(Occlusion) 카메라 딜레이가 발생해도, **이동 경로를 예측(Prediction)**하여 부드럽게 주행합니다.

Re-Identification (Re-ID): 타겟을 놓쳤을 때 HSV 색상 히스토그램 매칭을 통해 다시 나타난 주인을 정확히 재식별합니다.

수색 모드 (Search Mode): 타겟이 완전히 사라지면 제자리에서 회전하며 주변을 탐색합니다.

3. ⚙️ 적응형 제어 (Adaptive Control)
Adaptive Gain PID: 타겟과의 거리에 따라 회전 민감도(Angular Gain)를 실시간으로 조절하여, 근거리에서의 진동(Oscillation)을 잡고 원거리 반응성을 높였습니다.

Zero-Latency: 이미지 처리 큐(Queue)를 최적화하여 영상 처리 지연으로 인한 오버슈팅(Overshoot)을 방지했습니다.

## 🛠️ 하드웨어 구성 (Hardware)
| Component      | Description                                   |
|----------------|-----------------------------------------------|
| SBC            | Raspberry Pi 5 (Ubuntu 24.04 / ROS 2 Jazzy)   |
| Camera         | USB Webcam (TS-B7WQ30 Webcam)                 |
| Motor Driver   | L298N (x2)                                    |
| Motors         | DC Gear Motors (x4)                           |
| Power          | USB Power Bank (generic)                      |
| Motor Power    | 9V Battery (prototype / testing only)         |

## 📦 설치 방법 (Installation)
1. 필수 라이브러리 설치
ROS 2 Jazzy 환경이 설치되어 있다고 가정합니다.

```bash
# 시스템 패키지 설치
sudo apt install python3-pip v4l-utils ros-jazzy-v4l2-camera -y

# Python 의존성 설치 (가상환경 권장)
pip install ultralytics opencv-python numpy
```

2. 워크스페이스 설정 및 빌드
```bash
mkdir -p ~/human_tracking_ws/src
cd ~/human_tracking_ws/src
git clone https://github.com/flashbulb20/human_tracking_bot.git
cd ..

# 의존성 해결 및 빌드
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 🚀 실행 방법 (Usage)
모든 노드(카메라, 제어기, YOLO)를 한 번에 실행합니다.
```bash
source install/setup.bash
ros2 launch ht_bringup robot_bringup.launch.py
```

작동 시나리오
대기 모드: 로봇이 켜지면 제자리에서 대기합니다.

타겟 락온: 카메라 정면(1~2m 거리)에 서서 손을 머리 위로 듭니다.

확인: 화면에 노란색 게이지바가 차오르며, 완료 시 **빨간색 박스(TARGET LOCKED)**로 변합니다. (rqt로 /yolo_result 확인)

주행: 이제 손을 내려도 로봇이 당신을 따라다닙니다.

숨바꼭질: 로봇 시야에서 사라지면 로봇이 잠시 예측 주행 후, 제자리에서 회전하며 당신을 찾습니다.

## 📂 패키지 구조 (Package Structure)
ht_vision: 시각 인지 패키지

yolo_detector_pose.py: Pose Estimation 기반 추적 및 Re-ID 로직 (메인)

yolo_detector_seg.py: Segmentation 기반 추적 (옵션)

ht_control: 로봇 제어 패키지

tracking_controller.py: 적응형 PID 제어기 및 모터 명령 생성

ht_hardware: 하드웨어 인터페이스

motor_driver.py: GPIO 모터 제어

ht_bringup: 런처 패키지

robot_bringup.launch.py: 전체 시스템 실행 스크립트

## 🔧 튜닝 가이드 (Configuration)
환경에 맞춰 src/ht_control/ht_control/tracking_controller.py의 파라미터를 수정하세요.

```bash
# 목표 거리 설정 (값이 클수록 가까이 붙음)
self.target_area = 0.5 

# 속도 반응성 조절
self.linear_k = 2.5   # 전진 속도 게인
self.angular_k = 2.0  # 회전 속도 게인 (Adaptive 적용됨)
```

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.