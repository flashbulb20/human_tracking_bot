# 🐍 Python 가상환경(venv) 설정 가이드 (YOLO / ROS2 필수)

본 프로젝트는 YOLO (ultralytics) 및 딥러닝 라이브러리를 사용하므로,
라즈베리파이 시스템 Python이 아닌 전용 가상환경(venv) 에서 실행해야 합니다.

ROS2 노드는 기본적으로 시스템 Python을 사용하기 때문에,
설정하지 않으면 아래와 같은 오류가 발생합니다.
```bash
ModuleNotFoundError: No module named 'ultralytics'
```
이를 해결하기 위해 아래 절차를 반드시 수행하세요.

## 1️⃣ venv 생성
```bash
cd ~/human_tracking_ws
python3 -m venv venv
```

## 2️⃣ venv 활성화
```bash
source ~/human_tracking_ws/venv/bin/activate
```
프롬프트 앞에 (venv) 가 보이면 정상입니다.

## 3️⃣ 필수 패키지 설치
```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```
설치 확인:
```bash
python -c "from ultralytics import YOLO; print('OK')"
```

## 4️⃣ ROS2 노드가 venv Python을 사용하도록 수정
ROS2 빌드 후 생성되는 실행 파일은 기본적으로 시스템 Python을 사용합니다:
```bash
#!/usr/bin/python3
```
이를 venv Python으로 강제 변경해야 합니다.

### 수동 적용
```bash
sed -i '1c #!/home/raspi/human_tracking_ws/venv/bin/python' \
install/ht_vision/lib/ht_vision/yolo_detector

sed -i '1c #!/home/raspi/human_tracking_ws/venv/bin/python' \
install/ht_vision/lib/ht_vision/yolo_detector_pose
```

## 5️⃣ 🔄 자동 적용 (권장)
ROS2를 다시 빌드하면 shebang이 다시 /usr/bin/python3 로 돌아가기 때문에
매번 수정하는 번거로움을 방지하기 위해 venv 활성화 시 자동 적용하도록 설정합니다.

### 설정 방법
```bash
nano ~/human_tracking_ws/venv/bin/activate
```
맨 아래에 다음 줄 추가:
```bash
# ROS2 노드가 venv Python을 사용하도록 강제
sed -i '1c #!/home/raspi/human_tracking_ws/venv/bin/python' \
~/human_tracking_ws/install/ht_vision/lib/ht_vision/yolo_detector_seg 2>/dev/null

sed -i '1c #!/home/raspi/human_tracking_ws/venv/bin/python' \
~/human_tracking_ws/install/ht_vision/lib/ht_vision/yolo_detector_pose 2>/dev/null
```
이제 venv를 활성화할 때마다 자동으로 수정됩니다.

## 6️⃣ 실행 순서 (중요)
항상 아래 순서를 지켜야 합니다.
```bash
cd ~/human_tracking_ws
source venv/bin/activate
source install/setup.bash
ros2 run ht_vision yolo_detector_pose
```

### ✅ 정상 동작 기준
다음 명령이 성공하면 환경 설정이 완료된 것입니다.
```bash
python -c "import ultralytics; print('YOLO OK')"
ros2 run ht_vision yolo_detector_pose
```