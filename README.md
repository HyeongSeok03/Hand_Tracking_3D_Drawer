# Hand_Tracking_3D_Drawer

## 개요
이 프로그램은 **MediaPipe Hand Tracking**과 **OpenCV**를 활용해 실시간으로 손 동작을 인식하고, 3D 공간에 그림을 그릴 수 있는 인터랙티브 애플리케이션입니다. 사용자는 손가락 핀치 제스처를 통해 2D 화면에 그림을 그리며 동시에 3D 좌표계에 선분을 저장할 수 있습니다.

## 시연 영상
[유튜브](https://youtu.be/XFzMkgWucH4)

## 주요 기능
- **실시간 손 추적**: 21개의 핸드 랜드마크 감지
- **핀치 제스처 인식**: 엄지-검지 간 거리 측정을 통한 드로잉 트리거
- **다색상 드로잉 시스템**: 3가지 색상(빨강/초록/파랑) 선택 지원
- **3D 공간 시각화**: matplotlib을 이용한 3D 선분 렌더링
- **실행 취소/다시 실행**: Ctrl+Z/Ctrl+Y 단축키 지원

## 의존성
~~~
pip install opencv-python mediapipe numpy matplotlib
~~~

## 실행 방법
~~~
python main.py
~~~

## 사용 방법
1. 웹캠 앞에서 **핀치 동작** 시작 → 드로잉 시작
2. 핀치 동작 유지 → 선분 확장
3. 핀치 동작 해제 → 선분 완성
4. 화면 하단 **색상 원**에 핀치 접촉 시 색상 변경
5. **ESC** 키 입력으로 종료 후 3D 렌더링 확인

## 키보드 컨트롤
| 키        | 기능          |
|-----------|--------------|
| `Ctrl+Z`  | 실행 취소     |
| `Ctrl+Y`  | 다시 실행     |
| `ESC`     | 프로그램 종료 |

## 문제와 해결

### 핀치점의 3차원 공간 상 좌표를 어떻게 효과적으로 계산할 것인가.
1. 최초에는 손뼘의 길이를 재고 그 길이가 크냐 작냐에 따라 z축 값을 계산하였음. x, y는 화면 상에 보이는 좌표를 기반으로 계산하였음. 하지만 이것은 3차원 공간상에 대응되는 정확도가 너무 낮았음.
2. solvePnP를 이용하여 계산하는 것을 시도하였음. object_point는 최초에 고정되어 있어야 하므로 변형이 어려운 손뼘을 이루는 5개의 랜드마크를 스페이스바를 눌러 순간 저장하고 그것을 object_point로 이용하여 rvec과 tvec을 계산하였음. 하지만 스페이스바를 누르는 과정이 번거롭고 object_point를 저장하는 순간의 손의 방향에 따라 rvec의 기준값이 달라지는 문제 발생. (손을 이렇게 돌리고 스페이스바를 누르느냐, 저렇게 돌리고 스페이스바를 누르느냐에 따라 같은 방향의 손에 대한 rvec은 다른 값이 됨.)
3. 매 프레임 21개의 랜드마크를 저장하고, 손뼘을 기준으로 normalization을 수행하여 object_point로 사용하는 것을 시도하였음. 손바닥이 화면과 평행하게, 손바닥의 중앙을 가로지르는 선이 y축과 평행하게 랜마마크 전체를 회전하여 매 프레임 object_point에 사용되는 손이 일정한 방향을 하게 하였고, 전체 좌표는 손뼘의 길이로 나누어 길이도 화면에 보이는 손의 크기에 구애받지 않게 정규화를 진행하였음. 또한 좌표를 21개로 크게 늘릴 수 있어서 기존 solvePnP() 대신 solvePnPRansac()을 사용하여 안정성을 향상시켰음. 이것으로 계산량은 늘었지만 사용이 편리해지고 정확도가 크게 향상하였음.
~~~
# 두 벡터를 인수로 받아 한 벡터를 다른 벡터로 회전시키는 회전행렬을 반환하는 함수 구현
def get_R(from_vec, to_vec):
    axis = np.cross(from_vec, to_vec)
    axis = axis / np.linalg.norm(axis)
    cos = np.dot(from_vec, to_vec)
    theta = np.arccos(cos)

    rvec = axis * theta
    R, _ = cv.Rodrigues(rvec)

    return R

# 전체 랜드마크 좌표 저장
pts_hand = np.array([(hand.landmark[i].x * w, hand.landmark[i].y * h, hand.landmark[i].z * w) for i in range(21)])

# 손뼘의 길이 저장
d_palm = np.linalg.norm(pts_hand[0] - pts_hand[5])

# 1. 손목좌표를 뺌으로서 손목을 원점으로 맞추고 손뼘의 길이로 나누어 길이 정규화 진행
pts_norm_hand = (pts_hand - pts_hand[0]) / d_palm

# 2. 회전 정규화 과정
# 손바닥을 이루는 두 벡터 저장
v1 = pts_norm_hand[9]
v2 = pts_norm_hand[13]

# 위 두 벡터를 이용하여 손바닥에 수직인 벡터 계산
v_norm = np.cross(v1, v2)
v_norm = v_norm / np.linalg.norm(v_norm)

# z축 방향의 target vector 정의
v_z = np.array([0.0, 0.0, 1.0])

# 손바닥에 수직인 벡터를 z축으로 회전시키는 회전행렬 계산
R = get_R(v_norm, v_z)

# 길이 정규화가 진행된 손 좌표를 z축에 평행하게 회전
pts_rotated = np.dot(pts_norm_hand, R.T)

# 손바닥의 z축을 축으로한 회전방향도 마저 맞추기 위해 손바닥의 위쪽 방향의 벡터와 y축 방향의 target vector 정의
v_m = (pts_rotated[9] + pts_rotated[13]) / 2.0
v_y = np.array([0.0, -1.0, 0.0])

# 손바닥 위 방향 벡터를 y축으로 회전시키는 회전행렬 계산
R = get_R(v_m, v_y)

# 이전에 회전시킨 좌표들을 다시 y축으로 회전시켜 최종적인 pts_object 계산
pts_object = np.dot(pts_rotated, R.T)

# 이후 solvePnPRansac()으로 rvec과 tvec 계산
pts_hand_2d = pts_hand[:, :2].astype(np.float64)
ret, rvec, tvec, inliers = cv.solvePnPRansac(
    pts_object,
    pts_hand_2d,
    K,
    dist_coeff,
    flags=cv.SOLVEPNP_EPNP,
    reprojectionError=8.0,
    confidence=0.99,
    iterationsCount=100
)
~~~
-----------------------
### 핀치 상태를 어떻게 안정적으로 감지할 것인가
1. 최초에는 검지와 엄지 사이 거리를 재고 이것이 threshold보다 낮으면 핀치 상태로 판정하였음. 동일한 핀치 상황이라 하더라도 손이 가까우면 검지와 엄지 사이 거리는 멀어지기 때문에 이것을 보정하는 것이 필요했음. 핀치 상태일때 False로 바뀌는 경우가 잦은 문제도 존재하였음.
2. 위의 손 좌표 정규화를 이용하여 보정을 성공하였음. 하지만 여전히 핀치상태일 경우에 노이즈(True와 False 간 진동)가 심한 문제가 존재하였음.
3. Hysterisis Thresholding 기법을 이용하여 노이즈를 해결하였음. t_high와 t_low를 나누고 t_low보다 낮으면 pinch on, t_high보다 높으면 pinch off, 그 외에는 이전 상태를 유지하는 것으로 구현하였음.
4. 그럼에도 가끔 오인식으로 False가 뜨는데 이것에 의해 선분이 그려지다 끊기는 것은 치명적이었음. 그래서 이전 핀치 상태들을 기록하는 pinch_history 변수를 이용하여 3번 연속으로 핀치가 False가 될때만 False로 바뀌게 하였음.
~~~
# threshold를 high와 low로 나눔
t_low = 0.15
t_high = 0.2
is_pinch = False
pinch_history = np.zeros(3, dtype=bool)

while True:
  valid, img = cap.read()
  if not valid:
      break

  if results.multi_hand_landmarks:

    ...

    # Hysterisis를 이용한 pinch 탐지 구현. else인 경우 이전 값이 그대로 유지
    if d_norm_pinch < t_low:
        is_pinch = True
    elif d_norm_pinch > t_high:
        is_pinch = False

    # 일종의 큐연산 맨 앞은 나가고 뒤에 현재 핀치 상태 넣음
    pinch_history[:1] = pinch_history[:1]
    pinch_history[-1] = is_pinch

    # pinch_history 중 뭐 하나라도 True이면 True가 들어감 -> 모두 False이어야 False가 들어감
    is_pinch = np.any(pinch_history)
~~~

## 참고 사항
1. **카메라 파라미터**:
~~~
K = [1400, 1400, 1]  # 고정 캘리브레이션
~~~
2. **최적 화면 해상도**: 1920×1080 권장
3. **조명 조건**: 균일한 조명 환경에서 최적 성능

## 화면 구성
1. 실시간 FPS 표시 (우측 상단)
2. 3D 좌표값 디버깅 정보 (좌측 상단)
3. 색상 선택 팔레트 (화면 하단)
4. 실시간 드로잉 캔버스 (전체 화면)
