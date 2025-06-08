import cv2 as cv
import mediapipe as mp
import numpy as np

def get_R(from_vec, to_vec):
    axis = np.cross(from_vec, to_vec)
    axis = axis / np.linalg.norm(axis)
    cos = np.dot(from_vec, to_vec)
    theta = np.arccos(cos)

    rvec = axis * theta
    R, _ = cv.Rodrigues(rvec)

    return R

def putText(img, text, value, is_left, index):
    cv.putText(img, text + f"{value}", (10, 30 * index) if is_left else (1820, 30 * index), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 8)
    cv.putText(img, text + f"{value}", (10, 30 * index) if is_left else (1820, 30 * index), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

cap = cv.VideoCapture(0)

prev_tick = cv.getTickCount()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

is_pinch = False
pinch_history =np.zeros(3, dtype=bool)
is_drawing = False

segments_3d =[]
segments_2d = []
trash_3d = []
trash_2d = []

start_3d = None
end_3d = None
start_2d = None
end_2d = None

c_x, c_y = 960, 540
f_x = f_y = 1400

K = np.array([
    [f_x, 0.0, c_x],
    [0.0, f_y, c_y],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeff = np.array([
    0.00329361,
    0.24557882,
   -0.00259378,
    0.00467413,
   -0.50948872
], dtype=np.float32)

t_low = 0.15
t_high = 0.2

print1 = None
print2 = None

color = (0, 0, 255)

while True:
    valid, img = cap.read()
    if not valid:
        break

    img = cv.flip(img, 1)
    h, w = img.shape[:2]
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    rvec = None
    tvec = None

    key = cv.waitKey(1) & 0xFF

    point_color1 = (840, 1000)
    point_color2 = (960, 1000)
    point_color3 = (1080, 1000)
    radius_color1 = 20
    radius_color2 = 20
    radius_color3 = 20
    is_color1 = False
    is_color2 = False
    is_color3 = False

    # is_pinch = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        pts_hand = np.array([(hand.landmark[i].x * w, hand.landmark[i].y * h, hand.landmark[i].z * w) for i in range(21)])
        point_target_2d = (pts_hand[4] + pts_hand[8]) / 2.0
        point_target_2d = tuple(map(int, point_target_2d[:2]))

        d_palm = np.linalg.norm(pts_hand[0] - pts_hand[5])
        pts_norm_hand = (pts_hand - pts_hand[0]) / d_palm

        v1 = pts_norm_hand[9]
        v2 = pts_norm_hand[13]

        v_norm = np.cross(v1, v2)
        v_norm = v_norm / np.linalg.norm(v_norm)

        v_z = np.array([0.0, 0.0, 1.0])

        R = get_R(v_norm, v_z)

        pts_rotated = np.dot(pts_norm_hand, R.T)

        v_m = (pts_rotated[9] + pts_rotated[13]) / 2.0
        v_y = np.array([0.0, -1.0, 0.0])

        R = get_R(v_m, v_y)
        
        pts_object = np.dot(pts_rotated, R.T)

        d_norm_pinch = np.linalg.norm(pts_norm_hand[4][:2] - pts_norm_hand[8][:2])

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

        point_target_3d = ((pts_norm_hand[4] + pts_norm_hand[8]) / 2.0).reshape(3, 1) + tvec
        point_target_3d = np.round(point_target_3d, 1)

        if d_norm_pinch < t_low:
            is_pinch = True

        elif d_norm_pinch > t_high:
            is_pinch = False
        
        pinch_history[:-1] = pinch_history[1:]
        pinch_history[-1] = is_pinch

        is_pinch = np.any(pinch_history)

        x, y =point_target_2d
        if point_color1[0] - radius_color1 <= x <= point_color1[0] + radius_color1 and point_color1[1] - radius_color1 <= y <= point_color1[1] + radius_color1:
            point_color1 = (840, 990)
            radius_color1 = 30
            is_color1 = True
        elif point_color2[0] - radius_color2 <= x <= point_color2[0] + radius_color2 and point_color2[1] - radius_color2 <= y <= point_color2[1] + radius_color2:
            point_color2 = (960, 990)
            radius_color2 = 30
            is_color2 = True
        elif point_color3[0] - radius_color3 <= x <= point_color3[0] + radius_color3 and point_color3[1] - radius_color3 <= y <= point_color3[1] + radius_color3:
            point_color3 = (1080, 990)
            radius_color3 = 30
            is_color3 = True

        if is_pinch and not is_drawing:
            if is_color1:
                color = (0, 0, 255)
            elif is_color2:
                color = (0, 255, 0)
            elif is_color3:
                color = (255, 0, 0)
            else:
                start_3d = point_target_3d
                start_2d = point_target_2d
                is_drawing = True
        elif not is_pinch and is_drawing:
            end_3d = point_target_3d
            end_2d = point_target_2d
            segments_3d.append((start_3d, end_3d, color))
            segments_2d.append((start_2d, end_2d, color))
            is_drawing = False

        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
        print1 = is_pinch

    for sp, ep, c in segments_2d:
        cv.line(img, sp, ep, c, 2)
    if is_pinch and not is_color1 and not is_color2 and not is_color3:
        cv.line(img, start_2d, point_target_2d, color, 2)

    cv.circle(img, point_color1, radius_color1, (20, 20, 180), -1)
    cv.circle(img, point_color2, radius_color2, (20, 180, 20), -1)
    cv.circle(img, point_color3, radius_color3, (180, 20, 20), -1)

    putText(img, "Pinch: ", print1, True, 1)

    curr_tick = cv.getTickCount()
    fps = cv.getTickFrequency() / (curr_tick - prev_tick)
    prev_tick = curr_tick
    putText(img, "", round(fps, 2), False, 1)

    cv.imshow("test", img)

    if key is 26 and segments_3d and segments_2d:
        trash_3d.append(segments_3d.pop())
        trash_2d.append(segments_2d.pop())
    
    if key is 25 and trash_3d and trash_2d:
        segments_3d.append(trash_3d.pop())
        segments_2d.append(trash_2d.pop())

    if key is 27:
        break

cap.release()
cv.destroyAllWindows()

import matplotlib.pyplot as plt

def render_segments(segments):
    bgr_to_name = {
        (255, 0, 0): 'blue',
        (0, 255, 0): 'green',
        (0, 0, 255): 'red'
    }
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    for start, end, c in segments:
        xs, ys, zs = [start[0], end[0]], [start[1], end[1]], [start[2], end[2]]
        ax.plot(xs, ys, zs, color=bgr_to_name.get(c))
        ax.scatter(xs, ys, zs, s=20, color=bgr_to_name.get(c))
    ax.view_init(elev=-90, azim=-90)
    ax.set_axis_off()
    plt.show()

if segments_3d:
    render_segments(segments_3d)