import cv2


def main():
    cap, time, mode = initialization()
    frame_prev = get_frame(cap)
    frame_prev_p = prepare_frame(frame_prev)
    while True:
        time += 1
        frame_curr = get_frame(cap)
        frame_curr_p = prepare_frame(frame_curr)
        mode, time = timer(frame_curr, mode, time)
        motion_detection = get_md_frame(frame_curr_p, frame_prev_p, mode)
        cv2.imshow("Camera", frame_curr)
        cv2.imshow("Motion detection", motion_detection)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        frame_prev_p = frame_curr_p
    cap.release()


def change_mode(mode):
    if mode == 'Red light':
        mode = 'Green light'
    else:
        mode = 'Red light'
    return mode


def initialization():
    cap = cv2.VideoCapture(0)
    time = 0
    mode = 'Red light'
    return cap, time, mode


def get_frame(cap):
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (960, 540))
            break
    return frame


def prepare_frame(frame):
    frame_smooth = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_smooth_gray = cv2.cvtColor(frame_smooth, cv2.COLOR_BGR2GRAY)
    return frame_smooth_gray


def get_md_frame(frame_curr, frame_prev, mode):
    motion_detection = cv2.merge([frame_curr, frame_curr, frame_curr])
    h, w, _ = motion_detection.shape
    cv2.rectangle(motion_detection, (0, 0), (w, h), (0, 0, 255), -1)
    if mode == 'Red light':
        diff = cv2.absdiff(frame_curr, frame_prev)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        motion_detection = cv2.drawContours(motion_detection, contours, -1, (0, 255, 0), 2)
    return motion_detection


def timer(frame_curr, mode, time):
    cv2.putText(frame_curr, mode, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if time == 100:
        mode = change_mode(mode)
        time = 0
    return mode, time


main()
