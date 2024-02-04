import os
import cv2


def task1():
    file_list = os.listdir('nails_segmentation/images')
    pairs = []
    for file in file_list:
        image = cv2.imread(f'nails_segmentation/images/{file}')
        label = cv2.imread(f'nails_segmentation/labels/{file}')
        pairs.append((image, label))
    return pairs


def task2():
    pairs = task1()
    for (image, label) in pairs:
        cv2.imshow('Image', image)
        cv2.imshow('Label', label)
        key = cv2.waitKey(2500) & 0xff
        if key == 27:
            break
    cv2.destroyWindow('Image')
    cv2.destroyWindow('Label')


def task3():
    pairs = task1()
    for (image, label) in pairs:
        label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        _, bound = cv2.threshold(label_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Contours', image)
        key = cv2.waitKey(2500)
        if key == 27:
            break
    cv2.destroyWindow('Contours')


def task4():
    cap = cv2.VideoCapture("/Users/dasha_terminator/Documents/Frogger/win.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Frame", frame_gray)
        key = cv2.waitKey(20)
        if key == 27:
            break
    cv2.destroyWindow('Frame')
    cap.release()


task2()
# task3()
task4()
