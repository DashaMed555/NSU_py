import os
import cv2
import random


def gen1(count, height, width):
    file_names = os.listdir("nails_segmentation/images")
    while True:
        sample_ = file_names.copy()
        random.shuffle(sample_)
        sample_ = sample_[:count]
        dataset = []
        for file_name in sample_:
            image_ = cv2.imread(f"nails_segmentation/images/{file_name}")
            label_ = cv2.imread(f"nails_segmentation/labels/{file_name}")
            image_ = cv2.resize(image_, (height, width))
            label_ = cv2.resize(label_, (height, width))
            dataset.append((image_, label_))
        yield dataset


def gen2(count, height, width):
    init_dataset_gen = gen1(count, height, width)
    while True:
        init_sample = next(init_dataset_gen)
        dataset = []
        for (image, label) in init_sample:
            image_rotated, label_rotated = random_rotation(image, label)
            image_reflected, label_reflected = random_reflection(image_rotated, label_rotated)
            image_cut_out, label_cut_out = random_cutout(image_reflected, label_reflected)
            image_blurred = random_blur(image_cut_out)
            dataset.append((image_blurred, label_cut_out))
        yield dataset


def random_rotation(image, label):
    (h, w) = image.shape[:2]
    center = (int(w / 2), int(h / 2))
    angle = random.randint(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image_rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    label_rotated = cv2.warpAffine(label, rotation_matrix, (w, h))
    return image_rotated, label_rotated


def random_reflection(image, label):
    reflection_id = random.randint(1, 4)
    if reflection_id == 1:
        image_reflected = image[::-1].copy()
        label_reflected = label[::-1].copy()
    elif reflection_id == 2:
        image_reflected = image[::, ::-1].copy()
        label_reflected = label[::, ::-1].copy()
    elif reflection_id == 3:
        image_reflected = image[::-1, ::-1].copy()
        label_reflected = label[::-1, ::-1].copy()
    else:
        image_reflected = image
        label_reflected = label
    return image_reflected, label_reflected


def random_cutout(image, label):
    while True:
        (h, w) = image.shape[:2]
        x = random.randint(0, h - 2)
        y = random.randint(0, w - 2)
        xd = random.randint(1, h - x)
        yd = int(xd * h / w)
        image_cut_out = image[y:y+yd, x:x+xd].copy()
        label_cut_out = label[y:y+yd, x:x+xd].copy()
        image_cut_out = cv2.resize(image_cut_out, (h, w))
        label_cut_out = cv2.resize(label_cut_out, (h, w))
        if label_cut_out.max() == 255:
            break
    return image_cut_out, label_cut_out


def random_blur(image):
    max_kernel_size = min(min(image.shape[:2]), 15)
    random_kernel_size = random.randint(1, max_kernel_size)
    while random_kernel_size % 2 == 0:
        random_kernel_size = random.randint(3, max_kernel_size)
    image_blurred = cv2.medianBlur(image, random_kernel_size)
    return image_blurred


def show_nails(samples_num, nails_num, gen, height=640, width=640):
    samples_gen = gen(nails_num, height, width)
    for i in range(1, samples_num + 1):
        sample = next(samples_gen)
        for (image, label) in sample:
            cv2.imshow(f"Image{i}", image)
            cv2.imshow(f"Label{i}", label)
            key = cv2.waitKey(10000) & 0xff
            if key == 27:
                break
        cv2.destroyAllWindows()
        key = cv2.waitKey(1000) & 0xff
        if key == 27:
            cv2.destroyAllWindows()
            break


show_nails(3, 10, gen1)
show_nails(3, 10, gen2)
