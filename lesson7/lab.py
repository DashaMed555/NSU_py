import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def catch_ghosts(directory):
    search = load_search_image(directory)
    candy_ghost, pumpkin_ghost, scary_ghost = load_ghost_images(directory)

    search_key_points, search_descriptors = detect_and_compute(search)
    candy_key_points, candy_descriptors = detect_and_compute(candy_ghost)
    pumpkin_key_points, pumpkin_descriptors = detect_and_compute(pumpkin_ghost)
    scary_key_points, scary_descriptors = detect_and_compute(scary_ghost)

    framing(search, search_key_points, search_descriptors,
            candy_ghost, candy_key_points, candy_descriptors,
            pumpkin_ghost, pumpkin_key_points, pumpkin_descriptors,
            scary_ghost, scary_key_points, scary_descriptors)

    cv2.imshow("Caught Ghosts", search)
    cv2.waitKey(0)


def load_search_image(directory):
    search_image = cv2.imread(f"{directory}/lab7.png")
    return search_image


def load_ghost_images(directory):
    candy_ghost_image = cv2.imread(f"{directory}/candy_ghost.png")
    pumpkin_ghost_image = cv2.imread(f"{directory}/pampkin_ghost.png")
    scary_ghost_image = cv2.imread(f"{directory}/scary_ghost.png")
    return candy_ghost_image, pumpkin_ghost_image, scary_ghost_image


def detect_and_compute(image):
    sift = cv2.SIFT_create()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(image_gray, None)
    return key_points, descriptors


def match_key_points(ghosts_descriptors, search_descriptors):
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ghosts_descriptors, search_descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


def find_homography(image1, key_points1, image2, key_points2, matches):
    min_match_count = 20
    if len(matches) > min_match_count:
        source_points = np.float32([key_points1[m.queryIdx].pt for m in matches])
        destination_points = np.float32([key_points2[m.trainIdx].pt for m in matches])
        m, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = image1.shape[:2]
        points = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        destination = cv2.perspectiveTransform(points, m)
        image2 = cv2.polylines(image2, [np.int32(destination)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        image3 = cv2.drawMatches(image1, key_points1, image2, key_points2, matches, None, **draw_params)
        cv2.imshow("GhostsParty", image3)
        cv2.waitKey(0)
    else:
        print(f"Not enough matches are found - {len(matches)}/{min_match_count}")

# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
#                        matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def framing(search, search_key_points, search_descriptors,
            candy_ghost, candy_key_points, candy_descriptors,
            pumpkin_ghost, pumpkin_key_points, pumpkin_descriptors,
            scary_ghost, scary_key_points, scary_descriptors):
    key_points = np.array([search_key_points[i].pt for i in range(len(search_key_points))])
    bandwidth = estimate_bandwidth(key_points, quantile=0.1, n_samples=500)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    mean_shift.fit(key_points)

    labels = mean_shift.labels_
    n_clusters = len(np.unique(labels))
    cluster_key_points = []

    for i in range(n_clusters):
        d, = np.where(labels == i)
        cluster_key_points.append([search_key_points[x] for x in d])

    for i in range(n_clusters):
        search_key_points = cluster_key_points[i]
        d, = np.where(labels == i)
        search_descriptors_ = search_descriptors[d]

        candy_matches = match_key_points(candy_descriptors, search_descriptors_)
        pumpkin_matches = match_key_points(pumpkin_descriptors, search_descriptors_)
        scary_matches = match_key_points(scary_descriptors, search_descriptors_)

        find_homography(candy_ghost, candy_key_points, search, search_key_points, candy_matches)
        find_homography(pumpkin_ghost, pumpkin_key_points, search, search_key_points, pumpkin_matches)
        find_homography(scary_ghost, scary_key_points, search, search_key_points, scary_matches)

    cv2.destroyAllWindows()


catch_ghosts("хэллоуинский переполох")
