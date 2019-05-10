import logging
import argparse
import os
import cv2
import numpy as np
import math
import definitions
import random


def save_stiched_image(output_image_folder, stiched_image, i):
    cv2.imwrite(os.path.join(output_image_folder, "output_mosaic.jpg".format(i)), stiched_image)


def get_matching_pair_coordinates(kp_query, kp_train, match):
    kp_train_index = match.trainIdx
    kp_query_index = match.queryIdx
    train_point = kp_train[kp_train_index]
    query_point = kp_query[kp_query_index]
    return query_point.pt[0], train_point.pt[0], query_point.pt[1], train_point.pt[1]


def assemble_A_matrices(x, x_p, y, y_p):
    A_1 = np.zeros((2, 9))
    A_1[0, 0] = -1 * x
    A_1[1, 3] = -1 * x
    A_1[0, 1] = -1 * y
    A_1[1, 4] = -1 * y
    A_1[0, 2] = -1
    A_1[1, 5] = -1
    A_1[0, 6] = x * x_p
    A_1[0, 7] = x_p * y
    A_1[0, 8] = x_p
    A_1[1, 8] = y_p
    A_1[1, 7] = y_p * y
    A_1[1, 6] = y_p * x
    return A_1


def assemble_H_matrix(h):
    H = np.ones((3, 3))
    H[0, 0] = h[0]
    H[0, 1] = h[1]
    H[0, 2] = h[2]
    H[1, 0] = h[3]
    H[1, 1] = h[4]
    H[1, 2] = h[5]
    H[2, 0] = h[6]
    H[2, 1] = h[7]
    H[2, 2] = h[8]
    return H


def count_number_of_inlers(kp_query, kp_train, matches, H):
    count = 0
    matched_points = []
    for i in range(len(matches)):
        x, x_p, y, y_p = get_matching_pair_coordinates(kp_query, kp_train, matches[i])
        query_points = np.array([[x], [y], [1]])
        train_points = np.array([[x_p], [y_p], [1]])
        transform_points = np.matmul(H, query_points)
        transform_points = transform_points / transform_points[2]
        # Calculate tranformation error
        transform_points_error = transform_points - train_points
        abs_transform_points_error = np.abs(transform_points_error)
        # flag = abs_transform_points_error.all() < 0.001
        flag = ((abs_transform_points_error <= 0.5).sum() == abs_transform_points_error.size).astype(np.int)
        if flag:
            matched_points.append([x, x_p, y, y_p])
            count += 1
    return count, matched_points


def compute_homography_matrix(kp_query, kp_train, matches, i):
    # Get matching pair coordinates
    x1, x1_p, y1, y1_p = get_matching_pair_coordinates(kp_query, kp_train, matches[i])
    x2, x2_p, y2, y2_p = get_matching_pair_coordinates(kp_query, kp_train, matches[i + 1])
    x3, x3_p, y3, y3_p = get_matching_pair_coordinates(kp_query, kp_train, matches[i + 2])
    x4, x4_p, y4, y4_p = get_matching_pair_coordinates(kp_query, kp_train, matches[i + 3])

    # Assemble A matrix
    A_1 = assemble_A_matrices(x1, x1_p, y1, y1_p)
    A_2 = assemble_A_matrices(x2, x2_p, y2, y2_p)
    A_3 = assemble_A_matrices(x3, x3_p, y3, y3_p)
    A_4 = assemble_A_matrices(x4, x4_p, y4, y4_p)

    A = np.concatenate((A_1, A_2, A_3, A_4), axis=0)

    # Compute singular Value Decomposition
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = vh[-1, :]

    # Test h vector
    test_sol = np.matmul(A, h)
    if any(test_sol.flatten() < 0.0001):
        print("H vector error!")

    # Assemble H Matrix
    H = assemble_H_matrix(h)
    return H

    # kp_query, kp_train, matches, best_inlier_points

def compute_best_homography_matrix(kp_query, kp_train, matches, best_inliers=None):
    # First iteration
    if best_inliers:
        x, x_p, y, y_p = best_inliers[0]
        # Assemble A matrix
        A = assemble_A_matrices(x, x_p, y, y_p)
        for k in range(1, len(best_inliers)):
            x, x_p, y, y_p = best_inliers[k]
            # Assemble A matrix
            A_ = assemble_A_matrices(x, x_p, y, y_p)
            A = np.concatenate((A, A_), axis=0)
    else:
        x, x_p, y, y_p = get_matching_pair_coordinates(kp_query, kp_train, matches[random.randint(0, 11)])
        A = assemble_A_matrices(x, x_p, y, y_p)
        for i in range(5):
            random_index = random.randint(0, 11)
            x, x_p, y, y_p = get_matching_pair_coordinates(kp_query, kp_train, matches[random_index])
            A_ = assemble_A_matrices(x, x_p, y, y_p)
            A = np.concatenate((A, A_), axis=0)


    # Compute singular Value Decomposition
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = vh[-1, :]

    # Test h vector
    test_sol = np.matmul(A, h)
    print("H vector pixel error: {}".format(test_sol))

    # Assemble H Matrix
    H = assemble_H_matrix(h)
    return H

# Made my RANSAC Function as robust as possible!! Put most time in this..
def compute_H_from_RANSAC(kp_query, kp_train, matches):
    homography_matrices = []
    ransac_count = []
    inlier_points_list = []

    for i in range(100):
        H = compute_best_homography_matrix(kp_query, kp_train, matches)
        homography_matrices.append(H)
        inlier_count, inlier_points = count_number_of_inlers(kp_query, kp_train, matches, H)
        ransac_count.append(inlier_count)
        inlier_points_list.append(inlier_points)
    # Return H with most votes
    best_H = homography_matrices[ransac_count.index(max(ransac_count))]
    best_inlier_points = inlier_points_list[ransac_count.index(max(ransac_count))]
    # Find best H from inlier points that agreed with highest voted H
    best_H = compute_best_homography_matrix(kp_query, kp_train, matches, best_inlier_points)
    return best_H
    #

def compute_homography_transform(H, query_vector):
    query_vector = np.asarray(query_vector)
    # query_vector = np.array([[query_vector[0]], [query_vector[1]], [query_vector[2]]])
    transform_vector = np.matmul(H, query_vector)
    transform_vector = transform_vector / transform_vector[2]
    return transform_vector


def compute_size_of_mosaic(train_image, image, H):
    size_train_image = np.shape(train_image)
    max_x = size_train_image[1]
    max_y = size_train_image[0]
    min_x = 0
    min_y = 0
    size_query_image = np.shape(image)
    max_x_query_image = size_query_image[1]
    max_y_query_image = size_query_image[0]

    # Find projection of query image corners to determine max and min x and y values
    top_left_query = compute_homography_transform(H, [[0], [0], [1]])
    top_right_query = compute_homography_transform(H, [[max_x_query_image], [0], [1]])
    bottom_left_query = compute_homography_transform(H, [[0], [max_y_query_image], [1]])
    bottom_right_query = compute_homography_transform(H, [[max_x_query_image], [max_y_query_image], [1]])

    # Determine max and min x and y values
    min_x = min(float(top_left_query[0]), float(bottom_left_query[0]), min_x)
    min_y = min(float(top_left_query[1]), float(top_right_query[1]), min_y)
    max_x = max(float(top_right_query[0]), float(bottom_right_query[0]), max_x)
    max_y = max(float(bottom_right_query[1]), float(bottom_left_query[1]), max_y)

    # Compute new mosaic size
    mosaic_x_range = int(math.ceil(max_x + abs(min_x)))
    mosaic_y_range = int(math.ceil(max_y + abs(min_y)))

    # Compute x and y offset from training image
    train_image_x_offset = abs(min_x)
    train_image_y_offset = abs(min_y)

    return mosaic_x_range, mosaic_y_range, train_image_x_offset, train_image_y_offset, size_train_image[1], size_train_image[0]


# Putting indices into 3x3 matrix to directly multiply by H and get new index value
def assemble_indices_matrix(query_gray_image):
    size_image = np.shape(query_gray_image)
    x_indices_matrix = np.fromfunction(lambda j, i: i, (size_image[0], size_image[1]))
    y_indices_matrix = np.fromfunction(lambda j, i: j, (size_image[0], size_image[1]))
    ones_matrix = np.ones((size_image[0], size_image[1]))
    indices_matrix = np.stack((x_indices_matrix, y_indices_matrix, ones_matrix))
    print(indices_matrix)
    return indices_matrix

# Using Equalization image function..
def equalize_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


def main():
    # Setup Log
    logging.basicConfig(filename='Stitch_image.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    # Setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True,
                    help="image directory")

    args = vars(ap.parse_args())

    # Save Parameters
    input_directory = args['directory']

    # Print input parameters to log
    logging.info('Input Parameters:\n'
                 '\tDirectory: {}\n'
                 .format(input_directory))

    # Create output image folder if doesn't exits
    output_image_folder = os.path.join(os.path.dirname(input_directory), 'output_images/{}'.format(args['directory']))
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    ######
    image_directory = os.path.join(definitions.ROOT_DIR, input_directory)
    image_RGB_list = []
    image_GRAY_list = []
    for image_file_name in sorted(os.listdir(image_directory), reverse=True):
        if any(char.isdigit() for char in image_file_name):
            logging.info('Processing image {}'.format(image_file_name))
            image_path = os.path.join(image_directory, image_file_name)
            gray_image = cv2.imread(image_path, 0)
            image_GRAY_list.append(gray_image)
            rgb_image = cv2.imread(image_path)
            image_RGB_list.append(rgb_image)


    # Test 2 images
    orb = cv2.ORB_create()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    H_list = []
    mosaic_image = [image_RGB_list[0]]
    offset_list = []

    for i in range(len(image_GRAY_list)-1):
        # find the keypoints and descriptors with SIFT
        train_GRAY_image = image_GRAY_list[i]
        train_BGR_image = equalize_image(image_RGB_list[i])
        query_GRAY_image = image_GRAY_list[i+1]
        query_BGR_image = equalize_image(image_RGB_list[i+1])
        kp_train, des_train = orb.detectAndCompute(train_GRAY_image, None)
        kp_query, des_query = orb.detectAndCompute(query_GRAY_image, None)

        # Match descriptors.
        matches = bf.match(des_query, des_train)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Compute H Matrix implementing RANSAC algorithm
        H = compute_H_from_RANSAC(kp_query, kp_train, matches)
        if not H_list:
           pass
        else:
            H = np.matmul(H_list[i-1], H)
        H_list.append(H)

        # Compute size of mosaic
        new_mosaic_x_range, new_mosaic_y_range, train_image_x_offset, train_image_y_offset, x,y \
            = compute_size_of_mosaic(mosaic_image[i], query_GRAY_image, H_list[i])
        train_image_x_offset = int(round(train_image_x_offset))
        train_image_y_offset = int(round(train_image_y_offset))

        # new_mosaic_x_range = int(math.ceil(np.shape(mosaic_image[i])[1]))
        # new_mosaic_y_range = int(math.ceil(np.shape(mosaic_image[i])[0]))

        transform_array = np.array([[1, 0, train_image_x_offset],
                                    [0, 1, train_image_y_offset],
                                    [0, 0, 1]])

        im_out = cv2.warpPerspective(query_BGR_image, transform_array.dot(H_list[i]),
                                     (new_mosaic_x_range, new_mosaic_y_range))


        im_out[train_image_y_offset:y + train_image_y_offset,
        train_image_x_offset:x + train_image_x_offset] = mosaic_image[i]
        #
        # a= im_out[train_image_y_offset:y + train_image_y_offset,
        # train_image_x_offset:x + train_image_x_offset]
        #
        # for i in a:
        #     for j in a[i]:
        #         if a.all([i,j,0]==0):
        #             a[i,j,0] += mosaic_image[i,j,0]
        #             a[i,j,1] += mosaic_image[i,j,1]
        #             a[i,j,2] += mosaic_image[i,j,2]

        offset_list.append([train_image_y_offset, train_image_x_offset])

        save_stiched_image(output_image_folder, im_out, i)

        mosaic_image.append(im_out)



if __name__ == '__main__':
    main()
