import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color


### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        distances = np.linalg.norm(features[:, None] - centers, axis=2)
        new_assignments = np.argmin(distances, axis=1)

        for i in range(k):
            if np.sum(new_assignments == i) > 0:
                centers[i] = np.mean(features[new_assignments == i], axis=0)

        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments
        ### END YOUR CODE

    return assignments


### Clustering Methods for colorful image
def kmeans_color(features, k, num_iters=500):
    original_shape = features.shape
    features = features.reshape(-1, features.shape[-1])
    N = features.shape[0]
    assignments = np.zeros(N, dtype=np.uint32)
    # Like the kmeans function above
    ### YOUR CODE HERE
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    for n in range(num_iters):
        # Assign each point to the closest center
        distances = np.linalg.norm(features[:, None] - centers, axis=2)
        new_assignments = np.argmin(distances, axis=1)
        # Compute new center of each cluster
        centers = np.array(
            [features[new_assignments == i].mean(axis=0) for i in range(k)])

        # Stop if cluster assignments did not change
        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments
    assignments = assignments.reshape(original_shape[0:2])
    ### END YOUR CODE

    return assignments


# 找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while shift.all() > t:
        # 计算当前点和所有点之间的距离
        # 并筛选出在半径r内的点，计算mean vector（这里是最简单的均值，也可尝试高斯加权）
        # 用新的center（peak）更新当前点，直到满足要求跳出循环
        ### YOUR CODE HERE
        dist = np.linalg.norm(dataT - data_point, axis=1)
        new_data_point = np.mean(dataT[dist < r], axis=0)
        shift = new_data_point - data_point
        data_point = new_data_point
        ### END YOUR CODE

    return data_pointT.T


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = []  # 聚集的类中心
    label_no = 1  # 当前label
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through
    for idx in range(0, len(data.T)):
        # 遍历数据，寻找当前点的peak
        # 并实时关注当前peak是否会收敛到一个新的聚类（和已有peaks比较）
        # 若是，更新label_no，peaks，labels，继续
        # 若不是，当前点就属于已有类，继续
        ### YOUR CODE HERE
        peak = findpeak(data, idx, r)
        peakT = np.concatenate(peak, axis=0).T
        distances = np.linalg.norm(np.array(peaks) - np.array(peakT), axis=1)
        if np.any(distances < r):
            labels[idx] = np.argmin(distances) + 1
        else:  # 属于新聚类
            label_no += 1
            peaks.append(peakT)
            labels[idx] = label_no
        ### END YOUR CODE
    # print(set(labels))
    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img = color.lab2rgb(segmented_image)
    res_img = color.rgb2gray(res_img)
    return res_img


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    n_pixels = mask.shape[0] * mask.shape[1]
    n_correct_pixels = np.sum(mask_gt == mask)
    accuracy = n_correct_pixels / n_pixels
    ### END YOUR CODE

    return accuracy
