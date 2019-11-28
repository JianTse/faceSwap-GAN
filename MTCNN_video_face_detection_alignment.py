# Source: https://github.com/davidsandberg/facenet/blob/master/src/align/

""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K

from umeyama import umeyama
import mtcnn_detect_face

def create_mtcnn(sess, model_path):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = mtcnn_detect_face.PNet({'data':data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = mtcnn_detect_face.RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = mtcnn_detect_face.ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet


WEIGHTS_PATH = "./mtcnn_weights/"

sess = K.get_session()
with sess.as_default():
    global pnet, rnet, onet
    pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

#global pnet, rnet, onet
pnet = K.function([pnet.layers['data']], [pnet.layers['conv4-2'], pnet.layers['prob1']])
rnet = K.function([rnet.layers['data']], [rnet.layers['conv5-2'], rnet.layers['prob1']])
onet = K.function([onet.layers['data']], [onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])

def get_src_landmarks(x0, x1, y0, y1, pnts):
    """
    x0, x1, y0, y1: (smoothed) bbox coord.
    pnts: landmarks predicted by MTCNN
    """
    src_landmarks = [(int(pnts[i + 5][0] - x0),
                      int(pnts[i][0] - y0)) for i in range(5)]
    return src_landmarks


def get_tar_landmarks(img):
    """
    img: detected face image
    """
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
    ]

    img_size = img.shape
    tar_landmarks = [(int(xy[0] * img_size[0]),
                      int(xy[1] * img_size[1])) for xy in ratio_landmarks]
    return tar_landmarks


def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks):
    """
    umeyama(src, dst, estimate_scale)
    landmarks coord. for umeyama should be (width, height) or (y, x)
    """
    src_size = src_im.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
    result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE)
    return result


def process_mtcnn_bbox(bboxes, im_shape):
    """
    output bbox coordinate of MTCNN is (y0, x0, y1, x1)
    Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
    """
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h) / 2
        center = (int((x1 + x0) / 2), int((y1 + y0) / 2))
        new_x0 = np.max([0, (center[0] - length // 2)])  # .astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0] + length // 2)])  # .astype(np.int32)
        new_y0 = np.max([0, (center[1] - length // 2)])  # .astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1] + length // 2)])  # .astype(np.int32)
        bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes

frames = 0
save_interval = 6
def process_video(input_img):
    global frames, save_interval
    global pnet, rnet, onet
    minsize = 30  # minimum size of face
    detec_threshold = 0.7
    threshold = [0.6, 0.7, detec_threshold]  # three steps's threshold
    factor = 0.709  # scale factor

    frames += 1
    if frames % save_interval == 0:
        faces, pnts = mtcnn_detect_face.detect_face(
            input_img, minsize, pnet, rnet, onet, threshold, factor)
        faces = process_mtcnn_bbox(faces, input_img.shape)

        for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]

            # get src/tar landmarks
            src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts)
            tar_landmarks = get_tar_landmarks(det_face_im)

            # align detected face
            aligned_det_face_im = landmarks_match_mtcnn(
                det_face_im, src_landmarks, tar_landmarks)

            fname = f"./faces/aligned_faces/frame{frames}face{str(idx)}.jpg"
            #plt.imsave(fname, aligned_det_face_im, format="jpg")
            cv2.imwrite(fname, aligned_det_face_im)
            fname = f"./faces/raw_faces/frame{frames}face{str(idx)}.jpg"
            #plt.imsave(fname, det_face_im, format="jpg")
            cv2.imwrite(fname, det_face_im)

            bm = np.zeros_like(aligned_det_face_im)
            h, w = bm.shape[:2]
            bm[int(src_landmarks[0][0] - h / 15):int(src_landmarks[0][0] + h / 15),
            int(src_landmarks[0][1] - w / 8):int(src_landmarks[0][1] + w / 8), :] = 255
            bm[int(src_landmarks[1][0] - h / 15):int(src_landmarks[1][0] + h / 15),
            int(src_landmarks[1][1] - w / 8):int(src_landmarks[1][1] + w / 8), :] = 255
            bm = landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
            fname = f"./faces/binary_masks_eyes/frame{frames}face{str(idx)}.jpg"
            #plt.imsave(fname, bm, format="jpg")
            cv2.imwrite(fname, bm)

    return np.zeros((3, 3, 3))


video_fn = 'E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/bgr/2019-07-22-14-37-25_bgr.avi'
vid = cv2.VideoCapture(video_fn)
while True:
    return_value, input_img = vid.read()
    if return_value:
         process_video(input_img)
         cv2.imshow('img', input_img)
         cv2.waitKey(1)
    else:
        raise ValueError("No image!")

