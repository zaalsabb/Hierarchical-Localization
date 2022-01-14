from os.path import dirname, join, realpath
import shutil
import cv2
import numpy as np
from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import h5py
from hloc.utils.parsers import names_to_pair

# class Match:
#     def __init__(self, queryIdx, trainIdx):
#         self.queryIdx = queryIdx
#         self.trainIdx = trainIdx

# class FeaturesPoint:
#     def __init__(self, x, y):
#         self.pt = [int(x),int(y)]

def main(I1,I2,params):
    dataset = join(dirname(realpath(__file__)),'datasets')
    outputs = join(dirname(realpath(__file__)),'outputs')

    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(outputs, ignore_errors=True)

    os.mkdir(dataset)
    os.mkdir(outputs)

    cv2.imwrite(join(dataset,'I1.jpg'),I1)
    cv2.imwrite(join(dataset,'I2.jpg'),I2)

    dataset = Path(dataset)
    images = dataset 

    outputs = Path(outputs)
    pairs = outputs / 'pairs.txt'
    np.savetxt(str(pairs),[['I1.jpg','I2.jpg']], fmt="%s")

    if params == 'R2D2':
        feature_conf = extract_features.confs['r2d2']
        matcher_conf = match_features.confs['NN-ratio']
    elif params == 'SuperPoint+NN':
        feature_conf = extract_features.confs['superpoint_max']
        matcher_conf = match_features.confs['NN-superpoint']
    elif params == 'SuperPoint+superglue':
        feature_conf = extract_features.confs['superpoint_max']
        matcher_conf = match_features.confs['superglue']
    elif params == 'D2-Net':
        feature_conf = extract_features.confs['d2net-ss']
        matcher_conf = match_features.confs['NN-ratio']

    features = extract_features.main(feature_conf, images, outputs)
    matches = match_features.main(
        matcher_conf, pairs, feature_conf['output'], outputs)    

    feature_file = h5py.File(features, 'r')
    kp1 = feature_file['I1.jpg']['keypoints'].__array__()
    kp2 = feature_file['I2.jpg']['keypoints'].__array__()

    pair = names_to_pair('I1.jpg','I2.jpg')
    match_file = h5py.File(matches, 'r')
    matches1 = match_file[pair]['matches0'].__array__()

    kp1 = np.array([cv2.KeyPoint(int(kp1[i,0]),int(kp1[i,1]),3) for i in range(len(kp1))])
    kp2 = np.array([cv2.KeyPoint(int(kp2[i,0]),int(kp2[i,1]),3) for i in range(len(kp2))])
    matches = [cv2.DMatch(i,m,0) for i,m in enumerate(matches1) if m != -1]

    return matches, kp1, kp2


if __name__ == "__main__":
    main()