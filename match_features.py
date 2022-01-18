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

def main(f_kp1,f_kp2,params):
    dataset = join(dirname(realpath(__file__)),'datasets')
    outputs = join(dirname(realpath(__file__)),'outputs')

    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(outputs, ignore_errors=True)

    os.mkdir(dataset)
    os.mkdir(outputs)

    dataset = Path(dataset)
    images = dataset 

    outputs = Path(outputs)
    pairs = outputs / 'pairs.txt'
    np.savetxt(str(pairs),[['I1','I2']], fmt="%s")

    features_combined = outputs / 'features.h5'

    if params == 'R2D2':
        matcher_conf = match_features.confs['NN-ratio']
    elif params == 'SuperPoint+NN':
        matcher_conf = match_features.confs['NN-superpoint']
    elif params == 'SuperPoint+superglue':
        matcher_conf = match_features.confs['superglue']
    elif params == 'D2-Net':
        matcher_conf = match_features.confs['NN-ratio']

    f1 = h5py.File(f_kp1, 'r')
    f2 = h5py.File(f_kp2, 'r')

    with h5py.File(features_combined,'w') as f:
        f1.copy('image.jpg',f, name='I1')
        f2.copy('image.jpg',f, name='I2')

    matches = match_features.main(
        matcher_conf, pairs, 'features', outputs)    

    pair = names_to_pair('I1','I2')
    match_file = h5py.File(matches, 'r')
    matches1 = match_file[pair]['matches0'].__array__()
    matches = [cv2.DMatch(i,m,0) for i,m in enumerate(matches1) if m != -1]
    
    return matches


if __name__ == "__main__":
    main()