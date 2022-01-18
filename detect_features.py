from os.path import dirname, join, realpath
import shutil
import cv2
import numpy as np
from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import h5py
from hloc.utils.parsers import names_to_pair


def main(I1,fpath,params):
    dataset = join(dirname(realpath(__file__)),'datasets')
    outputs = join(dirname(realpath(__file__)),'outputs')

    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(outputs, ignore_errors=True)

    try:
        os.mkdir(dataset)
        os.mkdir(outputs)
    except:
        pass

    cv2.imwrite(join(dataset,'image.jpg'),I1)

    dataset = Path(dataset)
    images = dataset 

    outputs = Path(outputs)
    # pairs = outputs / 'pairs.txt'
    # np.savetxt(str(pairs) ,[['I1.jpg','I2.jpg']], fmt="%s")

    if params == 'R2D2':
        feature_conf = extract_features.confs['r2d2']
    elif params == 'SuperPoint+NN':
        feature_conf = extract_features.confs['superpoint_max']
    elif params == 'SuperPoint+superglue':
        feature_conf = extract_features.confs['superpoint_max']
    elif params == 'D2-Net':
        feature_conf = extract_features.confs['d2net-ss']

    features = extract_features.main(feature_conf, images, outputs, feature_path=Path(fpath))   

    feature_file = h5py.File(features, 'r')
    kp1 = feature_file['image.jpg']['keypoints'].__array__()
    kp1 = np.array([cv2.KeyPoint(int(kp1[i,0]),int(kp1[i,1]),3) for i in range(len(kp1))])

    return kp1


if __name__ == "__main__":
    main()