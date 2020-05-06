import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from sklearn.preprocessing import MinMaxScaler
import math as m
import utils
import cv2

def azim_proj(pos):
    [r, elev, az] = utils.cart2sph(pos[0], pos[1], pos[2])
    return utils.pol2cart(az, m.pi / 2 - elev)

def gen_image(coordinates, values, channels, bands):
    output = np.zeros((channels,channels,bands))
    for i in range(bands):
        data = values[:,i]
        CT_interpolator = CloughTocher2DInterpolator(coordinates, data)
        x2, y2 = np.linspace(-1, 1, channels), np.linspace(-1, 1, channels)
        X2, Y2 = np.meshgrid(x2, y2)
        interpolated = CT_interpolator(X2,Y2).reshape((channels, channels))
        scaler = MinMaxScaler()
        scaler.fit(interpolated)
        scaled = scaler.transform(interpolated)
        output[:,:,i] = scaled
    output = 255 * output # Now scale by 255
    output = output.astype(np.uint8)
    return output

def gen_data(data, ratings, coordinates_2d, participant, category, ratio, fs, channels, window, path):
    trials = data.shape[0]
    time = int(data.shape[2]/fs)
    np.random.seed()
    for t in range(trials):
        for s in range(0,time-int(window/2),int(window/2)):
            thetas = np.zeros((channels,1))
            alphas = np.zeros((channels,1))
            betas = np.zeros((channels,1))
            gammas = np.zeros((channels,1))
            for i in range(channels):
                data_temp = data[t][i][fs*s-1:fs*(s+window)]
                theta, alpha, beta, gamma = utils.psd_bands(data_temp, fs)
                alphas[i] = alpha
                betas[i] = beta
                gammas[i] = gamma
                thetas[i] = theta
            features = np.hstack((thetas, alphas, betas, gammas))
            image = gen_image(coordinates_2d,features,channels,4)
            path_name = path + "/" + category
            if(np.random.rand()<=ratio):
                path_name = path_name + "/train"
            else:
                path_name = path_name + "/test"
            if(category == "valence"):
                value = ratings[t,2]
                if(value<5):
                    path_name = path_name + "/low"
                else:
                    path_name = path_name + "/high"
            if(category == "arousal"):
                value = ratings[t,3]
                if(value<5):
                    path_name = path_name + "/low"
                else:
                    path_name = path_name + "/high"
            path_name = path_name+"/"+str(participant)+str(t)+str(s)+".png"
            cv2.imwrite(path_name,image)