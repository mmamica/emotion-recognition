import numpy as np
from scipy.interpolate import griddata,CloughTocher2DInterpolator
from sklearn.preprocessing import scale, normalize, MinMaxScaler
import math as m
from utils import cart2sph, pol2cart

def azim_proj(pos):
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

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
    return output