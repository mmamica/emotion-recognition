import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
import utils
import cv2
import random
import shutil


def azim_proj(center_point, point):
    '''
    counts azimuthal projection
    :param center_point: a relative, center point.
    :param point: a point, which coorinates are counted relatively to center_point
    :return: x and y coordinates
    '''
    azimuth = utils.count_azimuth(center_point, point)
    distance = utils.count_distance(center_point, point)
    x = distance*np.sin(azimuth)
    y = -distance*np.cos(azimuth)
    return x, y


def transfer(source, dest, split_rate):
    '''
    transfers files between given directories
    :param source: directory from which files are transfered
    :param dest: directory to which files are transfered
    :param split_rate: rate saying how many files should be transfered (files are taken randomly to match the split rate)
    :return:
    '''
    source_files = os.listdir(source)
    if (len(source_files) != 0):
        transfer_file_numbers = int(len(source_files) * split_rate)
        transfer_index = random.sample(range(0, len(source_files)), transfer_file_numbers)
        for index in transfer_index:
            shutil.move(source + str(source_files[index]), dest + str(source_files[index]))


def transfer_all_class(source, dest, split_rate, class_labels):
    '''

    :param source: directory from which files are transfered
    :param dest: directory to which files are transfered
    :param split_rate: rate saying how many files should be transfered (files are taken randomly to match the split rate)
    :param class_labels: classes given to classification (subfolders in a directory tree)
    :return:
    '''
    for label in class_labels:
        transfer(source + '/' + label + '/',
                 dest + '/' + label + '/',
                 split_rate)


def prepare_data_labels(classes, train_dir_name):
    '''
    prepare arraylike structures with names of images and classes assigned to them
    :param classes: array containing all classification classes
    :param train_dir_name: path to the directory containing training images
    :return: X array with images' names, Y array with assigned classes
    '''
    X=[]
    Y=[]
    for class_label in classes:
        source_files=os.listdir(train_dir_name+"/"+class_label)
        for val in source_files:
            X.append(val)
            if(class_label==classes[0]):
                Y.append(0)
            elif(class_label==classes[1]):
                Y.append(1)
    return X, Y


def gen_image(coordinates, values, channels, bands):
    '''
    generate image with the use of clough toucher interpolation
    :param coordinates: coordinates for which values are calculated
    :param values: values assigned to given coordinates for different frequency bands
    :param channels: number of channels used in EEG examination
    :param bands: number of frequency bands
    :return: an image with the number of layers equal to the bands number
    '''
    scaler = MinMaxScaler()
    coordinates = scaler.fit_transform(coordinates)
    output = np.zeros((channels,channels,bands))
    for i in range(bands):
        data = values[:,i]
        xi = yi = np.arange(0,1,0.032)
        xi,yi = np.meshgrid(xi,yi)
        zi = griddata(coordinates,data,(xi,yi),method='cubic', fill_value = 0)
        scaled_zi = scaler.fit_transform(zi)
        output[:,:,i] = scaled_zi
    output = 255 * output
    output = output.astype(np.uint8)
    return output


def gen_data(data, ratings, coordinates_2d, participant, category, fs, channels, window, path):
    '''
    generate images for given parameters and save them in given destination
    :param data: raw EEG data
    :param ratings: trials ratings done by examined participant
    :param coordinates_2d: EEG sensors positions projected into 2D space
    :param participant: participant id (1,2,3...32)
    :param category: emotion indicator (valence, arousal)
    :param fs: frequency sampling
    :param channels: number of channels
    :param window: size of the window for EEG data preprocessing
    :param path: relative path, where data should be saved
    :return:
    '''
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
                data_temp = data[t][i][fs*s:fs*(s+window)]
                theta, alpha, beta, gamma = utils.psd_bands_periodogram(data_temp, fs)
                alphas[i] = alpha
                betas[i] = beta
                gammas[i] = gamma
                thetas[i] = theta
            features = np.hstack((thetas, alphas, betas, gammas))
            image = gen_image(coordinates_2d,features,channels,4)
            path_name = path + "/" + category
            path_name = path_name + "/train"
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