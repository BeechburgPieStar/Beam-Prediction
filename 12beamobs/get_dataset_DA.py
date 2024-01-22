import torch
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random
def rotate_matrix(theta):
    m = np.zeros((2,2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    print(np.shape(x))
    x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
    print(np.shape(x))
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))

    x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
    x_rotate1 = x_rotate1.reshape(x_rotate1.shape[0], x_rotate1.shape[2], x_rotate1.shape[1])
    x_rotate2 = x_rotate2.reshape(x_rotate2.shape[0], x_rotate2.shape[2], x_rotate2.shape[1])
    x_rotate3 = x_rotate3.reshape(x_rotate3.shape[0], x_rotate3.shape[2], x_rotate3.shape[1])
    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))  

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

def Gaussian_DA(x, y):
    [N, L, C] = np.shape(x)

    gaussian_noise1 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise1[n,:,:] = np.random.normal(0, 0.0005, size=(1, L, C))

    gaussian_noise2 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise2[n,:,:] = np.random.normal(0, 0.001, size=(1, L, C))

    gaussian_noise3 = np.zeros([N, L, C])
    for n in range(N):
        gaussian_noise3[n,:,:] = np.random.normal(0, 0.002, size=(1, L, C))


    x_DA = np.vstack((x, x+gaussian_noise1, x+gaussian_noise2, x+gaussian_noise3))  

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

def Flip_DA(x, y):
    [N, L, C] = np.shape(x)
####Flip
    x_flip_h = np.zeros([N, L, C])
    x_flip_h[:,:,0] = -x[:,:,0]
    x_flip_h[:,:,1] = x[:,:,1]

    x_flip_v = np.zeros([N, L, C])
    x_flip_v[:,:,0] = x[:,:,0]
    x_flip_v[:,:,1] = -x[:,:,1]

    x_flip_hv = np.zeros([N, L, C])
    x_flip_hv[:,:,0] = -x[:,:,0]
    x_flip_hv[:,:,1] = -x[:,:,1]

    x_DA = np.vstack((x, x_flip_h, x_flip_v, x_flip_hv))  

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

def Shift_DA(x, y):
    [N, C, L] = np.shape(x)

    x_shift1 = np.zeros((N, C, L))
    x_shift1[:,:, 0 : int(3*L/4)]= x[:, :, int(L/4):L]
    x_shift1[:,:, int(3*L/4) : L]= x[:, :, 0 : int(L/4)]

    x_shift2 = np.zeros((N, C, L))
    x_shift2[:,:, 0: int(2*L/4)]= x[:, :, int(2*L/4) : L]
    x_shift2[:,:, int(2*L/4): L]= x[:, :, 0 : int(2*L/4)]

    x_shift3 = np.zeros((N, C, L))
    x_shift3[:,:, 0: int(1*L/4)]= x[:, :, int(3*L/4) : L]
    x_shift3[:,:, int(1*L/4): L]= x[:, :, 0 : int(3*L/4)]

    x_DA = np.vstack((x, x_shift1, x_shift2,x_shift3))  

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA


def TrainDataset_Rotate(r):
    x = np.load(f'dataset/x_r={r}.npy')
    x = np.squeeze(x)
    y = np.load(f'dataset/y_r={r}.npy')
    y = np.squeeze(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_train, y_train = Rotate_DA(x_train, y_train)
    return x_train, x_val, y_train, y_val

def TrainDataset_CS(r):
    x = np.load(f'dataset/x_r={r}.npy')
    x = np.squeeze(x)
    y = np.load(f'dataset/y_r={r}.npy')
    y = np.squeeze(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_train, y_train = Shift_DA(x_train, y_train)
    return x_train, x_val, y_train, y_val


def TrainDataset_Flip(r):
    x = np.load(f'dataset/x_r={r}.npy')
    x = np.squeeze(x)
    y = np.load(f'dataset/y_r={r}.npy')
    y = np.squeeze(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_train, y_train = Flip_DA(x_train, y_train)
    return x_train, x_val, y_train, y_val

def TrainDataset_Gaussian(r):
    x = np.load(f'dataset/x_r={r}.npy')
    x = np.squeeze(x)
    y = np.load(f'dataset/y_r={r}.npy')
    y = np.squeeze(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_train, y_train = Gaussian_DA(x_train, y_train)
    return x_train, x_val, y_train, y_val

def TestDataset():
    x = np.load(f"dataset/x_test.npy")
    x = np.squeeze(x)
    y = np.load(f"dataset/y_test.npy")
    y = np.squeeze(y)
    return x, y