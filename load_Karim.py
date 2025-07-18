import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data
from scipy.io import loadmat

training_snr = [30]
channel_model_list = ['CDL-A']
Doppler_Shift_list = [75]
RX_Ants_list = [4]
sc_counter = 1

for ch in range(len(channel_model_list)):
    channel = channel_model_list[ch]
            
    for do in range(len(Doppler_Shift_list)):
        doppler = Doppler_Shift_list[do]

        for rx_at in range(len(RX_Ants_list)):
            rx = RX_Ants_list[rx_at]
                    
            for ttsnr in range(len(training_snr)):
                tsnr = training_snr[ttsnr]

                print("Training scenario [",sc_counter,"]: Channel Model: ", channel, " - Doppler Shift: ", doppler, " - RX Ants: ", rx, " - Training SNR", tsnr)
                sc_counter = sc_counter + 1
                
                
                # Load Dataset from Matlab
                mat = loadmat('C:/Users/abdul-karim.gizzini/Desktop/JCESD-main/JCESD-main/GenerateH/MatlabCode/Datasets/{}{}dB_{}Hz_R_{}.mat'.format(channel, tsnr, doppler, rx))
               
                Hls = mat['Hls']
                Ideal_H = mat['Ideal_H']
                Ideal_X = mat['Ideal_X']
                Received_Y = mat['Received_Y']
                Transmit_X = mat['Transmit_X']
                print("Hls: ", Hls.shape)
                print("Ideal_H: ", Ideal_H.shape)
                print("Ideal_X: ", Ideal_X.shape)
                print("Received_Y: ", Received_Y.shape)
                print("Transmit_X: ", Transmit_X.shape)
  
