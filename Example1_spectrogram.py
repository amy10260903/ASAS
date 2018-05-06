# -*- coding: utf-8 -*-
"""
Example code 1:
    1. plot spectrogram
    2. plot log-scaled spectrogram with triangular filterbanks
    3. some I/O tchniques

Li SU 2018/03/05
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

def load_audio(filename):

    # You may use the audio I/O packages 
    fs, x = wav.read(filename)
    if x.dtype != 'float32':
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    return x, fs

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, np.round(N/2), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1                                             
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
    tfr = scipy.fftpack.fft(tfr, n=N, axis=0)  # L2 norm                        
    return tfr, f, t, N

def Spectrogram(x, fr, fs, Hop, h): 
    tfr, f, t, N = STFT(x, fr, fs, Hop, h)   
    tfr = abs(tfr)**2
    tfr = tfr[:int(round(N/2)),:] # positive frequency
    return tfr, f, t, N

def Inst_energy(x, fr, fs, Hop, h): 
    tfr, f, t, N = STFT(x, fr, fs, Hop, h)  
    tfr = abs(tfr)**2
    E = np.sum(tfr, axis=0)
    E = np.sqrt(E)
    return t, E

def LogFreqMapping(tfr, f, fr, f_low, f_high, NumPerOct):
    StartFreq = f_low
    StopFreq = f_high
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest+2):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        central_freq.append(CenFreq)

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-2, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i-1, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i-1, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    central_freq = central_freq[1:-1]
    return tfrL, central_freq

def plot_spectrogram(tfr, f, t, fs):
    axis_font = {'fontname':'Arial', 'size':'16'}
    plt.figure(figsize=(6,4), dpi=100)
    plt.pcolormesh(t/fs, f/1000, np.log(tfr), cmap='Purples')
    ax=plt.axes()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(16)    
    plt.xlabel("Time (s)", **axis_font)
    plt.ylabel("Frequency (kHz)", **axis_font)
    return 0

def plot_logspec(tfr, f, t):
    axis_font = {'fontname':'Arial', 'size':'16'}
    plt.figure(figsize=(6,4), dpi=100)
    plt.pcolormesh(t/44100, f, np.log(tfr+1E-10), cmap='plasma')
    ax=plt.axes()
    ax.semilogy()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(16)    
    plt.xlabel("Time (s)", **axis_font)
    plt.ylabel("Frequency (Hz)", **axis_font)
    return 0

def plot_E(t, fs, E):
    axis_font = {'fontname':'Arial', 'size':'16'}
    plt.figure(figsize=(6,4), dpi=100)
    plt.plot(t/fs, E)
    plt.xlabel("Time (s)", **axis_font)
    plt.ylabel("RMS energy", **axis_font)
    return 0

if __name__== "__main__":
    filename = "01-AchGottundHerr.wav"
    Hop = 4097 # hop size (in sample)
    Win = 4097
    fr = 4
    h = scipy.signal.blackmanharris(Win)
    x, fs = load_audio(filename)
    
    # spectrogram
    f_high = 2000
    tfr, f, t, Nfft = Spectrogram(x, fr, fs, Hop, h) # Nfft: number of FFT points
    plot_spectrogram(tfr, f, t, fs)
    
#    plot_spectrogram(tfr[:1000,:], f[:1000], t, fs)
    
    f_low = 200
    f_high = 2000
    NumPerOctave = 24

    # log-frequency spectrogram with triangular filterbank
    tfrL, central_frequencies = LogFreqMapping(tfr, f, fr, \
                                               f_low, f_high, \
                                               NumPerOctave)
    plot_logspec(tfrL, central_frequencies, t)
    
    t, E = Inst_energy(x, fr, fs, Hop, h)
    plot_E(t, fs, E)
#    os.system("start " + filename) # play audio