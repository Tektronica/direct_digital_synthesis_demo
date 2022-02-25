import numpy as np


########################################################################################################################
def windowed_fft(yt, Fs, N, windfunc='blackman'):
    """
    :param yt: time series data
    :param Fs: sampling frequency
    :param N: number of samples, or the length of the time series data
    :param windfunc: the chosen windowing function
    :return:
    xf_fft  : Two sided frequency axis.
    yf_fft  : Two sided power spectrum.
    xf_rfft : One sided frequency axis.
    yf_rfft : One sided power spectrum.
    main_lobe_width : The bandwidth (Hz) of the main lobe of the frequency domain window function.
    """

    # remove DC offset
    yt -= np.mean(yt)

    # Calculate windowing function and its length ----------------------------------------------------------------------
    if windfunc == 'rectangular':
        w = np.ones(N)
        main_lobe_width = 2 * (Fs / N)
    elif windfunc == 'bartlett':
        w = np.bartlett(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'hanning':
        w = np.hanning(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'hamming':
        w = np.hamming(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'blackman':
        w = np.blackman(N)
        main_lobe_width = 6 * (Fs / N)
    else:
        # TODO - maybe include kaiser as well, but main lobe width varies with alpha
        raise ValueError("Invalid windowing function selected!")

    # Calculate amplitude correction factor after windowing ------------------------------------------------------------
    # https://stackoverflow.com/q/47904399/3382269
    amplitude_correction_factor = 1 / np.mean(w)

    # Calculate the length of the FFT ----------------------------------------------------------------------------------
    if (N % 2) == 0:
        # for even values of N: FFT length is (N / 2) + 1
        fft_length = int(N / 2) + 1
    else:
        # for odd values of N: FFT length is (N + 1) / 2
        fft_length = int((N + 2) / 2)

    """
    Compute the FFT of the signal Divide by the length of the FFT to recover the original amplitude. Note dividing 
    alternatively by N samples of the time-series data splits the power between the positive and negative sides. 
    However, we are only looking at one side of the FFT.
    """
    try:
        yf_fft = (np.fft.fft(yt * w) / fft_length) * amplitude_correction_factor
        xf_fft = np.round(np.fft.fftfreq(N, d=1. / Fs), 6)  # two-sided

        yf_rfft = yf_fft[:fft_length]
        xf_rfft = np.round(np.fft.rfftfreq(N, d=1. / Fs), 6)  # one-sided

    except ValueError as e:
        print('\n!!!\nError caught while performing fft of presumably length mismatched arrays.'
              '\nwindowed_fft method in distortion_calculator.py\n!!!\n')
        raise ValueError(e)

    return xf_fft, yf_fft, xf_rfft, yf_rfft, main_lobe_width
