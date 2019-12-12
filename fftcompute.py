
import numpy
import os
from scipy.io import wavfile
from scipy.fftpack import dct

def _get_path(path):
    file_path=[]
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            if file.endswith(".wav"):
               file_path.append(os.path.join(root,file))
                # file_path.append(file)
    return file_path

def _fft_computing(file):
    sample_rate, signal = wavfile.read(file)
    signal = signal[0:int(3.5 * sample_rate)]
    #预加重
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    #分帧
    frame_stride = 0.01
    frame_size = 0.025
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(
        float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    #加窗
    frames *= numpy.hamming(frame_length)

    #傅里叶变换
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return  pow_frames

def main():
    wav_path='C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\wav'
    fft_path='C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\fftfeat\\'
    file_list=_get_path(wav_path)
    for file in file_list:
        fft_feat=_fft_computing(file)
        print(fft_feat.shape)
        numpy.savetxt(fft_path+file.split("\\")[-1].replace(".wav",".txt"),\
                      fft_feat,delimiter=' ')
        # fft_feat1 = numpy.loadtxt(fft_path+file.split("\\")[-1].replace(".wav",".txt"), delimiter=' ')
        # print(fft_feat1.shape)
if __name__ == '__main__':
    main()