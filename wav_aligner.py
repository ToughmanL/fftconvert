

"""
    该程序将录同一人同一段语音的两端录音对齐
    输入：两个不同长度wav文件 001.wav 002.wav
    输出：两个相同长度wav文件 001.wav 002.wav
"""

from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import os

"""
判断声音真正开始的地方,判断声音结束的地方
通过大于前两帧两倍的参数开始
"""

def _get_path(path):
    file_path=[]
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            if file.endswith(".wav"):
               file_path.append(os.path.join(root,file))
                # file_path.append(file)
    return file_path

def _wavcut(wavpath):
    rate, data = wavfile.read(wavpath)
    if rate != 8000:
        result=int(data.shape[0]/rate*8000)
        if result%2!=0:#偶数算的快
            result+=1
        data=signal.resample(data,result)
        rate = 8000
    #取前1600采样点也就是0.2s的平均值
    headframe=np.abs(data[0:1600])
    headthresh=np.mean(headframe)
    tailframe=np.abs(data[-1600:])
    tailthresh = np.mean(tailframe)
    end=len(data)
    for i in range(int(len(data)/10)):
        if np.mean(np.abs(data[i:i+80]))>headthresh*10:
            begin=i
            break
    for i in range(end,int(0.9*end),-1):
        if np.mean(np.abs(data[i-80:i]))>tailthresh*10:
            end=i
            break
    newdata=data[begin:end]
    return newdata,rate

def _aliggner(data1,data2):
    len1=len(data1)
    len2=len(data2)
    difflen = len1 - len2
    if difflen==0:
        return data1, data2
    elif abs(difflen)>1000:
        begin=0
        if difflen>0:
            difflen=len1-len2
            maxsum=np.sum(np.abs(data1[0:(len1-difflen)]-data2))
            for i in range(0,difflen,int(difflen/100)):
                datasum=np.sum(np.abs(data1[i:(i+len2)]-data2))
                if datasum>maxsum:
                    maxsum=datasum
                    begin=i
            data1=data1[begin:begin+len2]
        else:
            difflen = len2 - len1
            maxsum = np.sum(np.abs(data2[0:(len2 - difflen)] - data1))
            for i in range(0, difflen, int(difflen / 100)):
                datasum = np.sum(np.abs(data2[i:(i + len1)] - data1))
                if datasum > maxsum:
                    maxsum = datasum
                    begin = i
            data2 = data1[begin:begin + len1]
    else:
        if difflen>0:
            data1=data1[0:len2]
        else:
            data2=data2[0:len1]
    return data1, data2


def main():
    # channelpath1="E:\\Data\\phone"
    # channelpath2 = "E:\\Data\\app"
    # newfiledir="C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\wav\\"
    channelpath1 = "E:\\Data\\phone"
    channelpath2 = "E:\\Data\\app"
    newfiledir = "C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\wav\\"
    file_list1=_get_path(channelpath1)
    file_list2 = _get_path(channelpath2)
    if len(file_list1)!=len(file_list2):
        raise Exception("文件数目不对应",)

    for i in range(51,len(file_list1),1):
        wavpath1=file_list1[i]
        wavpath2=file_list2[i]
        data1,rate1 = _wavcut(wavpath1)
        data2,rate2 = _wavcut(wavpath2)
        data1, data2=_aliggner(data1, data2)
        data1 = data1.astype(np.int16)
        data2 = data2.astype(np.int16)
        newwavpath1=newfiledir+wavpath1.split("\\")[-1].replace(".wav","_1new.wav")
        newwavpath2=newfiledir+wavpath2.split("\\")[-1].replace(".wav","_2new.wav")
        wavfile.write(newwavpath1, rate1, data1)
        wavfile.write(newwavpath2, rate2, data2)
        print("completed: {com}%".format(com=i/len(file_list1)*100))

if __name__ == '__main__':
    main()