from scipy.io import wavfile
import wave
import pyworld as pw
import numpy as np
import soundfile as sf
import heapq

filename = "./utterance/slowly_walk.wav"#正常語速_往前走.wav"
result = "./cut_speech/"

num_words = 8
#---------------------------------------------------------------------------------------------------
data, fs = sf.read(filename)
data = data.astype(np.float)
_f0, t = pw.dio(data, fs)
f0 = pw.stonemask(data, _f0, t, fs)
sp = pw.cheaptrick(data, f0, t, fs)
ap = pw.d4c(data, f0, t, fs)
#---------------------------------------------------------------------------------------------------
energy = []
for i in range(len(sp)):
    energy.append(sum(sp[i,:]))

with open(result + "energy.txt", "w") as f :
    for en in energy :
        f.write(str(en) + "\n")

EH = np.mean(energy)
print(EH)
#---------------------------------------------------------------------------------------------------
EH = EH / 8 #3.3
#EH = 1.1
EHF = []
flag = 0
for i in range(len(energy)):
    if len(EHF) == 0 and flag == 0 and energy[i] > EH :
        EHF.append(i)
        flag = 1
    elif flag == 0 and energy[i] > EH and i - 5 > EHF[-1]: #10 #5 #3
        EHF.append(i)
        flag = 1
    elif flag == 0 and energy[i] > EH and i - 5 <= EHF[-1]: #10 #5 #3
        EHF = EHF[:-1]
        flag = 1

    if flag == 1 and energy[i] < EH and i - 10 > EHF[-1]: #2 #3
        EHF.append(i)
        flag = 0
    elif flag == 1 and energy[i] < EH and i - 10 <= EHF[-1]: #2 #3
        EHF = EHF[:-1]
        flag = 0
#---------------------------------------------------------------------------------------------------
s = str(len(EHF)/2.0)
print("片段數：", s)

if s.split('.')[1] != '0':
    #EHF = EHF[:-1]
    EHF.append(i)

EHF = np.array(EHF) * 0.005
print("區間：", EHF)
#---------------------------------------------------------------------------------------------------
length = []
i = 0
while i < len(EHF) :
    length.append(EHF[i+1]-EHF[i])
    i = i + 2
average1 = np.mean(length)

if len(length) < num_words:
    d = num_words - len(length)
    ol = list(map(length.index, heapq.nlargest(d, length)))
else:
    ol = np.where(length > average1)[0]

one_word = [x for x in length if x <= average1]
average2 = np.mean(one_word)

word_sec = (average1 + average2) / 2

with open(result + "length.txt", "w") as f :
    for i in length:
        f.write(str(i) + "\n")

    f.write("\n" + str(average1) + "\n")
    f.write(str(ol) + "\n")
    f.write("\n" + str(average2) + "\n")
    f.write(str(word_sec) + "\n")
#---------------------------------------------------------------------------------------------------
while len(EHF)/2.0 > num_words:
    length = []
    i = 0
    while i < len(EHF) :
        length.append(EHF[i+1]-EHF[i])
        i = i + 2

    dl = length.index(min(length))
    EHF = np.delete(EHF, [2*dl, 2*dl+1])
#---------------------------------------------------------------------------------------------------
EHF2 = EHF
#---------------------------------------------------------------------------------------------------
if len(EHF)/2.0 < num_words:
    while len(EHF2)/2.0 < num_words:
        ind = int(ol[0])
        start = EHF2[2*ind]
        EHF2 = np.delete(EHF2, [2*ind, 2*ind+1])
        t = round(length[ind] / word_sec)
        if t < 2 :
            t = 2
        s = length[ind] / t
        for j in range(t):
            EHF2 = np.insert(EHF2, 2*ind+2*j, start)
            start = start + s
            EHF2 = np.insert(EHF2, 2*ind+2*j+1, start)

        length = []
        i = 0
        while i < len(EHF2) :
            length.append(EHF2[i+1]-EHF2[i])
            i = i + 2
        ol = list(map(length.index, heapq.nlargest(2, length)))

    """
    ind = 0
    for i in ol:
        start = EHF2[2*i+ind]
        EHF2 = np.delete(EHF2, [2*i+ind, 2*i+1+ind])
        t = round(length[i] / word_sec)
        if t < 2 :
            t = 2
        s = length[i] / t
        for j in range(t):
            EHF2 = np.insert(EHF2, 2*i+ind+2*j, start)
            start = start + s
            EHF2 = np.insert(EHF2, 2*i+ind+2*j+1, start)
        ind = ind + t*2-2
    """
    #---------------------------------------------------------------------------------------------------
    print("區間：", EHF2)
    
    s = str(len(EHF2)/2.0)
    print("片段數：", s)
    #---------------------------------------------------------------------------------------------------
    while len(EHF2)/2.0 > num_words:
        length2 = []
        i = 0
        while i < len(EHF2) :
            length2.append(EHF2[i+1]-EHF2[i])
            i = i + 2

        dl = length2.index(min(length2))
        EHF2 = np.delete(EHF2, [2*dl, 2*dl+1])
#---------------------------------------------------------------------------------------------------

f = wave.open(filename, "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

str_data = f.readframes(nframes)
f.close()
wave_data = np.fromstring(str_data, dtype = np.short)
#---------------------------------------------------------------------------------------------------
with wave.open(result + "偵測結果.wav", 'wb') as wavfile:
    wavfile.setparams((nchannels, sampwidth, framerate, 0, 'NONE', 'NONE'))
    i = 0
    while i < len(EHF2) :
        for num in wave_data[round(EHF2[i] * 44100) : round(EHF2[i+1] * 44100)] :
            wavfile.writeframes(num)
        wavfile.writeframes(np.zeros(5000))
        i = i + 2

i = 0
j = 1
while i < len(EHF2) :
    with wave.open(result + str(j) + ".wav", 'wb') as wavfile:
        wavfile.setparams((nchannels, sampwidth, framerate, 0, 'NONE', 'NONE'))
        for num in wave_data[round(EHF2[i] * 44100) : round(EHF2[i+1] * 44100)] :
            wavfile.writeframes(num)
    j = j + 1
    i = i + 2
