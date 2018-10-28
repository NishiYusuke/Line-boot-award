#coding:utf-8
import sys,os,wave,glob,copy
import scipy
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms
from scipy import io
from scipy.io import wavfile
#import librosa
#import librosa.display
import copy
from pysptk.sptk import swipe,rapt
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    src = copy.deepcopy(x)
    try:
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate)
        if len(x)>input_length:
            return x[:input_length]
        else:
            return np.pad(x, (0, max(0, input_length - len(x))), "constant")
    except:
        return src

def wavread(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    return x, float(fs)

def hz2mel(f):
    """Hzをmelに変換"""
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをhzに変換"""
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = nfft / 2
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((int(numChannels), int(nmax)))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            i=int(i)
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i=int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
    return filterbank, fcenters

def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def mfcc(signal, nfft, fs, nceps):
    """信号のMFCCパラメータを求める
    signal: 音声信号
    nfft  : FFTのサンプル数
    nceps : MFCCの次元"""
    # プリエンファシスフィルタをかける
    p = 0.97         # プリエンファシス係数
    signal = preEmphasis(signal, p)

    # ハミング窓をかける
    hammingWindow = np.hamming(len(signal))
    signal = signal * hammingWindow

    # 振幅スペクトルを求める
    spec = np.abs(np.fft.fft(signal, nfft))[:nfft//2]
    fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:nfft//2]

    # メルフィルタバンクを作成
    numChannels = nceps#20  # メルフィルタバンクのチャネル数
    df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
    filterbank, fcenters = melFilterBank(fs, nfft, numChannels)

    # 定義通りに書いた場合
    # 振幅スペクトルに対してフィルタバンクの各フィルタをかけ、振幅の和の対数をとる
    mspec = np.log10(np.dot(spec, filterbank.T))

    # 離散コサイン変換
    ceps = scipy.fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)

    # 低次成分からnceps個の係数を返す
    return ceps[:nceps]

def cal_ceps(wavdata,fs,cepCoef=40):# cepCoef:ケプストラム次数
    # ハニング窓をかける
    n = len(wavdata)
    hanningWindow = np.hanning(n)# n:切り出したサンプル数
    wav = wavdata * hanningWindow
    
    # 切り出した音声のスペクトルを求める
    # 離散フーリエ変換
    dft = np.fft.fft(wav, n)
    # 振幅スペクトル
    Adft = np.abs(dft)
    # 周波数スケール
    fscale = np.fft.fftfreq(n, d = 1.0 / fs)
    # 対数振幅スペクトル
    AdftLog = 20 * np.log10(Adft)
    
    # ケプストラム分析
    # 対数スペクトルを逆フーリエ変換して細かく振動する音源の周波数と
    # ゆるやかに振動する声道の周波数を切り分ける
    cps = np.real(np.fft.ifft(AdftLog))
    
    # ローパスリフタ
    # ケプストラムの高次成分を0にして微細構造を除去し、
    # 緩やかなスペクトル包絡のみ抽出
    cpsLif = np.array(cps)   # arrayをコピー
    # 高周波成分を除く（左右対称なので注意）
    cpsLif[cepCoef:len(cpsLif) - cepCoef + 1] = 0

    # ケプストラム領域をフーリエ変換してスペクトル領域に戻す
    # リフタリング後の対数スペクトル
    dftSpc = np.fft.fft(cpsLif, n)
    fscale,result = fscale[0:n//2], dftSpc[0:n//2]
    return cpsLif[:cepCoef]

def process_f0(data,NFFT,fs=8000,_min=40,_max=800):
    """
    基本周波数処理
    ------------
    data:音声データ
    fs:サンプリング周波数
    hopsize:抽出間隔
    min:最小周波数
    max:最大周波数
    """
    f0 = swipe(data,fs=fs,hopsize=NFFT,min=_min,max=_max,otype="f0")
    return f0

def data_aug(x,NFFT=1024):
    whx = add_white_noise(x, rate=0.002)
    sfx = shift_sound(x, rate=2)
    stx = stretch_sound(x, rate=1.1)
    choice = np.array([1,2,3,4])# 1:wh+sf 2:wh+st 3:sf+st 4:ALL
    end = len(x)//NFFT
    idx = np.random.randint(1,4,end)# 1〜4 の整数を len(x)//128 個生成
    random_choice = choice[idx]
    n1,n2,n3,n4 = 0,0,0,0
    n1 = np.sum(random_choice == 1)
    n2 += n1 + np.sum(random_choice == 2)
    n3 += n2 + np.sum(random_choice == 3)
    n4 += n3 + np.sum(random_choice == 4)
    x1,x2,x3,x4 = x[:n1*NFFT],x[n1*NFFT:n2*NFFT],x[n2*NFFT:n3*NFFT],x[n3*NFFT:]
    x1,x2,x4 = add_white_noise(x1, rate=0.002),add_white_noise(x2, rate=0.002),add_white_noise(x4, rate=0.002)
    x1,x3,x4 = shift_sound(x1, rate=2),shift_sound(x3, rate=2),shift_sound(x4, rate=2)
    x2,x3,x4 = stretch_sound(x2, rate=1.1),stretch_sound(x3, rate=1.1),stretch_sound(x4, rate=1.1)
    return np.concatenate([x[:end*NFFT],whx[:end*NFFT],sfx[:end*NFFT],stx[:end*NFFT],x1[:end*NFFT],x2[:end*NFFT],x3[:end*NFFT],x4[:end*NFFT]],axis=0)

def wav2train(LABEL_LIST,nceps,NFFT=1024):
    SHIFT = NSHIFT
    for LABEL in LABEL_LIST:
        for wavfile in glob.glob(os.path.join("Voice","Train",LABEL,"*.wav")):
            #print(wavfile)
            # 音声をロード
            wav, fs = wavread(wavfile)
            #if wav != []:
            #    wav = data_aug(wav,NFFT)
            for s in range(0,len(wav),SHIFT):
                if (s+NFFT) < len(wav):
                    f0 = process_f0(wav[s:s+NFFT],NFFT,fs=8000,_min=40,_max=800)
                    ceps = mfcc(wav[s:s+NFFT], nfft=NFFT, fs=fs, nceps=nceps)
                    #ceps = cal_ceps(wav[s:s+NFFT],fs=fs,cepCoef=nceps)
                    isNaN = False
                    for cc in ceps:
                        if not cc == cc:#NaN
                            isNaN = True
                    if not isNaN:
                        if not "Train" in locals():
                            Train = np.array(ceps)
                            Train = Train[None,:]
                        else:
                            _Train = np.array(ceps)
                            _Train = _Train[None,:]
                            Train = np.concatenate([Train,_Train],axis=0)
                    else:
                        pass
        print(Train.shape)
        np.save(LABEL+".npy",Train)
    return Train

def wav2test(nceps,NFFT=1024):
    SHIFT = NSHIFT
    wav, fs = wavread(os.path.join("Train","test30.wav"))
    for s in range(0,len(wav),NFFT):
        if (s+NFFT) < len(wav):
            f0 = process_f0(wav[s:s+NFFT],NFFT,fs=8000,_min=40,_max=800)
            ceps = mfcc(wav[s:s+NFFT], nfft=NFFT, fs=fs, nceps=nceps)
            #ceps = cal_ceps(wav[s:s+NFFT],fs=fs,cepCoef=nceps)
            isNaN = False
            for cc in ceps:
                if not cc == cc:#NaN
                    isNaN = True
            if not isNaN:
                if not "Test" in locals():
                    Test = np.array(ceps)
                    Test = Test[None,:]
                else:
                    _Test = np.array(ceps)
                    _Test = _Test[None,:]
                    Test = np.concatenate([Test,_Test],axis=0)
            else:
                pass
    print(Test.shape)
    np.save("Test.npy",Test)
    return Test

def wav2demo(nceps,NFFT=1024):
    SHIFT = NSHIFT
    for wavfile in glob.glob(os.path.join("Test","F","*.wav")):
        #print(wavfile)
        # 音声をロード
        wav, fs = wavread(wavfile)
        for s in range(0,len(wav),NFFT):
            if (s+NFFT) < len(wav):
                f0 = process_f0(wav[s:s+NFFT],NFFT,fs=8000,_min=10,_max=1600)
                #print(f0)
                ceps = mfcc(wav[s:s+NFFT], nfft=NFFT, fs=fs, nceps=nceps)
                #ceps = cal_ceps(wav[s:s+NFFT],fs=fs,cepCoef=nceps)
                isNaN = False
                for cc in ceps:
                    if not cc == cc:#NaN
                        isNaN = True
                if not isNaN:
                    if not "Demo" in locals():
                        Demo = np.array(ceps)
                        Demo = Demo[None,:]
                    else:
                        _Demo = np.array(ceps)
                        _Demo = _Demo[None,:]
                        Demo = np.concatenate([Demo,_Demo],axis=0)
                else:
                    pass
    np.save("DemoF.npy",Demo)
    del Demo
    for wavfile in glob.glob(os.path.join("Test","S","*.wav")):
        #print(wavfile)
        # 音声をロード
        wav, fs = wavread(wavfile)
        #if wav != []:
        #    wav = data_aug(wav,NFFT)
        for s in range(0,len(wav),NFFT):
            if (s+NFFT) < len(wav):
                f0 = process_f0(wav[s:s+NFFT],NFFT,fs=8000,_min=10,_max=1600)
                #print(f0)
                ceps = mfcc(wav[s:s+NFFT], nfft=NFFT, fs=fs, nceps=nceps)
                #ceps = cal_ceps(wav[s:s+NFFT],fs=fs,cepCoef=nceps)
                #ceps = cal_librosa_mfcc(wav[s:s+NFFT], n_fft=NFFT, hop_length=SHIFT,n_mels=nceps)
                isNaN = False
                for cc in ceps:
                    if not cc == cc:#NaN
                        isNaN = True
                if not isNaN:
                    #ceps = cal_ceps(wav[s:s+NFFT],fs=fs,cepCoef=nceps)
                    if not "Demo" in locals():
                        Demo = np.array(ceps)
                        #Demo = np.concatenate([np.array(ceps),np.array(f0)])
                        Demo = Demo[None,:]
                    else:
                        _Demo = np.array(ceps)
                        #_Demo = np.concatenate([np.array(ceps),np.array(f0)])
                        _Demo = _Demo[None,:]
                        #print(_Train.shape)
                        Demo = np.concatenate([Demo,_Demo],axis=0)
                else:
                    pass
                    #print(s)
    np.save("DemoS.npy",Demo)

def read_ceps(LABEL):
    LABEL_LIST = [0,1]#["A","B"]#np.identity(len(LABEL))
    count = 0
    for label in LABEL_LIST:
        ceps = np.load(LABEL[count]+".npy")
        ceps,_test = ceps[:len(ceps)*9//10],ceps[len(ceps)*9//10:]
        if not "test_x" in locals():
            test_x = np.array(_test)
            test_y = np.ones([len(test_x)])*label#label
        else:
            _test_x = np.array(_test)
            _test_y = np.ones([len(_test_x)])*label#label
            test_x = np.concatenate([test_x,_test_x],axis=0)
            test_y = np.concatenate([test_y,_test_y],axis=0)
        if not "train_x" in locals():
            train_x = np.array(ceps)
            train_y = np.ones([len(train_x)])*label#label
        else:
            _train_x = np.array(ceps)
            _train_y = np.ones([len(_train_x)])*label#label
            train_x = np.concatenate([train_x,_train_x],axis=0)
            train_y = np.concatenate([train_y,_train_y],axis=0)
        count += 1
    # Demo
    #ceps = np.load("Demo.npy")
    #demof_x = np.array(ceps)
    print(train_x.shape,train_y.shape)
    print(test_x.shape,test_y.shape)
    #print(demo_x.shape)
    return train_x,train_y,test_x,test_y#,demo_x

def normalisation(cm):
    new_cm = []
    for line in cm:
        sum_val = sum(line)
        new_array = [float(num)/float(sum_val) for num in line]
        #new_array = [float(num) for num in line]
        new_cm.append(new_array)
    return new_cm

def plot_confusion_matrix(cm,name_list,title):
    plt.clf()
    plt.matshow(normalisation(cm),fignum=False,cmap='Blues',vmin=0,vmax=1.0)
    ax = plt.axes()
    ax.set_xticks(range(len(name_list)))
    ax.set_xticklabels(name_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(name_list)))
    ax.set_yticklabels(name_list)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel('Predict class')
    plt.ylabel('True class')
    plt.grid(False)
    plt.show()

def Make_Train_Data():
    #wav2demo(nceps,NFFT=NFFT)
    #Test = wav2test(nceps,NFFT=NFFT)
    #print(np.array(Test).shape)
    Train = wav2train(Label,nceps=nceps,NFFT=NFFT)
    print(np.array(Train).shape)

def cba(inputs, filters, kernel_size, strides):
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

rad = [1,2,4,8]
nceps = 13*8//rad[1]
NFFT = 44100//rad[1]#rad[2]
NSHIFT = int(NFFT//rad[1]*2)

Label = ["A","B"]#,"C","D"]
#Make_Train_Data()

# define init train
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint

# redefine target data into one hot vector
X_train, Y_train, X_test, Y_test = read_ceps(Label)
classes = 2
y_train = keras.utils.to_categorical(Y_train, classes)
y_test = keras.utils.to_categorical(Y_test, classes)


'''
# define CNN
X_train = X_train[:,:,None]

inputs = Input(shape=(X_train.shape[1:]))
print(X_train.shape[1:])

x_1 = cba(inputs, filters=32, kernel_size=8, strides=2)
x_1 = cba(x_1, filters=64, kernel_size=4, strides=2)

x_2 = cba(inputs, filters=32, kernel_size=16, strides=2)
x_2 = cba(x_2, filters=64, kernel_size=16, strides=2)

x_3 = cba(inputs, filters=32, kernel_size=32, strides=2)
x_3 = cba(x_3, filters=64, kernel_size=32, strides=2)

x_4 = cba(inputs, filters=32, kernel_size=64, strides=2)
x_4 = cba(x_4, filters=64, kernel_size=64, strides=2)

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=4, strides=2)
print(x)
x = GlobalAveragePooling1D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)
'''

# define Full
num = X_train.shape[1:]
inputs = Input(shape=num)
num = num[0]
num = num//2
x_1 = Dense(num)(inputs)
num = num//2
x_2 = Dense(num)(x_1)
num = num//2
x_3 = Dense(num)(x_2)
x = Dense(classes)(x_3)
x = Activation("softmax")(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
            batch_size=100,
            epochs=12,
            verbose=1)

score = model.evaluate(X_test, y_test)
print(score[0])
print(score[1])


########################
# LinearSVC
########################
from sklearn.svm import LinearSVC

# トレーニング・テストデータ分割
#X_train, Y_train, X_test, Y_test,X_demo,Y_demo = read_ceps(Label)
X_train, Y_train, X_test, Y_test = read_ceps(Label)

# SVM
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


clf=svm.LinearSVC(loss='hinge', C=1.0,class_weight='balanced', random_state=0)#loss='squared_hinge' #loss="hinge", loss="log"
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))
cm = confusion_matrix(Y_test,Y_pred)
plot_confusion_matrix(cm,name_list=["Others","Swindler"],title="Result")


'''
Voice = np.load("Test.npy")
print(Voice.shape)
X_train, Y_train, X_test, Y_test,X_demo,Y_demo = read_ceps()
idx = list(range(len(X_train)))
random.shuffle(idx)
N = 10000#len(X_train)#30000
_x,_y = X_train[idx][:N],Y_train[idx][:N]
#clf = svm.SVC()
clf = svm.SVC(kernel='rbf', C=10, gamma=0.5)
#clf.fit(X_train[2:], Y_train[2:])
clf.fit(_x, _y)
Y_pred = clf.predict(X_test)

# デモ動画用
Voice_pred = clf.predict(Voice)
np.save("Voice_pred.npy",Voice_pred)

# デモテスト
Demo_pred = clf.predict(X_demo)
np.save("Demo_pred.npy",Demo_pred)
print(Demo_pred.shape)
#np.set_printoptions(threshold=3)
#print(Y_test)
#print(Y_pred)
'''

