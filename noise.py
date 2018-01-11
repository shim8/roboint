#! usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind):
    """MNISTデータをpathからロード"""
    # 引数に指定したパスを結合（ラベルや画像のパスを作成）
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    # ファイルを読み込む：
    # 引数にファイル、モードを指定（rbは読み込みのバイナリモード）
    with open(labels_path, 'rb') as lbpath:
        # バイナリを文字列に変換：unpack関数の引数にフォーマット、8バイト分のバイナリデータを指定してマジックナンバー、アイテムの個数を読み込む
        magic, n = struct.unpack('>II', lbpath.read(8))
        # ファイルからラベルを読み込み配列を構築：fromfile関数の引数にファイル、配列のデータ形式を指定
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # 画像ピクセル情報の配列サイズを変更
        # (行数：ラベルのサイズ、列数：特徴量の個数)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images / 255.0

    return images, labels

def main():
    images_train, labels_train = load_mnist(path='mnist', kind='train')
    images_test, labels_test = load_mnist(path='mnist', kind='t10k')

    DNN(images_train, labels_train, images_test, labels_test)

def DNN(images_train, labels_train, images_test, labels_test):
    num_train = len(images_train)
    num_test = len(images_test)
    rate = []

    eta = 0.01
    num_mid1 = 100
    num_mid2 = 50
    batch = 100
    finish = 50
    noise_max = 25
    w_mid1 = np.empty((785, num_mid1))
    w_mid2 = np.empty((num_mid1+1, num_mid2))
    w_out = np.empty((num_mid2+1, 10))

    y_train = np.zeros((num_train, 10))
    for i in range(num_train):
        label = labels_train[i]
        y_train[i][label] = 1

    w_mid1 = init_w(w_mid1)
    w_mid2 = init_w(w_mid2)
    w_out = init_w(w_out)

    for epochs in range(finish):
        for i in range(num_train / batch):
            w_mid1, w_mid2, w_out = train(i, batch, images_train, y_train, w_mid1, w_mid2, w_out, sigmoid, dsigmoid, eta)

        print("{0} / {1} epochs".format(epochs+1, finish))

    for noise in range(noise_max+1):
        images_test_noise = noise_generation(images_test, noise)
        y_hat = test(images_test_noise, w_mid1, w_mid2, w_out, sigmoid)
        labels_DNN = np.argmax(y_hat, axis=1)
        match_DNN = (labels_DNN == labels_test)
        r = 100.0 * sum(match_DNN) / len(match_DNN)
        rate = np.append(rate, r)

        print("{0} / {1} %".format(noise, noise_max))

    show(rate, noise_max)

def train(i, batch, x_train, y_train, w_mid1, w_mid2, w_out, func, dfunc, eta):
    x_in = x_train[batch*i:batch*(i+1)]
    x_in, u_mid1, x_mid1 = part_calc(x_in, w_mid1, func)
    x_mid1, u_mid2, x_mid2 = part_calc(x_mid1, w_mid2, func)
    x_mid2, u_out, x_out = part_calc(x_mid2, w_out, func)
    delta_out = calc_delta_out(i, batch, x_out, y_train, u_out, dfunc)
    delta_mid2 = calc_delta_mid(w_out, delta_out, u_mid2, dfunc)
    delta_mid1 = calc_delta_mid(w_mid2, delta_mid2, u_mid1, dfunc)
    w_mid1_new = update_w(eta, w_mid1, x_in, delta_mid1)
    w_mid2_new = update_w(eta, w_mid2, x_mid1, delta_mid2)
    w_out_new = update_w(eta, w_out, x_mid2, delta_out)
    return w_mid1_new, w_mid2_new, w_out_new

def test(x_test, w_mid1, w_mid2, w_out, func):
    x_in, u_mid1, x_mid1 = part_calc(x_test, w_mid1, func)
    x_mid1, u_mid2, x_mid2 = part_calc(x_mid1, w_mid2, func)
    x_mid2, u_out, x_out = part_calc(x_mid2, w_out, func)
    return x_out

def noise_generation(x, d):
    for i in range(len(x)):
        for j in range(len(x[0])):
            P = np.random.randint(100)
            if(P < d):
                x[i][j] = np.random.rand()
    return x

def init_w(x):
    w = np.random.randn(len(x), len(x[0])) / np.sqrt(len(x))
    return w

def add_bias(x):
    return np.hstack((np.ones((len(x), 1)), x))

def part_calc(x_in, w, func):
    x_in = add_bias(x_in)
    u = np.dot(x_in, w)
    x_out = func(u)
    return x_in, u, x_out

def calc_delta_mid(w, delta, u, dfunc):
    w = np.delete(w, 0, 0)
    w_t = w.T
    return np.dot(delta, w_t) * dfunc(u)

def calc_delta_out(i, batch, x_out, y_train, u, dfunc):
    y_train = y_train[batch*i:batch*(i+1)]
    y_hat = softmax(x_out)
    return (y_hat - y_train) * dfunc(u)

def update_w(eta, w, x, delta):
    return w - eta * np.dot(x.T, delta) / len(x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-alpha * x))

def dsigmoid(x):
    return alpha * sigmoid(x) * (1.0 - sigmoid(x))

def ReLU(x):
    return np.where(x > 0, x, 0)

def dReLU(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.power(tanh(x), 2)

def softmax(x):
    exp = np.exp(x)
    sums = np.sum(exp, axis=1)
    sums = sums[:, np.newaxis]
    return exp / sums

def show(Y, end):
    X = np.array(range(end+1))
    plt.figure()
    plt.plot(X, Y)
    plt.xlabel("noise [%]")
    plt.ylabel("accuracy [%]")
    plt.xlim(0, end)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    alpha = 4.0
    main()
