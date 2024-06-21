import numpy as np
import matplotlib.pyplot as plt


def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return noise


x = np.linspace(0, 50, 40000)
signal = np.sin(x)
noise = gen_gaussian_noise(signal, -2)
noisy_signal = signal + noise
plt.plot(x, signal, label='signal')
plt.plot(x, noisy_signal, label='noise_signal')
plt.legend()
plt.show()