import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq
from scipy.fft import rfft

amplitude = 1
phi0 = 0


def get_function(f, x):
    return amplitude * np.sin(2 * np.pi * f * x + phi0)


def get_harmonic_signal(f, time):
    values = []
    for t in time:
        values.append(get_function(f, t))
    return values


def get_digital_signal(f, time):
    values = []
    for t in time:
        val = np.sign(get_function(f, t))
        if val < 0:
            val = 0
        values.append(val)
    return values


def draw_harmonic_graphics(f, time):
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Harmonic graphics")

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.plot(time, get_harmonic_signal(f[0], time))
    plt.title(f"f = {f[0]}")

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.plot(time, get_harmonic_signal(f[1], time))
    plt.title(f"f = {f[1]}")

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.plot(time, get_harmonic_signal(f[2], time))
    plt.title(f"f = {f[2]}")

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.plot(time, get_harmonic_signal(f[3], time))
    plt.title(f"f = {f[3]}")

    plt.show()


def draw_digital_graphics(f, time):
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Digital graphics")

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.plot(time, get_digital_signal(f[0], time))
    plt.title(f"f = {f[0]}")

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.plot(time, get_digital_signal(f[1], time))
    plt.title(f"f = {f[1]}")

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.plot(time, get_digital_signal(f[2], time))
    plt.title(f"f = {f[2]}")

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.plot(time, get_digital_signal(f[3], time))
    plt.title(f"f = {f[3]}")

    plt.show()


def draw_spectrum_harmonic_graphics(f):
    time = np.linspace(0, 1, num=100)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Spectrum harmonic graphics")

    plt.subplot(2, 2, 1)
    plt.grid()
    signal = get_harmonic_signal(f[0], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[0]}")

    plt.subplot(2, 2, 2)
    plt.grid()
    signal = get_harmonic_signal(f[1], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[1]}")

    plt.subplot(2, 2, 3)
    plt.grid()
    signal = get_harmonic_signal(f[2], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[2]}")

    plt.subplot(2, 2, 4)
    plt.grid()
    signal = get_harmonic_signal(f[3], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[3]}")

    plt.show()


def draw_spectrum_digital_graphics(f):
    time = np.linspace(0, 1, num=100)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Spectrum digital graphics")

    plt.subplot(2, 2, 1)
    plt.grid()
    signal = get_digital_signal(f[0], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    yf[0] = 0
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[0]}")

    plt.subplot(2, 2, 2)
    plt.grid()
    signal = get_digital_signal(f[1], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    yf[0] = 0
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[1]}")

    plt.subplot(2, 2, 3)
    plt.grid()
    signal = get_digital_signal(f[2], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    yf[0] = 0
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[2]}")

    plt.subplot(2, 2, 4)
    plt.grid()
    signal = get_digital_signal(f[3], time)
    yf = rfft(signal)
    xf = rfftfreq(len(time), 1 / len(time))
    yf[0] = 0
    plt.plot(xf, np.abs(yf))
    plt.title(f"f = {f[3]}")

    plt.show()


def main():
    f = [1, 2, 4, 8]
    time = np.linspace(0, 1, num=450)
    draw_harmonic_graphics(f, time)
    draw_digital_graphics(f, time)
    draw_spectrum_harmonic_graphics(f)
    draw_spectrum_digital_graphics(f)


if __name__ == '__main__':
    main()
