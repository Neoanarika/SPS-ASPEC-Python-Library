from math import sin, pi
import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
from sklearn import linear_model


class SignalProcessing:
  """
  SignalProcessing Class

  Methods avaliable
  ------------------------
  1. fft (Public)
  2. butter_lowpass_filter (Public)
  3. __butter_lowpass (Private)

  Notes
  -----
  I was thinking of implementing other filter with different properties such as
  - chebyshev type 2 filter
  - FIR filters

  Side note:
  - chebyshev type 1 filter
  - ellipitic filter doesn't have a uniform response for the passband so they might
  introduce artiefacts.
  """

  # For those unfamillar with python classes, __<class name> defines a private class
  # similar to java private classes
  @staticmethod
  def __butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    return b, a

  @staticmethod
  def fft(signal ,f_s=10):
    """
    Fast Fourier Transform (FFT) transform

    Parameters
    ----------

    signal : np.array
      This repersents the input signal

    f_s : int
      This repersents the sampling frequency of the fft.

    """
    X = fftpack.fft(signal)
    freqs = fftpack.fftfreq(len(signal)) * f_s

    return X, freqs, f_s

  def butter_lowpass_filter(self, data, cutoff, fs, order=5):
    """
    Butterworth Lowpass Filter

    Parameters
    ----------

    data : np.array
      This repersents the input data

    cutoff : float
      This is the cutoff frequency

    fs : float
      This is the sampling frequency of the data, how often was the data collected
    """

    b, a = self.__butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)

    return y

class SpectralData(SignalProcessing):
  """
  Spectral Data Loader class

  Parameters
  ----------
  filename : str
      Filename of the text file obtained from the datalogger.

  Atrributes avaliable
  --------------------

  get_angle : np.array (public)
    The angular position the rotary sensor measured (ground truth).

  get_intensity : np.array (public)
    The light intensity the light sensor measured (ground truth).

  get_cutoff_frequency : np.array (private)
    The cutoff frequency to be used by the low pass filter.

  get_angle_at_max_peak : float (private)
    To center the graph

  Methods avaliable
  -----------------

  fft : X, freqs, f_s
    FFT results, pretty dank cos this uses python method polymorphism

  low_pass_filter:
    Performs a butterworth low pass filter

  Notes
  -----
  The reason why we use a low pass filter is because the jerks and gear slippage
  occur at short time intervals which correspond to the higher frequency region
  of the ffted data.

  """

  def __init__(self, filename, scale=60):
    self.data = pd.read_csv(filename, sep='\t',header=1)

  @property
  def get_angle(self):

    return self.data['Angular Position, Ch 1&2 ( deg )'].to_numpy()

  @property
  def get_intensity(self):

    return self.data['Light Intensity, Ch A ( % max )'].to_numpy()

  @property
  def get_angle_at_max_peak(self):

    return self.get_angle[np.argmax(self.get_intensity)]

  @property
  def get_cutoff_frequency(self):
    n_samples = 1

    X, freqs, f_s = super().fft(self.get_intensity)
    f, y = np.expand_dims(freqs[:100], axis=1), np.abs(X[:100])

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(f, y)

    c = ransac.predict([[0]])[0]
    m = ransac.estimator_.coef_[0]

    return abs(c/m)

  def fft(self):
    return super().fft(self.get_intensity)

  def low_pass_filter(self, cutoff=None, fs=10):

    if not cutoff:
      cutoff = self.get_cutoff_frequency

    return super().butter_lowpass_filter(self.get_intensity,cutoff,fs)

class AngleData(object):
  """
  AngleData class

  Parameters
  ----------
  filename : str
      Filename of the text file obtained from the datalogger.

  Atrributes avaliable
  --------------------
  get_angular_pos
  get_time
  get_gradients

  Notes
  -----
  Reads in the angle data file and then outputs the slope of an interval
  """
  def __init__(self, filename):
    self.data = pd.read_csv(filename, sep='\t',header=1)

  @property
  def get_angle(self):
    return self.data["Angular Position ( deg )"].to_numpy()

  @property
  def get_time(self):
    return self.data["Time ( s )"].to_numpy()

  @property
  def get_gradients(self):

    def gradient(pair1, pair2):
        x1, y1 = pair1
        x2, y2 = pair2
        return float(y2 - y1)/float(x2 - x1)

    return [gradient((self.get_time[i], self.get_angular_pos[i]), (self.get_time[i+1], self.get_angular_pos[i+1])) for i in range(len(self.x)-1)]


class ProcessedSpectralData(object):
    """
    ProcessedSpectralData Class
    The class does the relevant signal processing and error correction to correct for the error collected by the experiment.

    Parameters
    ----------

    spectral : Spectral Class Object
        Spectral class object to get the spectral data from.

    angle : Angle Class Object
        Angular vs time class object to get the angle data from.

    gear_ratio : int, float [Default 0]
        The gear ratio of the setup. If the gear ratio is zero, it will use angular data to estimate
        the gear ratio.

    weight : float [Default 0.5]
        The moving average "gain" (Kalman Gain). This means that if the value is higher we trust the
        data more than the model, the lower this value gets the more we based our estimates on the model.

    max_theta : int, float [Default 120]
        The angle covered by the entire setup.

    cutoff_freq : float, str [Default 0]
        This defines the cutoff frequency of the low pass filter. If the cutoff_freq is 'auto'
        then the class will use the RANSAC Algorithm to find the best cutoff_freq based on the FFT.

    d : float [Default 166e-9]
        Distance of diffraction grating

    m : float [Default 0.6168]
        Calibration line gradient

    c : float [Default 250.39]
        Calibration line y-intercept

    Methods avaliable
    --------------------
    __interval (private)
    __gradient_map (private)
    wavelength
    theta_peak_at_interval
    angle_peak_at_interval
    find_peak_wavelength_at_interval

    Notes
    -----
    Weight works somewhat like Kalaman Gain the idea is that if u believe in the local moving average more than the
    surrounding data you should account for more of it as well, hence the range of influence also expands as u decrease the weight.

    """

    def __init__(self, spectral, angle, gear_ratio=0, weight=0.5, max_theta=120, cutoff_freq=0, d=166e-9, m=0.6168, c=250.39):

        self.m = m
        self.c = c

        if not isinstance(spectral, SpectralData):
            raise Exception("Spectral object was not given for the first parameter. Please refer to doc string")

        if not isinstance(angle, AngleData):
            raise Exception("Angle object was not given for the second parameter. Please refer to doc string")

        interval = self.__interval(angle.get_angle, angle.get_time)

        assert (angle.get_angle == spectral.get_angle).all

        if cutoff_freq:
            intensity = spectral.low_pass_filter(cutoff_freq)[interval[0]:interval[1]]
        elif cutoff_freq == 'auto':
            cutoff_freq = sepctral.get_cutoff_frequency()
            intensity = spectral.low_pass_filter(cutoff_freq)[interval[0]:interval[1]]
        else:
            intensity = spectral.get_intensity[interval[0]:interval[1]]

        ang = spectral.get_angle[interval[0]:interval[1]]
        time = angle.get_time[interval[0]:interval[1]]

        dang_dtimes = self.__gradient_map(ang, time)
        dtime = time[1] - time[0]
        self.dang_dtimes = np.array(dang_dtimes)

        rescale_axis = lambda data: [i - data[0] for i in data]

        self.time = np.array(rescale_axis(time))
        self.angle = np.array(rescale_axis(ang))
        self.light_intensity = np.array(rescale_axis(intensity))
        assert len(self.light_intensity) == len(self.angle) and len(self.angle) == len(self.time)

        if not gear_ratio:
            gear_ratio = float(max_theta/(self.angle[-1] - self.angle[0]))

        # Weighted average (Kalaman Filter)
        # The core assumption of the model is this (At small time scale the change in angle is likely smooth and continous)
        # This accounts for gear slippage

        avg_dang_dtime = []
        for i,x in enumerate(dang_dtimes):
            avg_dang_dtime.append(weight*x + (1-weight)*(avg_dang_dtime[i-1] if i!=0 else 0))
            if x==0 : dang_dtimes[i] = avg_dang_dtime[-1]
        self.avg_dang_dtime = np.array(avg_dang_dtime)

        dtheta_dtimes = [0] + [gear_ratio*dang_dtime for dang_dtime in avg_dang_dtime]
        dtheta = [dtheta_dtime*dtime for dtheta_dtime in dtheta_dtimes]

        theta = [sum(dtheta[:i]) for i in range(len(dtheta))]
        theta_max = theta[np.argmax(self.light_intensity)]
        self.theta = np.array([t-theta_max for t in theta])
        self.wavelength = np.array([self.calibration(self.compute_wavelength(abs(theta/360)*2*pi, d, 1)/1e-10) for theta in self.theta])


    @staticmethod
    def __gradient_map(y, x):

        def gradient(pair1, pair2):

            x1, y1 = pair1
            x2, y2 = pair2

            #print(x1,y1,x2,y2)
            if float(y2 - y1) == 0: return 0
            return float(y2 - y1)/float(x2 - x1)


        return [gradient((x[i], y[i]), (x[i+1], y[i+1])) for i in range(len(x)-1)]

    @staticmethod
    def compute_wavelength(theta, d, order):
        return d*sin(theta)/order

    def __interval(self, x, y):
        lst = self.__gradient_map(x, y)
        for i in range(len(lst)-1):
            if lst[i] == 0 and lst[i+1] ==0 and lst[i+2] !=0:
                start = i+2
            elif lst[i] != 0 and lst[i+1] ==0 and lst[i+2] ==0:
                end = i+1
                break

        return start, end

    def xaxis_value_for_intensity_peak_within_interval(self, value, theta1, theta2):
        """
        Method to find the x axis value for a given intensity peak

        Parameters
        ----------
        value : str
            Takes in an attribute name (e.g. wavelength)

        theta1 : float
            Takes in the minumum value defining the interval

        theta2 : float
            Takes in the minumum value defining the interval
        """
        attr = self.__getattribute__(value)
        arr = attr[(attr >= theta1) & (attr <= theta2)]
        intensity = self.light_intensity[(attr >= theta1) & (attr <= theta2)]
        return arr[np.argmax(intensity)]

    def calibration(self, measured_theta):
        return self.m*measured_theta + self.c
