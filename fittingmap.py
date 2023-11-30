"""
The part of micromap package (https://github.com/romus33/micromap) for curve fitting based on lmfit  for fitting of different types of curves. 

Release: 1.0.0
"""
__author__ = "Roman Shendrik"
__copyright__ = "Copyright (C) 2023R"
__license__ = "GNU GPL 3.0"
__version__ = "1.0.0"

import time as tm
from ctypes import *
from multiprocessing import Pool, Value, Manager
import lmfit
import numpy as np
# import scipy.sparse as sparse
from lmfit.model import Model
# from scipy.sparse.linalg import spsolve
#from termcolor import colored
import progressbar as pb
import os, sys
import platform
import pathlib
pth_=pathlib.Path(__file__).parent.resolve()
if platform.uname()[0] == "Windows":
    path = os.path.abspath(pth_)
    os.add_dll_directory(pth_)
else:
    path = os.path.abspath(pth_)
    os.environ["LD_LIBRARY_PATH"] = "pth_"

if path not in sys.path:
    sys.path.append(path)

os.system('color')
# selection of OS platform
if platform.uname()[0] == "Windows":
    name_dll = "convolution.dll"
else:
    name_dll = "convolution.so"
# Load library contains ALS and TSL curve fitting algorithms
myclib = cdll.LoadLibrary(name_dll)

# Multithreading
num_proc = 4


class FittingMap(object):
    """
    Class for maps using fitting of the peaks and determine thin of samples
    Attributes:
        map_baseline(dict): The dictionary contains hyperspectral map of the best fitted spectra.
        Their structure is the same to __map_spectra__ in MapEdit class. The name of each dictionary elements is [str(mx) + '_' + str(my)]
        The structure of the dictionary is [array_besfit_y, mx, my, nv, nv, xmin, xmax, step], where
            array_bestfit_y: array of the best fit of y values in the point (mx,my) of the hyperspectral map
            mx: x-coordinate of the point in the hyperspectral map
            my: y-coordinate of the point in the hyperspectral map
            nv: the length of y-array
            x1: minimal wavenumber in the spectrum
            x2: maximum wavenumber in the spectrum
            step: distance between adjacent x values (step of wavenumber measurement)

        map_bline(dict): The dictionary contains parameters of fitted baseline (als algorithm) in each point (mx,my) of the hyperspectral map.
        The name of the element of this dictionary is [str(mx) + '_' + str(my)].
        The structure of the dictionary map_bline[name] is [x, y, bg_lam, bg_p] is:
            x: array of x-values of the best fitted baseline
            y: array of y-values of the best fitted baseline
            bg_lam: Fitted 2nd derivative constraint in als algorithm
            bg_p: Fitted Weighting of positive residuals in algorithm

        components(dict): The dictionary contains evaluated components out.eval_components

        lam(double): 2nd derivative constraint in case of manual als baseline fitting

        p(double): Weighting of positive residuals in case of manual als baseline fitting

        niter(int): Number of iterations of manual als baseline fitting

        __pBar__(object): init progress bar

        leng(int): the length of map_baseline dictionary

        param(dict): Dictionary contains parameters of fitting

            'range': range of fitting on absciss axis. This is array contains xmin and xmax values [xmin,xmax]. By default the input spectrum is fitted in all range.
            'method': the array of names of the fitting method. The length of array is the number of fitting curve that will be used. There are Gaussian/Voigt/Lorenzian,PseudoVoigt for peak detection. For thermally stimullated processes the TSL glow curve of 1 order TSL_1, second order TSL_2 and thermally stimulated decay curve  TD_1 and TD_2 is available.
            'amplitude': the array of values of initial amplitudes of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
            'center': the array of values of initial centers (x-coordinate) of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
            'width': the array of values of initial widths of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
            'baseline'[lam,p,niter]: the array contains manual parameters for baseline als
                baseline[0](double): 2nd derivative constraint in case of manual als baseline fitting
                baseline[1](double): Fitted Weighting of positive residuals in algorithm
                baseline[2](int): Number of iterations of manual als baseline fitting
            'baseline_auto'[lam,p,niter]: if this parameters is given, the auto fit of baseline occurs. There are array with the following parameters:
                baseline_auto[0](double): 2nd derivative constraint in case of manual als baseline fitting
                baseline_auto[1](double): Fitted Weighting of positive residuals in algorithm
                baseline_auto[2](int): Number of iterations of manual als baseline fitting
            'kws' (dict): This is tolerance parameter of fitting. The default values are {'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12}
            'max_nfev' (int): number of least square fit iterations.
            'energy'[]: Activation energy for each peak in TSL_ and TD_ fits
            'factor'[]: Frequency factor in TSL_ and TD_ fits

        limit(dict): Dictionary contains limits of parameter fitting
            'amplitude'[][]: the array contains pairs minimum and maximum values of amplitude.
                Example of limit amplitude calculation:
                    limit['amplitude'][0]*param['amplitude'] - min value of amplitude
                    limit['amplitude'][1]*param['amplitude'] - max value of amplitude
            'center'[][]: the array contains pairs minimum and maximum values of peak centers.
                Example of limit of center calculation:
                    limit['center'][0]-param['center'] - min value of peak center
                    limit['center'][1]+param['center] - max value of peak center
            'width'[][]: the array contains pairs minimum and maximum values of peak width.
                Example of limit peak width calculation:
                    limit['width'][0]*param['width'] - min value of peak width
                    limit['width'][1]*param['width] - max value of peak width
            'baseline_auto'[[lam_min, lam_max], [p_min, p_max]]:
                The limits of 2nd derivative constraint (lam) are lam_min<lam<lam_max
                The limits of Weighting of positive residuals (p) are p_min<p<p_max
            'energy'[E_min, E_max][]: Limits of activation energy E_min<E<E_max in TSL_ and TD_ fits
            'factor'[f_min, f_max][]: Limits of frequency factor in TSL_ and TD_ fits

    """

    def __init__(self):
        """
        Init method of the class FittingMap.
        """
        # self.__map_spectra__ = {}
        # manager = Manager()
        self.map_baseline = {}
        self.map_bline = {}
        self.components = {}
        self.lam = 1e7
        self.p = 0.01
        self.niter = 10
        # self.counter=None
        # self.it=0
        self.__pBar__ = pb.ProgressBar(tm.time())
        self.leng = 0
        self.param = {}
        self.limit = {}

    @staticmethod
    def __baseline_als__(t: np.ndarray, lam: np.double, p: np.double):
        """Baseline reconstruct using ALS algorithm using dll
            https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
            Asymmetric Least Squares Smoothing
            Args:
                t: input y array for baseline construction
                niter: Maximum number of iterations
                lam: 2nd derivative constraint
                p: Weighting of positive residuals
            Returns:
                outputarray: array with constructed baseline with length of input t array

        
        """

        py_ALS = myclib.ALS
        py_ALS.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_double, c_double, c_int]
        py_ALS.restype = None
        inputarray = (c_double * len(t))(*t)
        outputarray = (c_double * len(t))()
        N = c_int(len(t))
        lam_dll = c_double(lam)
        p_dll = c_double(p)
        # n_iter = c_int(niter)
        py_ALS(inputarray, outputarray, N, lam_dll, p_dll, 10)

        return np.array(outputarray)

    @staticmethod
    def __TSL1__(x: np.ndarray, factor: np.double, energy: np.double, amplitude: np.double):
        """TSL of first order fit

        Args:
            x (np.ndarray): array of temperature values
            factor (np.double): frequency factor (1e8-1e14 typically)
            energy (np.double): activation energy
            amplitude (np.double): amplitude of a glow peak
        Returns:
            output*amplitude (np.double): array of y-values of calculated TSL

        """

        py_TSL = myclib.TSLCalc
        # int TSLCalc(int iSize, double Tmin, double Tmax, double *vCalc, double iFactor, double iEnergy, int kOrder)
        py_TSL.argtypes = [c_int, c_double, c_double, POINTER(c_double), c_double, c_double, c_int]
        py_TSL.restype = None
        Tmin = c_double(x.min())
        Tmax = c_double(x.max())
        Npoints = c_int(len(x))
        outputarray = (c_double * len(x))()
        factor_dll = c_double(factor)
        energy_dll = c_double(energy)
        py_TSL(Npoints, Tmin, Tmax, outputarray, factor_dll, energy_dll, 1)
        output = np.array(outputarray)
        output = output / output.max()
        return amplitude * output

    @staticmethod
    def __TSL2__(x: np.ndarray, factor: np.double, energy: np.double, amplitude: np.double):
        """TSL of secondorder fit

        Args:
            x (np.ndarray): array of temperature values
            factor (np.double): frequency factor (1e8-1e14 typically)
            energy (np.double): activation energy
            amplitude (np.double): amplitude of a glow peak
        Returns:
            output*amplitude (np.double): array of y-values of calculated TSL

        """

        py_TSL = myclib.TSLCalc
        # int TSLCalc(int iSize, double Tmin, double Tmax, double *vCalc, double iFactor, double iEnergy, int kOrder)
        py_TSL.argtypes = [c_int, c_double, c_double, POINTER(c_double), c_double, c_double, c_int]
        py_TSL.restype = None
        Tmin = c_double(x.min())
        Tmax = c_double(x.max())

        Npoints = c_int(len(x))
        outputarray = (c_double * len(x))()
        factor_dll = c_double(factor)
        energy_dll = c_double(energy)
        py_TSL(Npoints, Tmin, Tmax, outputarray, factor_dll, energy_dll, 2)
        output = np.array(outputarray)
        output = output / output.max()
        return amplitude * output

    @staticmethod
    def __TD1__(x: np.ndarray, factor: np.double, energy: np.double, amplitude: np.double):
        """Temperature decay of first order

        Args:
            x (np.ndarray): array of temperature values
            factor (np.double): frequency factor (1e8-1e14 typically)
            energy (np.double): activation energy
            amplitude (np.double): amplitude of a glow peak
        Returns:
            output*amplitude (np.double): array of y-values of calculated TSL

        """

        py_TSL = myclib.TDCalc
        # int TSLCalc(int iSize, double Tmin, double Tmax, double *vCalc, double iFactor, double iEnergy, int kOrder)
        py_TSL.argtypes = [c_int, c_double, c_double, POINTER(c_double), c_double, c_double, c_int]
        py_TSL.restype = None
        Tmin = c_double(x.min())
        Tmax = c_double(x.max())
        Npoints = c_int(len(x))
        outputarray = (c_double * len(x))()
        factor_dll = c_double(factor)
        energy_dll = c_double(energy)
        py_TSL(Npoints, Tmin, Tmax, outputarray, factor_dll, energy_dll, 1)
        output = np.array(outputarray)
        output = output / output.max()
        return amplitude * output

    @staticmethod
    def __TD2__(x: np.ndarray, factor: np.double, energy: np.double, amplitude: np.double):
        """Temperature decay of second order

        Args:
            x (np.ndarray): array of temperature values
            factor (np.double): frequency factor (1e8-1e14 typically)
            energy (np.double): activation energy
            amplitude (np.double): amplitude of a glow peak
        Returns:
            output*amplitude (np.double): array of y-values of calculated TSL

        """

        py_TSL = myclib.TDCalc
        # int TSLCalc(int iSize, double Tmin, double Tmax, double *vCalc, double iFactor, double iEnergy, int kOrder)
        py_TSL.argtypes = [c_int, c_double, c_double, POINTER(c_double), c_double, c_double, c_int]
        py_TSL.restype = None
        Tmin = c_double(x.min())
        Tmax = c_double(x.max())
        Npoints = c_int(len(x))
        outputarray = (c_double * len(x))()
        factor_dll = c_double(factor)
        energy_dll = c_double(energy)
        py_TSL(Npoints, Tmin, Tmax, outputarray, factor_dll, energy_dll, 1)
        output = np.array(outputarray)
        output = output / output.max()
        return amplitude * output

    @staticmethod
    def __get_numarray(item, wavenumber):
        """
            Private method to get corresponding array element number of wavenumber
        """
        return int((wavenumber - item[5]) / item[7])

    def __make_model__(self, num, params, limits):
        """
            Private method that generate model for least square fitting
            Args:
                num(int): number of fitting curve in case of multipeaks or multifunction fitting
                params{}(dict): Dictionary contains parameters of fitting
                    'range': range of fitting on absciss axis. This is array contains xmin and xmax values [xmin,xmax]. By default the input spectrum is fitted in all range.
                    'method': the array of names of the fitting method. The length of array is the number of fitting curve that will be used. There are Gaussian/Voigt/Lorenzian,PseudoVoigt for peak detection. For thermally stimullated processes the TSL glow curve of 1 order TSL_1, second order TSL_2 and thermally stimulated decay curve  TD_1 and TD_2 is available.
                    'amplitude': the array of values of initial amplitudes of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'center': the array of values of initial centers (x-coordinate) of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'width': the array of values of initial widths of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'baseline'[lam,p,niter]: the array contains manual parameters for baseline als
                            baseline[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline[2](int): Number of iterations of manual als baseline fitting
                    'baseline_auto'[lam,p,niter]: if this parameters is given, the auto fit of baseline occurs. There are array with the following parameters:
                            baseline_auto[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline_auto[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline_auto[2](int): Number of iterations of manual als baseline fitting
                    'kws' (dict): This is tolerance parameter of fitting. The default values are {'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12}
                    'max_nfev' (int): number of least square fit iterations.
                    'energy'[]: Activation energy for each peak in TSL_ and TD_ fits
                    'factor'[]: Frequency factor in TSL_ and TD_ fits

                limits(dict): Dictionary contains limits of parameter fitting {'amplitude':[],'center':[],'width':[],'baseline_auto':[]}
                    'amplitude'[][]: the array contains pairs minimum and maximum values of amplitude.
                        Example of limit amplitude calculation:
                            limit['amplitude'][0]*param['amplitude'] - min value of amplitude
                            limit['amplitude'][1]*param['amplitude'] - max value of amplitude
                    'center'[][]: the array contains pairs minimum and maximum values of peak centers.
                        Example of limit of center calculation:
                            limit['center'][0]-param['center'] - min value of peak center
                            limit['center'][1]+param['center] - max value of peak center
                    'width'[][]: the array contains pairs minimum and maximum values of peak width.
                        Example of limit peak width calculation:
                            limit['width'][0]*param['width'] - min value of peak width
                            limit['width'][1]*param['width] - max value of peak width
                    'baseline_auto'[[lam_min, lam_max], [p_min, p_max]]:
                        The limits of 2nd derivative constraint (lam) are lam_min<lam<lam_max
                        The limits of Weighting of positive residuals (p) are p_min<p<p_max
                    'energy'[E_min, E_max][]: Limits of activation energy E_min<E<E_max in TSL_ and TD_ fits
                    'factor'[f_min, f_max][]: Limits of frequency factor in TSL_ and TD_ fits
        """
        pref = 'f' + repr(num) + '_'
        method = params['method'][num]
        if (method == 'TSL1') or (method == 'TSL2') or (method == 'TD1') or (method == 'TD2'):
            if method == 'TSL1':
                model = Model(self.__TSL1__, prefix=pref)
            if method == 'TSL2':
                model = Model(self.__TSL2__, prefix=pref)
            if method == 'TD1':
                model = Model(self.__TD1__, prefix=pref)
            if method == 'TD2':
                model = Model(self.__TD2__, prefix=pref)
            if 'energy' not in limits:
                limits['energy'][num] = [0, 2]
            if 'factor' not in limits:
                limits['factor'][num] = [1e7, 1e14]
            if 'amplitude' not in limits:
                limits['amplitude'][num] = [0, 1000]

            model.set_param_hint(pref + 'amplitude', value=params['amplitude'][num],
                                 min=limits['amplitude'][num][0] * params['amplitude'][num],
                                 max=limits['amplitude'][num][1] * params['amplitude'][num])
            model.set_param_hint(pref + 'energy', value=params['energy'][num],
                                 min=limits['energy'][num][0],
                                 max=limits['energy'][num][1])
            model.set_param_hint(pref + 'factor', value=params['factor'][num],
                                 min=limits['factor'][num][0],
                                 max=limits['factor'][num][1])
            return model
        else:
            bar = getattr(lmfit.models, method + 'Model')
            model = bar(prefix=pref)
            if 'amplitude' not in limits:
                limits['amplitude'][num] = [0, 1000]
            if 'width' not in limits:
                limits['amplitude'][num] = [0.4, 4]
            if 'center' not in limits:
                limits['amplitude'][num] = [5, 5]
            model.set_param_hint(pref + 'amplitude', value=params['amplitude'][num],
                                 min=limits['amplitude'][num][0] * params['amplitude'][num],
                                 max=limits['amplitude'][num][1] * params['amplitude'][num])
            model.set_param_hint(pref + 'center', value=params['center'][num],
                                 min=params['center'][num] - limits['center'][num][0],
                                 max=params['center'][num] + limits['center'][num][1])
            model.set_param_hint(pref + 'sigma', value=params['width'][num],
                                 min=params['width'][num] * limits['width'][num][0],
                                 max=params['width'][num] * limits['width'][num][1])
            return model

    def map_intensity(self, item):
        """
        The method makes peak fitting in each point of hyperspectral map.
        Args:
            item(dict): the item is a hyperspectral dictionary element having the follow structure:
                item[0](floats): array-like y-values of the spectrum in a point of hyperspectral map
                item[1](int): mx coordinate of point in hyperspectral map
                item[2](int): my coordinate of point in hyperspectral map
                item[3](int): length of item[0]
                item[4](int): length of item[0]
                item[5](float): minimum value on abscissa axis of the spectra.
                item[6](float): maximum value on abscissa axis of the spectra.
                item[7](float): step between neighbouring abscissa values (step of the spectra)
        Returns:
            There two cases.
            If the peak fit takes place the method returns dictionary. The one contains arrays with fitting parameters. The length of the each array is equal to number of peaks in fitting procedure. The dictionary structure is:
                {'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'sigma': S, 'r-square': Rsq}
                'amplitude'(floats): array-like values of amplitudes of deconvoluted peaks
                'FWHM'(floats): array-like values of peaks FWHM
                'center'(floats): array-like values of x-coordinates of the peaks
                'height'(floats): array-like values of y-coordinates of the peaks
                'sigma'(floats): array-like values of peak sigmas
                'r-square'(float): the R-square value of fitting
            If the als fit is selected. The method returns:
                {'amplitude': 0, 'FWHM': 0, 'center': 0, 'height': 0, 'r-square': Rsq, 'sigma': 0}
        """
        params = self.param
        limits = self.limit
        z_={}
        if 'range' not in params:
            params['range'] = [item[5], item[6]]

        xmin = params['range'][0]
        xmax = params['range'][1]

        y = item[0][self.__get_numarray(item, xmin):self.__get_numarray(item, xmax)]
        x = np.linspace(xmin, xmax, len(y))

        if 'baseline' not in params:
            baseline_flag = False
        if baseline_flag:
            self.lam = params['baseline'][0]
            self.p = params['baseline'][1]
        # self.lam = params['baseline'][0]
        # self.p = params['baseline'][1]
        if params['method'][0] != 'als':
            if 'center' not in params:
                return print('No centers')
            center = params['center']
            number_of_peaks = len(center)
            if 'amplitude' not in params:
                for n in range(number_of_peaks):
                    params['amplitude'][n] = 1000
            if 'width' not in params:
                for n in range(number_of_peaks):
                    params['width'][n] = 2
            if 'method' not in params:
                for n in range(number_of_peaks):
                    params['method'][n] = 'Gaussian'

        if 'baseline_auto' not in params:

            if baseline_flag:
                b_line = self.__baseline_als__(y, self.lam, self.p)
                y = np.subtract(y, b_line)

        if 'kws' not in params:
            params['kws'] = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8}
        fit_kws = params['kws']
        mod = None
        global counter
        name = str(item[1]) + '_' + str(item[2])
        if params['method'][0] != 'als':

            for i in range(number_of_peaks):
                this_mod = self.__make_model__(i, params, limits)
                if mod is None:
                    mod = this_mod
                else:
                    mod = mod + this_mod
        if 'baseline_auto' in params:
            if 'baseline_auto' not in limits:
                limits['baseline_auto'] = [[1e4, 1e12], [0.0001, 0.1]]
            bl = Model(self.__baseline_als__, prefix='bg_')
            bl.set_param_hint('lam', value=params['baseline_auto'][0], min=limits['baseline_auto'][0][0],
                              max=limits['baseline_auto'][0][1])
            bl.set_param_hint('p', value=params['baseline_auto'][1], min=limits['baseline_auto'][1][0],
                              max=limits['baseline_auto'][1][1])
            if params['method'][0] != 'als':
                mod = mod + bl
            else:
                mod = bl

            out = mod.fit(y, x=x, t=y, method='least_squares', fit_kws=fit_kws)
            comps = out.eval_components(x=x)
            self.map_baseline[name] = [out.best_fit, item[1], item[2], len(y), len(y), xmin, xmax, item[7], comps]
            self.map_bline[name] = [x, comps['bg_'], out.params['bg_lam'], out.params['bg_p']]

        else:
            out = mod.fit(y, x=x, method='least_squares', fit_kws=fit_kws)
            # with counter.get_lock():
            self.components = out.eval_components(x=x)
            self.map_baseline[name] = [out.best_fit, item[1], item[2], len(y), len(y), xmin, xmax, item[7], self.components]
            if baseline_flag:
                self.map_bline[name] = [x, b_line, self.lam, self.p]
            else:
                self.map_bline[name] = [x, np.zeros(len(x)), 0, 0]
        Rsq = 1 - out.redchi / np.var(y, ddof=0)

        if params['method'][0] != 'als':
            A = np.array([out.params['f' + repr(num) + '_amplitude'] for num in range(number_of_peaks)])
            FWHM = np.array([out.params['f' + repr(num) + '_fwhm'] for num in range(number_of_peaks)])
            C = np.array([out.params['f' + repr(num) + '_center'] for num in range(number_of_peaks)])
            H = np.array([out.params['f' + repr(num) + '_height'] for num in range(number_of_peaks)])
            S = np.array([out.params['f' + repr(num) + '_sigma'] for num in range(number_of_peaks)])

        with counter.get_lock():
            counter.value += 1
            # time_elapsed=' '+str(int(tm.time()-time0))+' sec'
            self.__pBar__.printProgressBar(counter.value, self.leng, prefix='Progress:', suffix='', length=50)
        if params['method'][0] != 'als':

            return {'key': str(item[1])+'_'+str(item[2]), 'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'sigma': S, 'r-square': Rsq}
            # return {'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'sigma': S, 'r-square': Rsq}
        else:
            return {'key': str(item[1])+'_'+str(item[2]), 'amplitude': 0, 'FWHM': 0, 'center': 0, 'height': 0, 'r-square': Rsq, 'sigma': 0}
            # return {'amplitude': 0, 'FWHM': 0, 'center': 0, 'height': 0, 'r-square': Rsq, 'sigma': 0}


    def fit_array(self, x, y, params=None, limits=None, name=''):
        """
         The method makes peak fitting of a spectrum
         Args:
             x (floats): array-like x-axis of the fitted spectrum
             y (floats): array-like y-axis of the fitted spectrum. The length of x should be equal the length of y arrays.
             params{}(dict): Dictionary contains parameters of fitting
                    'range': range of fitting on absciss axis. This is array contains xmin and xmax values [xmin,xmax]. By default the input spectrum is fitted in all range.
                    'method': the array of names of the fitting method. The length of array is the number of fitting curve that will be used. There are Gaussian/Voigt/Lorenzian,PseudoVoigt for peak detection. For thermally stimullated processes the TSL glow curve of 1 order TSL_1, second order TSL_2 and thermally stimulated decay curve  TD_1 and TD_2 is available.
                    'amplitude': the array of values of initial amplitudes of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'center': the array of values of initial centers (x-coordinate) of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'width': the array of values of initial widths of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'baseline'[lam,p,niter]: the array contains manual parameters for baseline als
                            baseline[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline[2](int): Number of iterations of manual als baseline fitting
                    'baseline_auto'[lam,p,niter]: if this parameters is given, the auto fit of baseline occurs. There are array with the following parameters:
                            baseline_auto[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline_auto[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline_auto[2](int): Number of iterations of manual als baseline fitting
                    'kws' (dict): This is tolerance parameter of fitting. The default values are {'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12}
                    'max_nfev' (int): number of least square fit iterations.
                    'energy'[]: Activation energy for each peak in TSL_ and TD_ fits
                    'factor'[]: Frequency factor in TSL_ and TD_ fits

             limits(dict): Dictionary contains limits of parameter fitting {'amplitude':[],'center':[],'width':[],'baseline_auto':[]}
                    'amplitude'[][]: the array contains pairs minimum and maximum values of amplitude.
                        Example of limit amplitude calculation:
                            limit['amplitude'][0]*param['amplitude'] - min value of amplitude
                            limit['amplitude'][1]*param['amplitude'] - max value of amplitude
                    'center'[][]: the array contains pairs minimum and maximum values of peak centers.
                        Example of limit of center calculation:
                            limit['center'][0]-param['center'] - min value of peak center
                            limit['center'][1]+param['center] - max value of peak center
                    'width'[][]: the array contains pairs minimum and maximum values of peak width.
                        Example of limit peak width calculation:
                            limit['width'][0]*param['width'] - min value of peak width
                            limit['width'][1]*param['width] - max value of peak width
                    'baseline_auto'[[lam_min, lam_max], [p_min, p_max]]:
                        The limits of 2nd derivative constraint (lam) are lam_min<lam<lam_max
                        The limits of Weighting of positive residuals (p) are p_min<p<p_max
                    'energy'[E_min, E_max][]: Limits of activation energy E_min<E<E_max in TSL_ and TD_ fits
                    'factor'[f_min, f_max][]: Limits of frequency factor in TSL_ and TD_ fits
             name(str): the unique user-selected id of constructed fitting model.

         Returns:
             There two cases.
             In case of spectrum peak fitting the a dictionary containing arrays with fitting parameters. The length of the each array is equal to number of peaks in fitting procedure returns. The dictionary structure is:
                 {'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'sigma': S, 'r-square': Rsq}
                 'amplitude'(floats): array-like values of amplitudes of deconvoluted peaks
                 'FWHM'(floats): array-like values of peaks FWHM
                 'center'(floats): array-like values of x-coordinates of the peaks
                 'height'(floats): array-like values of y-coordinates of the peaks
                 'sigma'(floats): array-like values of peak sigmas
                 'r-square'(float): the R-square value of fitting
             In case of thermally stimulated process curve (TSL or TD) the dictionary with the following structure returns:
                 {'amplitude': A, 'factor': F, 'energy': E, 'r-square': Rsq}
                 'amplitude'(floats): array-like values of amplitudes of deconvoluted peaks
                 'factor'(doubles): array-like values of frequency factors
                 'energy'(floats): array-like values of activation energy
                 'r-square'(float): the R-square value of fitting

         """
        baseline_flag = True
        if params is None:
            params = {}
        if limits is None:
            limits = {}
        # if 'range' not in params:
        #    params['range'] = [x.min(), x.max()]
        # xmin = params['range'][0]
        # xmax = params['range'][1]
        # idx_min=self.__find_nearest_(x, xmin)
        # idx_max=self.__find_nearest_(x, xmax)
        # y=y[idx_min:idx_max]
        # x=x[idx_min:idx_max]
        if 'baseline' not in params:
            baseline_flag = False
        if baseline_flag:
            self.lam = params['baseline'][0]
            self.p = params['baseline'][1]

        if (params['method'][0] == 'TSL1') or (params['method'][0] == 'TSL2') or (params['method'][0] == 'TD1') or (
                params['method'][0] == 'TD2'):
            number_of_peaks = len(params['method'])
        elif (params['method'][0] == 'Als'):
            if 'baseline_auto' not in params:
                print('No start fitting parameters. Default parameters are used (lamda=1e0, p=0.5)') 
                params['baseline_auto']=[1e0, 0.5]           
        else:

            if 'center' not in params:
                return print('No centers')
            center = params['center']
            number_of_peaks = len(center)
            if 'baseline_auto' not in params:
                if baseline_flag:
                            b_line = self.__baseline_als__(y, self.lam, self.p)
                            y = np.subtract(y, b_line)
            if 'amplitude' not in params:
                for n in range(number_of_peaks):
                    params['amplitude'][n] = 1000
            if 'width' not in params:
                for n in range(number_of_peaks):
                    params['width'][n] = 2
            if 'method' not in params:
                for n in range(number_of_peaks):
                    params['method'][n] = 'Gaussian'
        if 'kws' not in params:
            params['kws'] = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8}
        fit_kws = params['kws']
        if 'max_nfev' not in params.keys():
            params['max_nfev'] = 1000
            
        mod = None
        if (params['method'][0] == 'Als'):
            if 'baseline_auto' not in limits:
                    limits['baseline_auto'] = [[2, 1e2], [0.1, 1]]
            
            mod = Model(self.__baseline_als__, prefix='bg_')
            mod.set_param_hint('lam', value=params['baseline_auto'][0], min=limits['baseline_auto'][0][0],
                              max=limits['baseline_auto'][0][1])
            mod.set_param_hint('p', value=params['baseline_auto'][1], min=limits['baseline_auto'][1][0],
                              max=limits['baseline_auto'][1][1])
            out = mod.fit(y, x=x, t=y, method='least_squares', fit_kws=fit_kws, max_nfev=params['max_nfev'])
            comps = out.eval_components(x=x)    
            self.map_baseline[name] = [x, out.best_fit]        
        else:           
            for i in range(number_of_peaks):
                this_mod = self.__make_model__(i, params, limits)
                if mod is None:
                    mod = this_mod
                else:
                    mod = mod + this_mod
            if 'baseline_auto' in params:
                if 'baseline_auto' not in limits:
                    limits['baseline_auto'] = [[1e4, 1e12], [0.0001, 0.1]]
                bl = Model(self.__baseline_als__, prefix='bg_')
                bl.set_param_hint('lam', value=params['baseline_auto'][0], min=limits['baseline_auto'][0][0],
                              max=limits['baseline_auto'][0][1])
                bl.set_param_hint('p', value=params['baseline_auto'][1], min=limits['baseline_auto'][1][0],
                              max=limits['baseline_auto'][1][1])
                mod = mod + bl

                out = mod.fit(y, x=x, t=y, method='least_squares', fit_kws=fit_kws, max_nfev=params['max_nfev'])
                comps = out.eval_components(x=x)
                self.map_baseline[name] = [x, out.best_fit]
                self.map_bline[name] = [x, comps['bg_'], out.params['bg_lam'], out.params['bg_p']]

            else:
                out = mod.fit(y, x=x, method='least_squares', fit_kws=fit_kws)
                self.map_baseline[name] = [x, out.best_fit]
                comps = out.eval_components(x=x)
                if baseline_flag:
                    self.map_bline[name] = [x, b_line, self.lam, self.p]
                else:
                    self.map_bline[name] = [x, np.zeros(len(x)), 0, 0]
        self.components = comps   

        if (params['method'][0] == 'TSL1') or (params['method'][0] == 'TSL2') or (params['method'][0] == 'TD1') or (
                params['method'][0] == 'TD2'):
            A = np.array([out.params['f' + repr(num) + '_amplitude'] for num in range(number_of_peaks)])
            F = np.array([out.params['f' + repr(num) + '_factor'] for num in range(number_of_peaks)])
            E = np.array([out.params['f' + repr(num) + '_energy'] for num in range(number_of_peaks)])
            Rsq = 1 - out.redchi / np.var(y, ddof=0)
            return {'amplitude': A, 'factor': F, 'energy': E, 'r-square': Rsq}
        if (params['method'][0] == 'Als'):
            A=out.params['bg_lam']
            F=out.params['bg_p']
            Rsq = 1 - out.redchi / np.var(y, ddof=0)
            return {'lam': A, 'p': F, 'r-square': Rsq}
        else:
            A = np.array([out.params['f' + repr(num) + '_amplitude'] for num in range(number_of_peaks)])
            FWHM = np.array([out.params['f' + repr(num) + '_fwhm'] for num in range(number_of_peaks)])
            C = np.array([out.params['f' + repr(num) + '_center'] for num in range(number_of_peaks)])
            H = np.array([out.params['f' + repr(num) + '_height'] for num in range(number_of_peaks)])
            Rsq = 1 - out.redchi / np.var(y, ddof=0)
            S = np.array([out.params['f' + repr(num) + '_sigma'] for num in range(number_of_peaks)])
            return {'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'r-square': Rsq, 'sigma': S, 'p': out.params['bg_p'], 'lam': out.params['bg_lam']}

    @staticmethod
    def __init_C__(args):
        """
        The private method of progress bar initialization
        """
        global counter
        counter = args

    def find_intensity(self, map_spectra, params, limits):
        """
        The method constructs a map with fitting parameters from the hyperspectral map.
        Args:
            map_spectra(dict): the dictionary contains hyperspectral map. Each element having name [str(mx) + '_' + str(my)] of the dictionary contains the spectrum measured at point x, y of the hyperspectral map with the following structure:
               [np.array(spectrum), mx, my, nv, nr, x1, x2, step]
                    np.array(spectrum): ordinate values of spectrum in the hyperspectral map in the point with mx, my coordinate
                    mx: x-coordinate of the point in the hyperspectral map
                    my: y-coordinate of the point in the hyperspectral map
                    nv: number of points in the spectrum np.array(spectrum)
                    nr: number of points in the reduced spectrum np.array(spectrum). The value is equal to nv in the most cases
                    x1: minimal wavenumber in the spectrum
                    x2: maximum wavenumber in the spectrum
                    step: distance between adjacent x values (step of wavenumber measurement)
            params{}(dict): Dictionary contains parameters of fitting
                    'range': range of fitting on absciss axis. This is array contains xmin and xmax values [xmin,xmax]. By default the input spectrum is fitted in all range.
                    'method': the array of names of the fitting method. The length of array is the number of fitting curve that will be used. There are Gaussian/Voigt/Lorentzian,PseudoVoigt for peak detection. For thermally stimullated processes the TSL glow curve of 1 order TSL_1, second order TSL_2 and thermally stimulated decay curve  TD_1 and TD_2 is available.
                    'amplitude': the array of values of initial amplitudes of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'center': the array of values of initial centers (x-coordinate) of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'width': the array of values of initial widths of peaks in peak fitting procedure. The length of array is the number of fitting curve that will be used.
                    'baseline'[lam,p,niter]: the array contains manual parameters for baseline als
                            baseline[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline[2](int): Number of iterations of manual als baseline fitting
                    'baseline_auto'[lam,p,niter]: if this parameters is given, the auto fit of baseline occurs. There are array with the following parameters:
                            baseline_auto[0](double): 2nd derivative constraint in case of manual als baseline fitting
                            baseline_auto[1](double): Fitted Weighting of positive residuals in algorithm
                            baseline_auto[2](int): Number of iterations of manual als baseline fitting
                    'kws' (dict): This is tolerance parameter of fitting. The default values are {'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12}
                    'max_nfev' (int): number of least square fit iterations.
                    'energy'[]: Activation energy for each peak in TSL_ and TD_ fits
                    'factor'[]: Frequency factor in TSL_ and TD_ fits

            limits(dict): Dictionary contains limits of parameter fitting {'amplitude':[],'center':[],'width':[],'baseline_auto':[]}
                    'amplitude'[][]: the array contains pairs minimum and maximum values of amplitude.
                        Example of limit amplitude calculation:
                            limit['amplitude'][0]*param['amplitude'] - min value of amplitude
                            limit['amplitude'][1]*param['amplitude'] - max value of amplitude
                    'center'[][]: the array contains pairs minimum and maximum values of peak centers.
                        Example of limit of center calculation:
                            limit['center'][0]-param['center'] - min value of peak center
                            limit['center'][1]+param['center] - max value of peak center
                    'width'[][]: the array contains pairs minimum and maximum values of peak width.
                        Example of limit peak width calculation:
                            limit['width'][0]*param['width'] - min value of peak width
                            limit['width'][1]*param['width] - max value of peak width
                    'baseline_auto'[[lam_min, lam_max], [p_min, p_max]]:
                        The limits of 2nd derivative constraint (lam) are lam_min<lam<lam_max
                        The limits of Weighting of positive residuals (p) are p_min<p<p_max
                    'energy'[E_min, E_max][]: Limits of activation energy E_min<E<E_max in TSL_ and TD_ fits
                    'factor'[f_min, f_max][]: Limits of frequency factor in TSL_ and TD_ fits
        Returns:
              x(ints): array-like mx coordinates of each point on hyperspectral map
              y(ints): array-like my coordinates of each point on hyperspectral map
              z1(dict): array of dictionaries with the following structure:
                 {'amplitude': A, 'FWHM': FWHM, 'center': C, 'height': H, 'sigma': S, 'r-square': Rsq}
                 'amplitude'(floats): array-like values of amplitudes of deconvoluted peaks
                 'FWHM'(floats): array-like values of peaks FWHM
                 'center'(floats): array-like values of x-coordinates of the peaks
                 'height'(floats): array-like values of y-coordinates of the peaks
                 'sigma'(floats): array-like values of peak sigmas
                 'r-square'(float): the R-square value of fitting

        """
        manager = Manager()
        self.map_baseline = manager.dict()
        self.map_bline = manager.dict()
        global counter
        counter = Value('i', 0)
        pool = Pool(processes=num_proc, initializer=self.__init_C__, initargs=(counter,))
        self.param = params
        self.limit = limits
        print(colored('Peak fitting:', 'cyan'))
        self.leng = len(map_spectra)
        self.__pBar__.printProgressBar(0, self.leng, prefix='Progress:', suffix='', length=50)
        x = np.array([item[1] for item in map_spectra.values()])
        y = np.array([item[2] for item in map_spectra.values()])
        dd = map_spectra
        cs = int(len(dd) / num_proc)
        z = pool.map_async(self.map_intensity, dd.values(), chunksize=cs)

        z.wait()
        z1 = z.get()
        pool.close()
        return x, y, z1
    
    def __find_nearest_(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    @staticmethod
    def peakdet(y_axis, lookahead=3, delta=0.02):
        """
        The method return peaks position on the spectra.
        Converted from/based on a MATLAB script at: http://billauer.co.il/peakdet.html
        Args:
            y_axis(floats): A list-like containing the signal over which to find peaks
            lookahead(int): A distance to look ahead from a peak candidate to determine (default=3)
            delta(floats): this specifies a minimum difference between a peak and the following points, before a peak may be considered a peak. Useful
            to hinder the function from picking up false peaks towards to end of the signal. To work well delta should be set to delta >= RMSnoise * 5.

        Returns:
            The ordered array of peaks found in y_axis
        """
        max_peaks = []
        dump = []  # Used to pop the first hit which almost always is false
        # store data length for later use
        length = len(y_axis)
        x_axis = range(length)
        # perform some checks
        if lookahead < 1:
            raise ValueError("Lookahead must be '1' or above in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        # maxima and minima candidates are temporarily stored in
        # mx and mn respectively
        mn, mx = np.Inf, -np.Inf
        mxpos = 0
        # Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                           y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
            if y < mx - delta and mx != np.Inf:
                # Maxima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].max() < mx:

                    if len(y_axis[index - lookahead:index + lookahead]) > 0:
                        max_peaks.append(mxpos)
                        dump.append(True)
                        # set algorithm to only find minima now
                        mx = np.Inf
                        mn = np.Inf
                        if index + lookahead >= length:
                            # end is within lookahead no more peaks can be found
                            break
                        continue
            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    dump.append(False)
                    mn = -np.Inf
                    mx = -np.Inf
                    if index + lookahead >= length:
                        # end is within lookahead no more peaks can be found
                        break

        # Remove the false hit on the first value of the y_axis
        try:
            if dump[0]:
                max_peaks.pop(0)
            del dump
        except IndexError:
            print("No peaks found")
            pass                
            
        return max_peaks

    @staticmethod
    def __ampd__(sigInput):
        """Find the peaks in the signal with the AMPD algorithm.

            Original implementation by Felix Scholkmann et al. in
            "An Efficient Algorithm for Automatic Peak Detection in
            Noisy Periodic and Quasi-Periodic Signals", Algorithms 2012,
             5, 588-603
            Args:
                sigInput(floats): ndarray
                    The 1D signal given as input to the algorithm
            Returns:
                pks: ndarray
                    The ordered array of peaks found in sigInput
        """

        # Create preprocessing linear fit
        sigTime = np.arange(0, len(sigInput))

        fitPoly = np.polyfit(sigTime, sigInput, 1)
        sigFit = np.polyval(fitPoly, sigTime)

        # Detrend
        dtrSignal = sigInput - sigFit

        N = len(dtrSignal)
        L = int(N / 2.0) - 1

        # Generate random matrix
        LSM = np.random.uniform(1.0, 2.0, size=(L, N))  # uniform + alpha = 1

        # Local minima extraction
        for k in np.arange(1, L):
            locMax = np.zeros(N, dtype=bool)
            mask = np.array((dtrSignal[k:N - k - 1] > dtrSignal[0: N - 2 * k - 1]) & (
                    dtrSignal[k:N - k - 1] > dtrSignal[2 * k: N - 1]))
            mask = mask.flatten()

            locMax[k:N - k - 1] = mask
            LSM[k - 1, locMax] = 0

        # Find minima
        G = np.sum(LSM, 1)
        l_ = np.where(G == G.min())[0][0]

        LSM = LSM[0:l_, :]

        S = np.std(LSM, 0)

        pks = np.flatnonzero(S == 0)
        return pks

    # Fast AMPD

    def ampd_fast(self, sigInput, order):
        """A slightly faster version of AMPD which divides the signal in 'order' windows
            Args:
                sigInput: ndarray
                    The 1D signal given as input to the algorithm
                order: int
                    The number of windows in which sigInput is divided
            Returns:
                pks: ndarray
                    The ordered array of peaks found in sigInput
        """

        # Check if order is valid (perfectly separable)
        print(len(sigInput))
        if len(sigInput) % order != 0:
            print("AMPD: Invalid order, decreasing order")
            while len(sigInput) % order != 0:
                order -= 1
            print("AMPD: Using order " + str(order))

        N = int((len(sigInput) / order / 2.0)) + 1

        # Loop function calls
        for i in range(0, len(sigInput) - N, N):
            # print("\t sector: " + str(i) + "|" + str((i+2*N-1)))
            pksTemp = self.__ampd__(sigInput[i:(i + 2 * N - 1)])
            if i == 0:
                pks = pksTemp
            else:
                pks = np.concatenate((pks, pksTemp + i))

        # Keep only unique values

        index = 1
        out = []
        while index < len(pks):
            if (pks[index] - pks[index - 1]) * 100.0 / pks[index] > 5:
                out.append(pks[index - 1])
                out.append(pks[index])
            index += 1
        pks = np.unique(out)
        print(pks)
        return pks
