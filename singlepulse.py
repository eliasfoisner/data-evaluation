import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly_resampler import FigureResampler


def calc_diameter(mass, density): # in nm
    diameter = (3*mass/(4*density*np.pi))**(1/3) * 2 * 1e7
    return diameter.real


def extract_string(text: str, first_character: str, second_character: str, right: bool = True):
    """
    Looks for the index of "first_character" in the string "text" and looks for the index of the first occurence of "second_character" relative to it.
    If right = True (default), the first occurence of "second_character" is searched to the right.
    """
    if right:
        first_index = text.find(first_character)
        second_index = text.find(second_character, first_index+len(first_character), len(text))
        if first_index != -1 and second_index != -1:
            value = text[first_index+len(first_character):second_index]
            return value
        return None
    else:
        first_index = text.find(first_character)
        second_index = text.rfind(second_character, 0, first_index)
        if first_index != -1 and second_index != -1:
            value = text[second_index+len(second_character):first_index]
            return value
        return None


def generate_colors(n, colormap_name: str = 'viridis'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i/n) for i in range(n)]
    return colors


def generate_marker_list(n, available_markers):
    # Wenn n größer als die Anzahl der verfügbaren Marker ist, wiederhole sie
    markers = [available_markers[i % len(available_markers)] for i in range(n)]
    return markers


def check_nanomodul(file):
    data = pd.read_csv(file, skiprows=0)
    if type(data.iloc[0, 0]) == str:
        return False
    else:
        return True


class SinglePulse:
    """
    The basic concept evolves around two DataFrames self.data and self.peaks which are filled with data when applying methods.
    Methods return the specific calculations in addition to saving the data to these DataFrames.
    """
    layout = go.Layout(legend=dict(orientation='h', yanchor='top', xanchor='center', y=-0.2, x=0.5))
    fig = FigureResampler(go.Figure(layout=layout))

    def __init__(self, file_path: str):
        """
        Initialize a regular Syngistix file or Nanomodul file (set nanomodul == True).
        """
        self.file_name = file_path.split("/")[-1]
        self.__timescale_count = 0
        self.__nanomodul = check_nanomodul(file_path)

        if not self.__nanomodul:
            try:
                self.data = pd.read_csv(file_path, skiprows=1)
                self.measured_isotopes = self.data.loc[:, self.data.columns != "Time in Seconds "].columns
            except:
                self.data = pd.read_csv(file_path, skiprows=2)
                self.measured_isotopes = self.data.loc[:, self.data.columns != "Time in Seconds "].columns
        else:
            self.data = pd.read_csv(file_path, skiprows=0).iloc[:, :-1]
            self.measured_isotopes = self.data.columns

        self.peaks = dict()
        for isotope in self.measured_isotopes:
            self.peaks[isotope] = pd.DataFrame(columns=['index', 'time', 'height', 'width', 'background', 'area'])

    def merge(self, objects: list):
        """
        Use this method to merge the self.data DataFrames of Syngistix objects (List). The objects are deleted afterwards.
        """
        self.data = self.data.concat(objects)
        del objects

    def timescale(self, isotope: str, cycle_time: float = False):
        """
        Add a timescale for a specific isotope to the dataframe.
        Enter a cycle_time if nanomodul files are used.
        """
        assert isotope in self.measured_isotopes, f"Isotope '{isotope}' was not measured! {self.file_name}"
        if not self.__nanomodul:
            self.data["Time in Seconds "] = np.linspace(0, max(self.data["Time in Seconds "]), len(self.data))
            self.__cycle_time = self.data["Time in Seconds "][1] - self.data["Time in Seconds "][0]
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return timestamps
        elif self.__nanomodul:
            assert cycle_time, "Please specify cycle time when using Nanomodul files!"
            self.__cycle_time = cycle_time
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return timestamps

    def savgol(self, isotope: str, window_length: int = 100, polyorder: int = 2, deriv: int = 0):
        """
        Smooth the data using the scipy.savgol_filter() method. Check method parameters if desired.
        """
        self.data[f"{isotope}_savgol"] = sc.signal.savgol_filter(self.data[isotope], window_length=window_length, polyorder=polyorder, deriv=deriv)
        return self.data[f"{isotope}_savgol"]

    def global_background(self, isotope: str, start: float = 0.1, end: float = 1):
        start_idx = int(start // self.__cycle_time)
        end_idx = int(end // self.__cycle_time)
        self.__global_bg = np.median(self.data[isotope][start_idx:end_idx])
        self.data[f"{isotope}_corr"] = self.data[f"{isotope}"] - self.__global_bg
        return self.__global_bg

    def peak_finding(self, isotope: str, threshold: float = 50, distance: float = 10e-3, width: float = 10e-6, savgol: bool = False):
        """
        Find peaks using scipy.signal.find_peaks(). Savitzky-Golay signal can be used optionally by setting savgol == True. Check find_peaks() settings if desired.
        """
        assert f'{isotope}_time' in self.data.columns, f"Calculate '{isotope}_time' before peak detection!"
        #self.peaks[isotope] = pd.DataFrame()
        if savgol:
            assert f'{isotope}_savgol' in self.data.columns, f"Please provide '{isotope}_savgol' for peak detection!"
            self.peaks[isotope]['index'] = sc.signal.find_peaks(self.data[f"{isotope}_savgol"],
                                                                height=threshold,
                                                                distance=distance/self.__cycle_time,
                                                                width=width/self.__cycle_time)[0]
        else:
            self.peaks[isotope]['index'] = sc.signal.find_peaks(self.data[f"{isotope}"],
                                                                height=threshold,
                                                                distance=distance/self.__cycle_time,
                                                                width=width/self.__cycle_time)[0]
        peak_idx = self.peaks[isotope]['index']
        self.peaks[isotope]['time'] = self.data[f'{isotope}_time'].iloc[peak_idx].tolist()
        self.peaks[isotope]['height'] = self.data[isotope].iloc[peak_idx].tolist()
        return self.peaks[isotope]['time']

    def peak_width(self, isotope: str, criterion: int = 10):
        """
        Calculate peak width using scipy.signal.peak_widths() method.
        Default value for criterion is 10 for 10-percent criterion.
        """
        width = sc.signal.peak_widths(x=self.data[f'{isotope}'], peaks=self.peaks[isotope][f'index'], rel_height=(1 - criterion / 100))
        self.peaks[isotope]['width'] = width[0] * self.__cycle_time
        peak_time_left = [float(i*self.__cycle_time) for i in width[2]]
        peak_time_right = [float(i*self.__cycle_time) for i in width[3]]
        self.__width_boundaries = list(zip(peak_time_left, peak_time_right))
        return self.peaks[isotope]['width']

    def peak_width_alt(self, isotope: str, criterion: int = 10, timeframe: float = 1e-3):
        """
        Alternative way to calculate peak width.
        Looks at the datapoints in a given timeframe around the peak maximum and compares the intensity to a defined threshold (criterion).
        If the intensity is lower, the peak boundaries are reached.
        The width is the difference between the two timestamps which are determined in this manner.
        """
        datapoint_amount = int(timeframe // self.__cycle_time) // 2
        widths = []
        for i in self.peaks[isotope]['index']:
            threshold_intensity = self.data[f'{isotope}'].iloc[i] * criterion / 100
            intensities = self.data[isotope][i-datapoint_amount:i+datapoint_amount]
            timestamps = self.data[f'{isotope}_time'][i-datapoint_amount:i+datapoint_amount]
            above_threshold = [t for k, t in list(zip(intensities, timestamps)) if k > threshold_intensity]
            widths.append((max(above_threshold) - min(above_threshold)) * 1e6)
        self.peaks[isotope]['width_alt'] = widths
        return self.peaks[isotope]['width_alt']

    def peak_background(self, isotope: str, distance: float = 0, window_size: float = 3):
        """
        Calculates the peak background left and right to the peak.
        The distance_from_peak specifies how far the background window is from the peak boundaries.
        The window_size specifies how many peak widths the background window is wide (on each side).
        """
        backgrounds = []
        self.__bgwindowsize = window_size
        self.__bgfactor = distance

        for i in range(len(self.peaks[isotope]['index'])):
            peak_width = self.peaks[isotope]['width'][i]
            peak_time = self.peaks[isotope]['time'][i]
            peak_time_left, peak_time_right = self.__width_boundaries[i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            bgbound_left_outer_idx = (peak_time - peak_width * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)) / self.__cycle_time
            bgbound_left_inner_idx = (peak_time - peak_width * self.__bgfactor * (1 - asymmetry)) / self.__cycle_time
            left_data = self.data[isotope][int(bgbound_left_outer_idx):int(bgbound_left_inner_idx)]

            bgbound_right_inner_idx = (peak_time + peak_width * self.__bgfactor * asymmetry) / self.__cycle_time
            bgbound_right_outer_idx = (peak_time + peak_width * (self.__bgfactor * asymmetry + self.__bgwindowsize)) / self.__cycle_time
            right_data = self.data[isotope][int(bgbound_right_inner_idx):int(bgbound_right_outer_idx)]

            backgrounds.append((left_data.mean(), right_data.mean()))

        self.__backgrounds = backgrounds

        self.peaks[isotope]['background'] = [i+j for i, j in backgrounds]

        return self.peaks[isotope]['background']

    def peak_area(self, isotope: str, mode: str = 'trapezoid', resize: float = 1, local_background: bool = False):
        """
        Calculates peak area using scipy 'trapezoid' or 'romberg'.
        If local_background = True, the peak_background() has to be calculated beforehand.
        If local_background is not specified or False, the global background is used (calculated from second 0.1 to 1).
        The integration window is peak_width * resize_factor.
        """
        self.__intfactor = resize
        areas = []
        for i in range(len(self.peaks[isotope]['index'])):
            peak_time = self.peaks[isotope]['time'][i]
            peak_width = self.peaks[isotope]['width'][i]
            peak_background = self.peaks[isotope]['background'][i]
            peak_time_left, peak_time_right = self.__width_boundaries[i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            intbound_left_idx = int((peak_time - peak_width * self.__intfactor * (1 - asymmetry)) / self.__cycle_time)
            intbound_right_idx = int((peak_time + peak_width * self.__intfactor * asymmetry) / self.__cycle_time)

            if local_background:
                assert peak_time.empty is False and not peak_background.isna().all(), f"Calculate local background before integration or set local_background = False!"
                signal = [k - self.peaks[isotope]['background'][i] for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]
            else:
                signal = [k - self.global_background(isotope, start=0.1, end=1) for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]

            if mode == 'trapezoid':
                areas.append(sc.integrate.trapezoid(signal))
            elif mode == 'romberg':
                areas.append(sc.integrate.romberg(signal))

        self.peaks[isotope]['area'] = areas
        return self.peaks[isotope]['area']

    def calibrate(self, isotope, calibration: object):
        """
        Uses FilmCalibration object to append 'mass' column to the peaks dataframe.
        """
        self.peaks[isotope]['mass'] = (self.peaks[isotope]['area'] - calibration.calibration[isotope].intercept) / calibration.calibration[isotope].slope

    def area_ratio(self, isotope_one: str, isotope_two: str):
        """
        Calculate the area ratio between two isotopes - for every particle separately. 
        This is achieved by looking at every peak of the large area isotope and checking if there was an adjacent smaller area peak found. 
        If yes, the ratio is calculated and stored in self.__ratios. 
        :param isotope_one: The isotope with the supposed larger area.
        :param isotope_two: The isotope with the smaller area.
        :return: Ratio between the two areas as float.
        """
        ratios = []
        for i_one in range(len(self.peaks[isotope_one])):
            for i_two in range(len(self.peaks[isotope_two])):
                if abs(self.peaks[isotope_one]["time"][i_one] - self.peaks[isotope_two]["time"][i_two]) < 5e-3:
                    area_one = self.peaks[isotope_one]["area"][i_one]
                    area_two = self.peaks[isotope_two]["area"][i_two]
                    ratio = area_two / (area_one + area_two)
                    ratios.append(ratio)
        return pd.Series(ratios)



    def gdc_diameter(self, isotope, mode):
        # mass fractions of one particle
        #gadolinium = 0.17997880
        #cerium = 0.64147307
        #oxygen = 0.17854813

        if mode == '5':
            gadolinium = 0.149569
            cerium = 0.675891
            oxygen = 0.174565
        if mode == '10':
            gadolinium = 0.124
            cerium = 0.699
            oxygen = 0.176
        if mode == '100':
            gadolinium = 0.098053
            cerium = 0.725061
            oxygen = 0.1768607

        if isotope == 'CeO/156':
            diameter = [calc_diameter((m * 140 / 156) / cerium, 7.32) for m in self.peaks[isotope]['mass']]
            self.peaks[isotope]['diameter'] = diameter
        if isotope == 'GdO/174':
            diameter = [calc_diameter((m * 158 / 174) / gadolinium, 7.32) for m in self.peaks[isotope]['mass']]
            self.peaks[isotope]['diameter'] = diameter

    def plot(self, isotope: str, savgol: bool = False, integration: bool = False, peaks: bool = False, background: bool = False, width: bool = False):
        """
        Plot the data and optional elements such as integration boundaries or local background.
        """

        SinglePulse.fig.update_layout(height=800, xaxis_title="Time (s)", yaxis_title="Intensity")
        SinglePulse.fig.update_traces(line_width=1, selector=dict(type='scatter'), showlegend=True)

        traces = []
        shapes = []

        assert f'{isotope}_time' in self.data.columns, f"'{isotope}_time' required!"
        original_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}"], mode='lines', name=f"{isotope} ({self.file_name})")
        traces.append(original_trace)

        if savgol:
            assert f'{isotope}_savgol' in self.data.columns, f"Use savgol() before plotting Savitzky-Golay signal!"
            savgol_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}_savgol"], mode='lines', name=f"'{self.file_name}' ({isotope}_savgol)")
            traces.append(savgol_trace)

        peak_index = self.peaks[isotope]['index']
        peak_time = self.peaks[isotope]['time']
        peak_height = self.peaks[isotope]['height']
        peak_width = self.peaks[isotope]['width']
        peak_background = self.peaks[isotope]['background']

        if peaks:
            peak_trace = go.Scatter(x=peak_time, y=peak_height, mode='markers', marker_color="red", name=f"{isotope} peaks ({self.file_name})")
            traces.append(peak_trace)

        for i in range(len(peak_index)):
            if width:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting peak widths!"
                peak_time_left, peak_time_right = self.__width_boundaries[i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                width_shape = dict(type='rect', x0=peak_time_left, x1=peak_time_right, y0=0, y1=peak_height[i], line_width=0, fillcolor='blue', opacity=0.1)
                shapes.append(width_shape)

            if background:
                assert peak_time.empty is False and not peak_background.isna().all(), "Use peak_finding() and peak_background() before plotting backgrounds!"
                bgleft, bgright = self.__backgrounds[i]
                peak_time_left, peak_time_right = self.__width_boundaries[i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                bgbound_left_outer_time = peak_time[i] - peak_width[i] * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)
                bgbound_left_inner_time = peak_time[i] - peak_width[i] * self.__bgfactor * (1 - asymmetry)
                bgbound_right_inner_time = peak_time[i] + peak_width[i] * self.__bgfactor * asymmetry
                bgbound_right_outer_time = peak_time[i] + peak_width[i] * (self.__bgfactor * asymmetry + self.__bgwindowsize)
                bg_shape_left = dict(type="line", y0=bgleft, y1=bgleft, x0=bgbound_left_outer_time, x1=bgbound_left_inner_time, line=dict(color="red", width=3))
                bg_shape_right = dict(type="line", y0=bgright, y1=bgright, x0=bgbound_right_inner_time, x1=bgbound_right_outer_time, line=dict(color="red", width=3))
                shapes.append(bg_shape_left)
                shapes.append(bg_shape_right)

            if integration:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting integration boundaries!"
                peak_time_left, peak_time_right = self.__width_boundaries[i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                intbound_left_time = peak_time[i] - peak_width[i] * (1 - asymmetry) * self.__intfactor
                intbound_right_time = peak_time[i] + peak_width[i] * asymmetry * self.__intfactor
                width_shape = dict(type='rect', x0=intbound_left_time, x1=intbound_right_time, y0=0, y1=peak_height[i], line_width=0,
                                   fillcolor='green', opacity=0.1)
                shapes.append(width_shape)

        for trace in traces:
            SinglePulse.fig.add_trace(trace)
        for shape in shapes:
            SinglePulse.fig.add_shape(shape)


class FilmCalibration:
    """
    Uses SinglePulse Objects for calibration. Spotsizes: x and y in micrometers, z in nanometers.
    """
    def __init__(self, standards: list, spotsizes: list[tuple], film_concentration_percent: float):
        self.__standards = standards
        self.__spotsizes = spotsizes
        self.__film_concentration_percent = film_concentration_percent
        self.calibration = dict()
        self.analyte_mass = dict()
        self.analyte_intensity = dict()

    def reg_function(self, isotope: str, analyte_ppm_film: float = 0, force_zero: bool = False, mass_correction: float = 1):
        self.analyte_mass[isotope] = []
        self.analyte_intensity[isotope] = []
        if force_zero:
            self.analyte_mass[isotope].append(0)
            self.analyte_intensity[isotope].append(0)
        for x, y, z in self.__spotsizes:
            ablated_drymass = (z * 1e-7 * x * y * 1e-8) * (self.__film_concentration_percent / 100)
            if analyte_ppm_film != 0:
                ablated_drymass = ablated_drymass * analyte_ppm_film * 1e-6
            self.analyte_mass[isotope].append(ablated_drymass*mass_correction)
        for s in self.__standards:
            self.analyte_intensity[isotope].append(s.peaks[isotope]['area'].mean())
        self.analyte_intensity[isotope] = [float(i) for i in self.analyte_intensity[isotope]]
        self.calibration[isotope] = sc.stats.linregress(x=self.analyte_mass[isotope], y=self.analyte_intensity[isotope])
        print(f'y = {self.calibration[isotope].intercept:+.3e} {self.calibration[isotope].slope:+.3e} * x (R^2 = {(self.calibration[isotope].rvalue ** 2) * 100:.2f} %)')
        return self.calibration[isotope]

    def reg_plot(self, isotope: str, confidence_interval: int = 90):
        sns.regplot(x=self.analyte_mass[isotope], y=self.analyte_intensity[isotope], ci=confidence_interval)
        plt.show()



"""
aft_values = [-150, 0, 50, 150, 350, 500]
colors = el.generate_colors(len(aft_values), colormap_name = 'hsv') # viridis, winter, plasma, summer, hsv
available_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
markers = el.generate_marker_list(len(aft_values), available_markers)
"""