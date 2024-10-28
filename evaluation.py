import sys, os, glob
sys.path.append('/Users/eliasfoisner/Documents/GitHub/data-evaluation')
import singlepulse as el
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tkinter.filedialog import askopenfilenames
from datetime import datetime

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern'] # oder eine andere LaTeX-Schriftart
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.labelsize'] = 14


selected_files = askopenfilenames(title="Wähle .csv Files zum Bearbeiten aus", filetypes=[("CSV Files", "*.csv")])
objects = dict()
results = []
ratios = []

for file in selected_files:
    f = os.path.basename(file)
    objects[f] = el.SinglePulse(file)
    isotopes = objects[f].measured_isotopes.tolist()

    for i in isotopes:
        if i == "Eu/153":
            criterion, threshold = 10, 40
        if i == "EuO/169":
            criterion, threshold = 1, 5
        objects[f].timescale(i, cycle_time=50e-6)
        objects[f].savgol(i)
        objects[f].peak_finding(i, threshold=threshold, distance=100e-3)
        objects[f].peak_width(i, criterion=criterion)
        objects[f].peak_area(i, resize=1.5)
        objects[f].plot(i, peaks=True, integration=True)

        results.append(
            {
                'species': i,
                'no. of peaks': len(objects[f].peaks[i]),
                f'peak width ({criterion}% criterion)': objects[f].peaks[i]['width'].mean() * 1e6,
                'peak width stdev': objects[f].peaks[i]['width'].std() * 1e6,
                f'peak height': objects[f].peaks[i]['height'].mean(),
                'peak height stdev': objects[f].peaks[i]['height'].std(),
                'peak area': objects[f].peaks[i]['area'].mean(),
                'peak area stdev': objects[f].peaks[i]['area'].std()
            }
        )
        print(f"Done: {f}")

    #ratios.append(objects[f].area_ratio("Eu/153", "EuO/169"))

#ratios = pd.concat(ratios)
#ratios = ratios.reset_index(drop=True).squeeze()
#print(ratios.mean(), ratios.std())

results = pd.DataFrame(results)
area_euo = results[results["species"] == "EuO/169"]["peak area"].mean()
area_eu = results[results["species"] == "Eu/153"]["peak area"].mean()
ratio = area_euo / (area_euo + area_eu)
print(results)
print(ratio)


#results = results.apply(pd.to_numeric, errors='ignore')
#results = results.dropna(axis=1, how='all')
#results = results.fillna(0)
#results.to_excel('output.xlsx')
#print(results)

el.SinglePulse.fig.show_dash()