# -*- coding: UTF-8 -*-

from datetime import datetime, timedelta
from matplotlib import pylab, get_backend
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.pylab import get_current_fig_manager
from matplotlib.widgets import Button
from numpy import (append, asarray, arange, concatenate, delete, diff,
                   inf, isnan, linspace, max, mean, mod, nan,
                   nanmean, nanstd, ones, sqrt, std, sum,
                   where, zeros)

# Global variables and general plot adjusts ====================================
PrefixID = "./results/ephem_"  # datetime.strftime(datetime.now(), "%Y%m%dT%H%M%S_")

pylab.rc('text', usetex=True)
FontTitle = FontProperties(size=22)
FontLabel = FontProperties(size=18)
FontTicks = FontProperties(size=16)
FontLegend = FontProperties(size=14)

# Load raw data ================================================================
# FileName = "../data/SN_d_tot_V2.0.txt"
FileName = "./data/data.txt"
DataFile = open(FileName, "r")

rawData = []
rawDate = []

for Line in DataFile:
    Line = Line.split()
    rawDate.append(datetime.strptime("-".join(Line[0:3]), "%Y-%m-%d"))
    rawData.append(float(Line[5]))

rawDate = asarray(rawDate)
rawData = asarray(rawData)

# Averaging to get monthly data ================================================
registryLength = len(rawData)
registryEnd = registryLength - 1

monthlyDate = []
monthlyData = []
monthlyStDv = []

pPos = 0

for nPos in range(registryLength):
    if nPos == registryEnd or rawDate[nPos].month != rawDate[nPos+1].month:
        monthlyDate.append(datetime(rawDate[nPos].year, rawDate[nPos].month, 1))

        if sum(isnan(rawData[pPos:nPos+1])) == len(rawData[pPos:nPos+1]):
            monthlyData.append(0.00)
            monthlyStDv.append(0.00)
            pPos = nPos+1
            continue

        monthlyData.append(nanmean(rawData[pPos:nPos+1]))
        monthlyStDv.append(nanstd(rawData[pPos:nPos+1]))
        pPos = nPos+1

monthlyDate = asarray(monthlyDate)
monthlyData = asarray(monthlyData)
monthlyStDv = asarray(monthlyStDv)

FileName = PrefixID + "monthly_average.dat"
DataFile = open(FileName, "w")
DataFile.write("# Year   Month    mean SSN   std SSN\n")

DataStr = "  {0:0>4d}      {1:0>2d}      {2:6.2f}    {3:6.2f}\n"

for (Date, Data, StDv) in zip(monthlyDate, monthlyData, monthlyStDv):
    DataFile.write(DataStr.format(Date.year,
                                  Date.month,
                                  Data,
                                  StDv))

DataFile.close()

# Averaging to get smoothed data ===============================================
registryLength = len(monthlyData)
registryEnd = registryLength - 1

smoothDate = monthlyDate
smoothData = zeros(registryLength)*nan
smoothStDv = zeros(registryLength)*nan

weights = asarray([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=float)
weights = weights/sum(weights)
period = len(weights)

shift, trail = divmod(period, 2)

for nPos in range(registryLength):
    if nPos-shift < 0:
        continue
    if nPos+shift+trail > len(monthlyData):
        break

    window = monthlyData[nPos-shift:nPos+shift+trail]
    smoothData[nPos] = sum(weights*window)
    smoothStDv[nPos] = sqrt(sum(((window-smoothData[nPos])**2)*weights))

smoothDate = smoothDate[shift:-shift]
smoothData = smoothData[shift:-shift]
smoothStDv = smoothStDv[shift:-shift]

FileName = PrefixID + "smooth_average.dat"
DataFile = open(FileName, "w")
DataFile.write("# Year   Month    running    running\n")
DataFile.write("#                 mean SSN   std SSN\n")

DataStr = "  {0:0>4d}      {1:0>2d}      {2:6.2f}    {3:6.2f}\n"

for (Date, Data, StDv) in zip(smoothDate, smoothData, smoothStDv):
    DataFile.write(DataStr.format(Date.year,
                                  Date.month,
                                  Data,
                                  StDv))

DataFile.close()

# Getting for maxima and minima ================================================
MaximaDate = []
MaximaData = []
MaximaIndx = []

MinimaDate = []
MinimaData = []
MinimaIndx = []

ToDrop = []

Lookahead = 20  # 10
Delta = 1

MinData = inf
MinPos = nan

MaxData = -inf
MaxPos = nan

for (Indx, (Date, Data)) in enumerate(zip(smoothDate[:-Lookahead], smoothData[:-Lookahead])):
    if Data > MaxData:
        MaxDate = Date
        MaxData = Data
        MaxPos = Indx

    if Data < MinData:
        MinDate = Date
        MinData = Data
        MinPos = Indx

    if Data < MaxData-Delta and MaxData != inf:
        if smoothData[Indx:Indx+Lookahead].max() < MaxData:
            MaximaDate.append(MaxDate)
            MaximaData.append(MaxData)
            MaximaIndx.append(MaxPos)
            ToDrop.append(True)

            MaxData = inf
            MinData = inf

            if Indx+Lookahead >= registryLength:
                break
            continue

    if Data > MinData+Delta and MinData != -inf:
        if smoothData[Indx:Indx+Lookahead].min() > MinData:
            MinimaDate.append(MinDate)
            MinimaData.append(MinData)
            MinimaIndx.append(MinPos)
            ToDrop.append(False)

            MaxData = -inf
            MinData = -inf

            if Indx+Lookahead >= registryLength:
                break

if ToDrop[0]:
    MaximaDate.pop(0)
    MaximaData.pop(0)
    MaximaIndx.pop(0)
else:
    MinimaDate.pop(0)
    MinimaData.pop(0)
    MinimaIndx.pop(0)

MaximaDate = asarray(MaximaDate)
MaximaData = asarray(MaximaData)
MaximaIndx = asarray(MaximaIndx)

MinimaDate = asarray(MinimaDate)
MinimaData = asarray(MinimaData)
MinimaIndx = asarray(MinimaIndx)

# Deleting false-positive in peaks detection manually ==========================
Fig = pylab.figure()
Grid = GridSpec(1, 1)

Axes = Fig.add_subplot(Grid[0, 0])

xLimL = 0
xLimU = len(smoothData)
xTicks = linspace(xLimL, xLimU, 16)
xTicksLabels = [r"${0:.2f}$".format(Tick) for Tick in xTicks]
xLabel = r"$\mathrm{Time\ (months\ since\ 01/01/1818)}$"
Axes.set_xlabel(xLabel, fontproperties=FontLabel)
Axes.set_xlim(xLimL, xLimU)
Axes.set_xticks(xTicks)
Axes.set_xticklabels(xTicksLabels, fontproperties=FontTicks)

yLimL = 0
yLimU = (int(max(monthlyData)/100)+1)*100
yTicks = range(yLimL, yLimU+50, 50)
yTicksLabels = [r"${0:}$".format(Tick) for Tick in yTicks]
yLabel = r"$\widetilde{\mathcal{R}}$"
Axes.set_ylabel(yLabel, fontproperties=FontLabel)
Axes.set_ylim(yLimL, yLimU)
Axes.set_yticks(yTicks)
Axes.set_yticklabels(yTicksLabels, fontproperties=FontTicks)

Title = r"$\mathrm{Pick\ on\ false-positive\ maxima\ or\ minima\ to\ delete\ them}$"
Axes.set_title(Title, fontproperties=FontTitle)

Axes.grid()

xData = range(len(smoothData))
yData = smoothData

SmoothGraph, = pylab.plot(xData,
                          yData,
                          linewidth=1.25,
                          antialiased=True,
                          label=r"$\widetilde{\mathcal{R}}$")

xData = MaximaIndx
yData = MaximaData

MaximaGraph, = pylab.plot(xData,
                          yData,
                          linewidth=0.0,
                          marker='^',
                          markersize=9.0,
                          markeredgewidth=0.0,
                          picker=5,
                          antialiased=True,
                          label=r"$\widetilde{\mathcal{R}}_\mathrm{m\acute{a}x}$")

xData = MinimaIndx
yData = MinimaData

MinimaGraph, = pylab.plot(xData,
                          yData,
                          linewidth=0.0,
                          marker='v',
                          markersize=9.0,
                          markeredgewidth=0.0,
                          picker=5,
                          antialiased=True,
                          label=r"$\widetilde{\mathcal{R}}_\mathrm{m\acute{i}n}$")

pylab.legend(prop=FontLegend, numpoints=1)

Backend = get_backend()
Handler = get_current_fig_manager()

if Backend == 'WXAgg':
    Handler.frame.Maximize(True)
elif Backend == 'TkAgg':
    Handler.resize(*Handler.window.maxsize())
elif Backend == 'Qt4Agg':
    Handler.window.showMaximized()

Fig.tight_layout(rect=[0, 0.1, 1, 0.9])
pylab.draw()

MaxMask = arange(len(MaximaData))
MinMask = arange(len(MinimaData))


def on_pick(event):
    global MaxMask
    global MinMask

    thisline = event.artist
    xData, yData = thisline.get_data()
    Indx = event.ind

    xData = delete(xData, Indx, 0)
    yData = delete(yData, Indx, 0)

    thisline.set_data(xData, yData)
    pylab.draw()

    if thisline == MaximaGraph:
        MaxMask = delete(MaxMask, Indx, 0)

    if thisline == MinimaGraph:
        MinMask = delete(MinMask, Indx, 0)

cid = Fig.canvas.mpl_connect('pick_event', on_pick)

axcolor = 'lightgoldenrodyellow'
resetax = pylab.axes([0.85, 0.10, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    global MaxMask
    global MinMask

    MaxMask = arange(len(MaximaData))
    MinMask = arange(len(MinimaData))

    MaximaGraph.set_data(MaximaIndx, MaximaData)
    MinimaGraph.set_data(MinimaIndx, MinimaData)

button.on_clicked(reset)

pylab.show()

MaximaDate = MaximaDate[MaxMask]
MaximaData = MaximaData[MaxMask]
MaximaIndx = MaximaIndx[MaxMask]

MinimaDate = MinimaDate[MinMask]
MinimaData = MinimaData[MinMask]
MinimaIndx = MinimaIndx[MinMask]

FileName = PrefixID + "maxima_and_minima.dat"
DataFile = open(FileName, "w")
DataFile.write("# Year   Month     min SSN     " +
               "  Year   Month     max SSN\n")

DataStr = "  {0:0>4d}      {1:0>2d}      {2:6.2f}{3:}"

for Indx in range(max((len(MinimaData), len(MaximaData)))):
    try:
        DataFile.write(DataStr.format(MinimaDate[Indx].year,
                                      MinimaDate[Indx].month,
                                      MinimaData[Indx],
                                      "     "))
    except IndexError:
        DataFile.write("{0:26}{1:}".format(" ", "     "))

    try:
        DataFile.write(DataStr.format(MaximaDate[Indx].year,
                                      MaximaDate[Indx].month,
                                      MaximaData[Indx],
                                      "\n"))
    except IndexError:
        DataFile.write("{0:26}{1:}".format(" ", "\n"))

DataFile.close()

# Time series decomposition in cycles between consecutive minima ===============
numCycles = len(MinimaIndx) + 1
lenCycles = diff(concatenate(([0], MinimaIndx)))
maxLenCyc = max(lenCycles)

CyclesDate = ones((numCycles, maxLenCyc), dtype=object)*nan
CyclesData = ones((numCycles, maxLenCyc))*nan
CyclesStDv = ones((numCycles, maxLenCyc))*nan
for i in range(numCycles):
    if i == 0:
        Aux = len(smoothData[:MinimaIndx[i]])
        CyclesDate[i][:lenCycles[i]] = smoothDate[:MinimaIndx[i]]
        CyclesData[i][:lenCycles[i]] = smoothData[:MinimaIndx[i]]
        CyclesStDv[i][:lenCycles[i]] = smoothStDv[:MinimaIndx[i]]
        continue

    if i == numCycles - 1:
        Aux = len(smoothData[MinimaIndx[i-1]:])
        lenCycles = append(lenCycles, Aux)
        CyclesDate[i][:Aux] = smoothDate[MinimaIndx[i-1]:]
        CyclesData[i][:Aux] = smoothData[MinimaIndx[i-1]:]
        CyclesStDv[i][:Aux] = smoothStDv[MinimaIndx[i-1]:]
        continue

    CyclesDate[i][:lenCycles[i]] = smoothDate[MinimaIndx[i-1]:MinimaIndx[i]]
    CyclesData[i][:lenCycles[i]] = smoothData[MinimaIndx[i-1]:MinimaIndx[i]]
    CyclesStDv[i][:lenCycles[i]] = smoothStDv[MinimaIndx[i-1]:MinimaIndx[i]]

FileName = PrefixID + "cycles.dat"
DataFile = open(FileName, "w")
DataFile.write("# ")

for i in range(len(CyclesData)):
    DataStr = "{0:0>2d}th Cycle".format(i+1)
    DataFile.write("{0:^26}".format(DataStr))

    if i != len(CyclesData)-1:
        DataFile.write("       ")

DataFile.write("\n")
DataFile.write("# ")

for i in range(len(CyclesData)):
    DataFile.write("Year   Month    smooth SSN")

    if i != len(CyclesData)-1:
        DataFile.write("       ")

DataFile.write("\n")

DataStr = "  {0:0>4d}      {1:0>2d}        {2:6.2f}"
EmptStr = "  {0: >4s}      {1: >2s}        {2: >6s}"

for i in range(maxLenCyc):
    for j in range(len(CyclesData)):
        if isnan(CyclesData[j][i]):
            DataFile.write(EmptStr.format("----", "--", "-.--"))
        else:
            DataFile.write(DataStr.format(CyclesDate[j][i].year,
                                          CyclesDate[j][i].month,
                                          CyclesData[j][i]))

        if j != len(CyclesData)-1:
            DataFile.write("     ")

    DataFile.write("\n")

DataFile.close()

# Selecting ongoing cycle ======================================================
OnGoingDate = CyclesDate[-1]
OnGoingData = CyclesData[-1]
OnGoingStDv = CyclesStDv[-1]

# Select cycles to make the forecast ===========================================
Fig = pylab.figure()
Grid = GridSpec(1, 1)

Axes = Fig.add_subplot(Grid[0, 0])

xLimL = smoothDate[0]
xLimU = smoothDate[-1]
xTicks = [xLimL + timedelta(i*(xLimU-xLimL).days/10) for i in range(10)] + \
         [xLimU]
xTicksLabels = [r"${0:0>2}/{1:>4}$".format(x.month, x.year) for x in xTicks]
xLabel = r"$\mathrm{Date}$"
Axes.set_xlabel(xLabel, fontproperties=FontLabel)
Axes.set_xlim(xLimL, xLimU)
Axes.set_xticks(xTicks)
Axes.set_xticklabels(xTicksLabels, fontproperties=FontTicks)

yLimL = 0
yLimU = (int(max(monthlyData)/100)+1)*100
yTicks = range(yLimL, yLimU+50, 50)
yTicksLabels = [r"${0:}$".format(Tick) for Tick in yTicks]
yLabel = r"$\widetilde{\mathcal{R}}$"
Axes.set_ylabel(yLabel, fontproperties=FontLabel)
Axes.set_ylim(yLimL, yLimU)
Axes.set_yticks(yTicks)
Axes.set_yticklabels(yTicksLabels, fontproperties=FontTicks)

Title = r"$\mathrm{Pick\ on\ cycles\ to\ use\ them\ in\ forecast\ process}$"
Axes.set_title(Title, fontproperties=FontTitle)

Axes.grid()

CyclesGraph = []
GraphColor = []

for Indx in range(numCycles):
    xData = CyclesDate[Indx, :lenCycles[Indx]]
    yData = CyclesData[Indx, :lenCycles[Indx]]

    Graph, = pylab.plot(xData,
                        yData,
                        linewidth=2.0,
                        picker=5,
                        antialiased=True,
                        )

    GraphColor.append(Graph.get_color())
    Graph.set_color('k')

    # if Indx != numCycles-1:
    CyclesGraph.append(Graph)

xData = MinimaIndx
yData = MinimaData

MinimaGraph, = pylab.plot(xData,
                          yData,
                          linewidth=0.0,
                          marker='v',
                          markersize=9.0,
                          markeredgewidth=0.0,
                          antialiased=True,
                          label=r"$\widetilde{\mathcal{R}}_\mathrm{m\acute{i}n}$")

Backend = get_backend()
Handler = get_current_fig_manager()

if Backend == 'WXAgg':
    Handler.frame.Maximize(True)
elif Backend == 'TkAgg':
    Handler.resize(*Handler.window.maxsize())
elif Backend == 'Qt4Agg':
    Handler.window.showMaximized()

Fig.tight_layout(rect=[0, 0.1, 1, 0.9])

pylab.draw()

CycleMask = []


def on_pick(event):
    global CycleMask

    thisline = event.artist

    if thisline in CyclesGraph:
        Indx = CyclesGraph.index(thisline)

        if Indx not in CycleMask:
            thisline.set_color(GraphColor[Indx])
            thisline.set_linewidth(4)
            CycleMask.append(Indx)
        else:
            thisline.set_color('k')
            thisline.set_linewidth(2)
            CycleMask.pop(CycleMask.index(Indx))

        pylab.draw()

cid = Fig.canvas.mpl_connect('pick_event', on_pick)

axcolor = 'lightgoldenrodyellow'
resetax = pylab.axes([0.85, 0.10, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    global CycleMask

    CycleMask = []

    for Graph in CyclesGraph:
        Graph.set_linewidth(2)
        Graph.set_color('k')

button.on_clicked(reset)

pylab.show()

CycleMask.sort()
CycleMask = asarray(CycleMask)

CyclesDate = CyclesDate[CycleMask]
CyclesData = CyclesData[CycleMask]
CyclesStDv = CyclesStDv[CycleMask]

CyclesData[isnan(CyclesData)] = 0.0
CyclesStDv[isnan(CyclesStDv)] = 0.0

# Constructing the mean cycle and obtaining the regression coefficients ========
SSNAverageCycleData = mean(CyclesData, 0)
SSNAverageCycleStDv = std(CyclesData, 0)

Numerator = (CyclesData - SSNAverageCycleData)
Numerator = Numerator*concatenate((Numerator[:, 1:], zeros((len(Numerator), 1))), axis=1)
Numerator = sum(Numerator, 0)
Denominator = (CyclesData - SSNAverageCycleData)**2
Denominator = sum(Denominator, 0)

SSNAverageCycleRgCf = Numerator/Denominator

FileName = PrefixID + "mean_cycle.dat"
DataFile = open(FileName, "w")
DataFile.write("#     Time    mean SSN     std SSN    Regresion\n")
DataFile.write("# (months)       cycle       cycle   Coeficient\n")

DataStr = "       {0:>3d}    {1:>8.2f}    {2:>8.2f}     {3:>8.2f}\n"

for (Indx, (Data, StDv, RgCf)) in enumerate(zip(SSNAverageCycleData,
                                                SSNAverageCycleStDv,
                                                SSNAverageCycleRgCf)):
    DataFile.write(DataStr.format(Indx,
                                  Data,
                                  StDv,
                                  RgCf))

DataFile.close()

# Forecasting Ongoing cycle ====================================================
LastIndx = where(isnan(OnGoingData))[0][0]

LastMeasureDate = OnGoingDate[LastIndx-1]
LastMeasureData = OnGoingData[LastIndx-1]

for Indx in range(LastIndx-1, maxLenCyc-1):
    AuxMonth = LastMeasureDate.month + Indx - LastIndx + 1

    OnGoingDate[Indx+1] = datetime(LastMeasureDate.year + int(AuxMonth/12),
                                   mod(AuxMonth, 12) + 1,
                                   1)

    OnGoingData[Indx+1] = (SSNAverageCycleData[Indx+1] +
                           SSNAverageCycleRgCf[Indx] * (OnGoingData[Indx] -
                           SSNAverageCycleData[Indx]))

SSNForecastCycleDate = OnGoingDate[LastIndx-1:]
SSNForecastCycleData = OnGoingData[LastIndx-1:]

FileName = PrefixID + "forecast.dat"
DataFile = open(FileName, "w")
DataFile.write("# Year   Month    mean SSN\n")

DataStr = "  {0:0>4d}      {1:0>2d}      {2:6.2f}\n"

for (Date, Data) in zip(SSNForecastCycleDate, SSNForecastCycleData):
    DataFile.write(DataStr.format(Date.year, Date.month, Data))

DataFile.close()
