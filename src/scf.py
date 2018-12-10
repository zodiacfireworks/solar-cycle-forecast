from collections import OrderedDict
from datetime import datetime
from numpy import arange, array, concatenate, delete, inf, isnan, nan, ones, \
                  sqrt, where

from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib import get_backend
import argparse
import pandas
import pylab
import os


# ============================================================================ #
# Function definitions
# ============================================================================ #
# File reading *************************************************************** #
def readFile(filePath):
    try:
        fileHandler = open(filePath)
    except Exception as msg:
        raise Exception(msg)

    fileContent = fileHandler.read().splitlines()
    fileHandler.close()

    return fileContent


# Get data from file raw content ********************************************* #
def getData(rawTextData):
    try:
        fileData = zip(
            (
                "date",
                "ssn"
            ),
            zip(
                *map(
                    lambda x: (
                        datetime.strptime(
                            "-".join(x[0:3]),
                            "%Y-%m-%d"
                        ),
                        float(x[4])
                    ),
                    map(
                        lambda x:
                            x.split(),
                        rawTextData
                    )
                )
            )
        )
    except Exception as msg:
        raise Exception(msg)

    return dict(fileData)


# Time series builder ******************************************************** #
def getTimeSeries(rawData):
    timeSeries = pandas.Series(
        rawData["ssn"],
        index=rawData["date"],
        dtype=float
    )

    timeSeries[timeSeries == -1] = nan

    return timeSeries


# Build monthly average time series ****************************************** #
def getMonthlyAverage(timeSeries, skip=[-1, 0], roundTo=1, replaceNan=0):
    monthlyTimeSeries = timeSeries.copy()

    for i in skip:
        monthlyTimeSeries[monthlyTimeSeries == i] = nan

    monthlyTimeSeries = monthlyTimeSeries.resample(
        rule='1M',
        how='mean',
        label='left',
        loffset='1D'
    )

    monthlyTimeSeries = monthlyTimeSeries.round(roundTo)
    monthlyTimeSeries[isnan(monthlyTimeSeries)] = replaceNan

    return monthlyTimeSeries


# Build smoothed average time series ***************************************** #
def getSmoothedAverage(timeSeries, weight=None, window=13, roundTo=1):
    if weight is None:
        weight = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

    smoothedAverage = pandas.rolling_apply(
        timeSeries,
        window,
        lambda x, w: (x*w).sum()/w.sum(),
        args=(array(weight), ),
        min_periods=window,
        center=True
    )

    smoothedAverage = smoothedAverage.round(roundTo)

    return smoothedAverage


# Build smoothed standard deviation time series ****************************** #
def getSmoothedStd(timeSeries, weight=None, window=13, roundTo=1):
    if weight is None:
        weight = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

    smoothedStd = pandas.rolling_apply(
        timeSeries,
        window,
        lambda x, w:
            sqrt(
                (w*(x - (x*w).sum()/w.sum())*(x - (x*w).sum()/w.sum())).sum()/(((w != 0).sum() - 1)*w.sum()/(w != 0).sum())
            ),
        args=(array(weight), ),
        min_periods=window,
        center=True
    )

    smoothedStd = smoothedStd.round(roundTo)

    return smoothedStd


# Build extrema time series ************************************************** #
def getExtrema(timeSeries, delta=5, lookahead=5):
    maxDate = []
    maxData = []

    minDate = []
    minData = []

    toDrop = []

    registryLength = len(timeSeries)

    minima = inf
    maxima = -inf

    for (Indx, Data) in enumerate(timeSeries[:-lookahead]):
        if Data > maxima:
            mdate = timeSeriesSmoth.index[Indx]
            maxima = Data

        if Data < minima:
            mdate = timeSeriesSmoth.index[Indx]
            minima = Data

        if Data < maxima-delta and maxima != inf:
            if timeSeries[Indx:Indx+lookahead].max() < maxima:
                maxDate.append(mdate)
                maxData.append(maxima)
                toDrop.append(True)

                maxima = inf
                minima = inf

                if Indx+lookahead >= registryLength:
                    break
                continue

        if Data > minima+delta and minima != -inf:
            if timeSeries[Indx:Indx+lookahead].min() > minima:
                minDate.append(mdate)
                minData.append(minima)
                toDrop.append(False)

                maxima = -inf
                minima = -inf

                if Indx+lookahead >= registryLength:
                    break

    if toDrop[0]:
        maxDate.pop(0)
        maxData.pop(0)
    else:
        minDate.pop(0)
        minData.pop(0)

    return (pandas.Series(maxData, index=maxDate),
            pandas.Series(minData, index=minDate))


# Checking extrema time series *********************************************** #
def checkExtrema(timeSeries, maximaTimeSeries, minimaTimeSeries):
    global MaxMask
    global MinMask

    pylab.rc('text', usetex=True)
    FontTitle = FontProperties(size=22)
    FontLabel = FontProperties(size=18)
    FontTicks = FontProperties(size=16)
    FontLegend = FontProperties(size=14)

    Fig = pylab.figure()
    Grid = GridSpec(1, 1)
    Axes = Fig.add_subplot(Grid[0, 0])

    xLimL = timeSeries.index[0].to_datetime()
    xLimU = timeSeries.index[-1].to_datetime()
    xTicks = [xLimL + (i/10.0)*(xLimU - xLimL) for i in range(11)]
    xTicksLabels = [r"${0:%Y-%m}$".format(Tick) for Tick in xTicks]
    xLabel = r"$\mathrm{Time\ (months\ since\ 01/01/1818)}$"
    Axes.set_xlabel(xLabel, fontproperties=FontLabel)
    Axes.set_xlim(xLimL, xLimU)
    Axes.set_xticks(xTicks)
    Axes.set_xticklabels(xTicksLabels, fontproperties=FontTicks)

    yLimL = 0
    yLimU = (int(max(timeSeries)/100)+1)*100
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

    SmoothGraph, = pylab.plot(
        timeSeries.index,
        timeSeries,
        linewidth=1.5,
        antialiased=True,
        label=r"$\widetilde{\mathcal{R}}$")

    MaximaGraph, = pylab.plot(
        maximaTimeSeries.index,
        maximaTimeSeries,
        linewidth=0.0,
        marker='^',
        markersize=9.0,
        markeredgewidth=0.0,
        picker=5,
        antialiased=True,
        label=r"$\widetilde{\mathcal{R}}_\mathrm{m\acute{a}x}$")

    MinimaGraph, = pylab.plot(
        minimaTimeSeries.index,
        minimaTimeSeries,
        linewidth=0.0,
        marker='v',
        markersize=9.0,
        markeredgewidth=0.0,
        picker=5,
        antialiased=True,
        label=r"$\widetilde{\mathcal{R}}_\mathrm{m\acute{i}n}$")

    pylab.legend(prop=FontLegend, numpoints=1)

    Backend = get_backend()
    Handler = pylab.get_current_fig_manager()

    if Backend == 'WXAgg':
        Handler.frame.Maximize(True)
    elif Backend == 'TkAgg':
        Handler.resize(*Handler.window.maxsize())
    elif Backend == 'Qt4Agg':
        Handler.window.showMaximized()

    Fig.tight_layout(rect=[0, 0.1, 1, 0.9])
    pylab.draw()

    MaxMask = arange(len(maximaTimeSeries))
    MinMask = arange(len(minimaTimeSeries))

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

    Fig.canvas.mpl_connect('pick_event', on_pick)

    axcolor = 'lightgoldenrodyellow'
    resetax = pylab.axes([0.85, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        global MaxMask
        global MinMask

        MaxMask = arange(len(maximaTimeSeries))
        MinMask = arange(len(minimaTimeSeries))

        MaximaGraph.set_data(maximaTimeSeries.index, maximaTimeSeries.values)
        MinimaGraph.set_data(minimaTimeSeries.index, minimaTimeSeries.values)

    button.on_clicked(reset)
    pylab.show()

    return (maximaTimeSeries[MaxMask], minimaTimeSeries[MinMask])


# Getting individual solar cycles ******************************************** #
def getSolarCycles(timeSeries, stopsTimeSeries, offset=0):
    cyclesStops = list(
        map(
            lambda x:
                where(timeSeries.index == x)[0][0],
            stopsTimeSeries.index
        )
    )

    cyclesStops = [0] + cyclesStops

    solarCycles = OrderedDict({})

    for i in range(len(cyclesStops)):
        start = cyclesStops[i]
        try:
            end = cyclesStops[i+1]
        except:
            end = len(timeSeries) + 1

        label = "Cycle {0:0>2d}"

        solarCycles.update({
            label.format(i+offset):
                pandas.Series(
                    timeSeries[start:end],
                    index=timeSeries.index[start:end])
        })

    return solarCycles


# Solar Cycle Selection ****************************************************** #
def selectSolarCycles(cyclesSeries):
    pylab.rc('text', usetex=True)
    FontTitle = FontProperties(size=22)
    FontLabel = FontProperties(size=18)
    FontTicks = FontProperties(size=16)

    Fig = pylab.figure()
    Grid = GridSpec(1, 1)
    Axes = Fig.add_subplot(Grid[0, 0])

    cyclesLabel = list(cyclesSeries.keys())
    cyclesLabel.sort()

    xLimL = cyclesSeries[cyclesLabel[0]].index[0].to_datetime()
    xLimU = cyclesSeries[cyclesLabel[-1]].index[-1].to_datetime()

    xTicks = [xLimL + (i/10.0)*(xLimU - xLimL) for i in range(11)]
    xTicksLabels = [r"${0:%Y-%m}$".format(Tick) for Tick in xTicks]
    xLabel = r"$\mathrm{Date}$"
    Axes.set_xlabel(xLabel, fontproperties=FontLabel)
    Axes.set_xlim(xLimL, xLimU)
    Axes.set_xticks(xTicks)
    Axes.set_xticklabels(xTicksLabels, fontproperties=FontTicks)

    yLimL = 0
    yLimU = (int(max(map(lambda x: max(cyclesSeries[x]), cyclesSeries))/100)+1)*100
    yTicks = range(yLimL, yLimU+50, 50)
    yTicksLabels = [r"${0:}$".format(Tick) for Tick in yTicks]
    yLabel = r"$\widetilde{\mathcal{R}}$"
    Axes.set_ylabel(yLabel, fontproperties=FontLabel)
    Axes.set_ylim(yLimL, yLimU)
    Axes.set_yticks(yTicks)
    Axes.set_yticklabels(yTicksLabels, fontproperties=FontTicks)

    Title = r"$\mathrm{Pick\ on\ cycles\ to\ use\ them\ in\ forecast\ process}$"
    Axes.set_title(Title, fontproperties=FontTitle)

    cycleLines = {}

    for cycle in cyclesSeries:
        line, = pylab.plot(
            cyclesSeries[cycle].index,
            cyclesSeries[cycle],
            linewidth=2.0,
            picker=5,
            antialiased=True,
            color='k'
        )

        cycleLines.update({cycle: line})

    Backend = get_backend()
    Handler = pylab.get_current_fig_manager()

    if Backend == 'WXAgg':
        Handler.frame.Maximize(True)
    elif Backend == 'TkAgg':
        Handler.resize(*Handler.window.maxsize())
    elif Backend == 'Qt4Agg':
        Handler.window.showMaximized()

    Fig.tight_layout(rect=[0, 0.1, 1, 0.9])
    pylab.draw()

    cycleMask = []

    def on_pick(event):
        thisline = event.artist

        if thisline in cycleLines.values():
            cycleID = list(cycleLines.keys())[list(cycleLines.values()).index(thisline)]

            if cycleID in cycleMask:
                cycleMask.remove(cycleID)
                thisline.set_linewidth(2)
                thisline.set_color('k')
            else:
                cycleMask.append(cycleID)
                thisline.set_linewidth(4)
                thisline.set_color('r')

        pylab.draw()

    axcolor = 'lightgoldenrodyellow'
    resetax = pylab.axes([0.85, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        cycleMask = []

        for line in cycleLines.values():
            line.set_linewidth(2)
            line.set_color('k')

        pylab.draw()

    button.on_clicked(reset)

    Fig.canvas.mpl_connect('pick_event', on_pick)

    pylab.show()

    cycleMask.sort()

    return OrderedDict(map(lambda x: (x, cyclesSeries[x]), cycleMask))


# Normalizing solar cycle length ********************************************* #
def getNormalizeSolarCycles(cyclesSeries):
    pastCycles = list(cyclesSeries.keys())
    pastCycles.sort()
    pastCycles = pastCycles[:-1]

    cyclesLengt = max([len(cycle) for cycle in cyclesSeries.values()])

    normalizedCycles = pandas.DataFrame(
        dict(
            map(
                lambda x:
                    (x, concatenate([cyclesSeries[x].values, nan*ones(cyclesLengt - len(cyclesSeries[x]))])),
                cyclesSeries
            )
        ),
        index=range(cyclesLengt)
    )

    return normalizedCycles


# Getting ongoing normalized cycle ******************************************* #
def getNormalizeOngoingSolarCycles(ongoingCycle, normalizeTo=None):
    return pandas.Series(
        concatenate([ongoingCycle.values, nan*ones(normalizeTo - len(ongoingCycle))]),
        index=range(normalizeTo)
    )


# ============================================================================ #
# Argument parsing utilities
# ============================================================================ #
# Function to validate command line date input ******************************* #
def dateValidation(dateStr, formats=["%Y", "%Y-%m"]):
    dateObj = None
    parseErrors = 0

    for fmt in formats:
        try:
            dateObj = datetime.strptime(dateStr, fmt)
        except:
            parseErrors += 1

    if parseErrors == len(formats):
        raise ValueError(
            'Time data "{0}" does not match any format of "{1}"'.format(
                dateStr,
                ', '.join(formats)
            )
        )

    return dateObj


# Function to validate command line date input ******************************* #
class SmartFormatter(argparse.RawTextHelpFormatter):
    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if '\n' in text:
            return text.splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


# Arguments parsing definitions ********************************************** #
scfParser = argparse.ArgumentParser(
    prog='scf',
    formatter_class=SmartFormatter,
    description='Make a forecast of the solar cycle forecast using the'
                ' modified McNish-Lincol method.',
    epilog='Program developed by Martín Josemaría Vuelta Rojas at '
           'DIAST - CONIDA.\n'
           '* Github: https://github.com/ZodiacFireworks\n'
           '* e-mail: martin.vuelta@gmail.com')

scfParser.add_argument(
    '-f', '--file',
    metavar='FILE',
    type=str,
    nargs=1,
    required=True,
    help='File where is the data data about solar cycle indicator.'
         'Preferably use the files provided by sunspot SILSO.'
         '(http://sidc.oma.be/silso/datafiles).')

scfParser.add_argument(
    '-e',
    '--end-date',
    metavar='DATE',
    type=dateValidation,
    required=True,
    help='Date on which ends the prediction.'
         'This should be in the formats "YYYY-MM" or "YYYY".')


# ============================================================================ #
# Main Function
# ============================================================================ #
if __name__ == '__main__':
    # Getting command line arguments *******************************************
    scfArgs = scfParser.parse_args()
    filePath = os.path.abspath(scfArgs.file[0])
    endDate = scfArgs.end_date

    # Getting data from file ***************************************************
    fileText = readFile(filePath)
    rawData = getData(fileText)

    rawTimeSeries = getTimeSeries(rawData)

    # Processing data **********************************************************
    monthlyTimeSeries = getMonthlyAverage(
        rawTimeSeries,
        skip=[-1],
        roundTo=1,
        replaceNan=0
    )

    timeSeriesSmoth = getSmoothedAverage(
        monthlyTimeSeries,
        weight=array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]),
        window=13,
        roundTo=1
    )

    timeSeriesSmothStd = getSmoothedStd(
        monthlyTimeSeries,
        weight=array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]),
        window=13,
        roundTo=2
    )

    timeSeriesSmoth = timeSeriesSmoth[6:-6]
    timeSeriesSmothStd = timeSeriesSmothStd[6:-6]

    # Getting maxima and minima ************************************************
    timeSeriesMaxima, timeSeriesMinima = getExtrema(
        timeSeriesSmoth,
        delta=5,
        lookahead=20
    )

    # Checking Cycles **********************************************************
    # timeSeriesMaxima, timeSeriesMinima = checkExtrema(timeSeriesSmoth, timeSeriesMaxima, timeSeriesMinima)

    solarCycles = getSolarCycles(timeSeriesSmoth, timeSeriesMinima, offset=6)
    ongoingCycle = solarCycles[list(solarCycles.keys())[-1]]
    selectedCycles = selectSolarCycles(solarCycles)

    nomalizedSolarCycles = getNormalizeSolarCycles(selectedCycles)
    normalizedOngoingCycle = getNormalizeOngoingSolarCycles(ongoingCycle, normalizeTo=nomalizedSolarCycles.shape[0])

    nomalizedSolarCycles[isnan(nomalizedSolarCycles)] = 0
    meanSolarCycle = nomalizedSolarCycles.mean(axis=1, skipna=True, numeric_only=True).round(1)
    stdSolarCycle = nomalizedSolarCycles.std(axis=1, skipna=True, numeric_only=True).round(2)

    print(nomalizedSolarCycles.add(meanSolarCycle, axis=0).apply(pandas.Series.round, args=(1,)))
    # meanSolarCycle.plot(linewidth=3, label="Mean cycle")
    # (meanSolarCycle + 2*stdSolarCycle).plot(linewidth=1)
    # (meanSolarCycle - 2*stdSolarCycle).plot(linewidth=1)
    # normalizedOngoingCycle.plot(linewidth=3, label="On going cycle")
    # pylab.legend(numpoints=1)
    # pylab.show()
