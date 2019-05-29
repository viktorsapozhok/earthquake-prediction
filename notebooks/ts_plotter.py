# -*- coding: utf-8 -*-

"""
tools for plotting of dataframes with datetime index
"""

import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


class TimeSeriesPlot(object):
    def __init__(self, **kwargs):
        self.fig_size = kwargs.get('fig_size', (22, 4))
        self.n_rows = kwargs.get('n_rows', None)
        self.n_cols = kwargs.get('n_cols', None)
        self.hspace = kwargs.get('hspace', 0.2)
        self.wspace = kwargs.get('wspace', 0.2)
        self.type = kwargs.get('type', 'line')
        self.group_vars = kwargs.get('group_vars', [])
        self.show = kwargs.get('show', True)

    def plot(self, data, **kwargs):
        self.type = kwargs.get('type', self.type)
        self.fig = kwargs.get('fig', None)
        self.id = kwargs.get('id', 1)
        self.close_fig = kwargs.get('close_fig', True)
        self.x_var = kwargs.get('x_var', 'index')
        self.y_vars = kwargs.get('y_vars', [])
        self.date_var = kwargs.get('date_var', None)
        self.dates_fmt = kwargs.get('dates_fmt', 'auto')
        self.x_ticks_step = kwargs.get('x_ticks_step', None)
        self.lw = kwargs.get('line_width', 0.75)
        self.marker = kwargs.get('marker', None)
        self.ms = kwargs.get('marker_size', 10)
        self.out_var = kwargs.get('out_var', None)
        self.out_alpha = kwargs.get('out_alpha', 0.15)
        self.bins = kwargs.get('bins', 10)
        self.lags = kwargs.get('lags', range(24))
        self.color = kwargs.get('color', None)
        self.alpha = kwargs.get('alpha', 1)
        self.fs = kwargs.get('font_size', 12)
        self.title = kwargs.get('title', None)
        self.title_inside = kwargs.get('title_inside', False)
        self.xlim = kwargs.get('xlim', None)
        self.ylim = kwargs.get('ylim', None)
        self.legend = kwargs.get('legend', None)
        self.xlabel = kwargs.get('xlabel', '')
        self.ylabel = kwargs.get('ylabel', '')
        # annotations
        self.ant_var = kwargs.get('ant_var', None)
        # boxcox lambda upper and lower bounds
        self.la = kwargs.get('la', None)
        self.lb = kwargs.get('lb', None)
        # fill area between lines (valid only self.type=line)
        self.fill_between = kwargs.get('fill_between', [])

        fig = plt.figure(figsize=self.fig_size) if self.fig is None else self.fig

        if len(self.group_vars) > 0:
            groups = self.__init_groups(data)

            # if subplots grid is not initialized then get it from groups count
            if self.n_rows is None or self.n_cols is None:
                self.__init_subplots_grid(len(groups))

            # make plot for every group
            for i, group in enumerate(groups):
                condition = data[self.group_vars[0]].notnull()

                for j, val in enumerate(group):
                    condition &= data[self.group_vars[j]] == val

                data_group = data[condition]
                dates = self.__init_dates(data_group)
                title = self.__init_title(data_group, group)

                ax = plt.subplot(self.n_rows, self.n_cols, i + 1)
                ax = self.__make_plot(ax, data_group)
                self.__set_props(ax, dates=dates, title=title)
        else:
            dates = self.__init_dates(data)
            if self.n_rows is None:
                ax = plt.subplot(1, 1, 1)
            else:
                ax = plt.subplot(self.n_rows, self.n_cols, self.id)
            ax = self.__make_plot(ax, data)
            self.__set_props(ax, dates=dates, title=self.title)

        if self.show:
            plt.show()

        if self.close_fig:
            fig.subplots_adjust(
                left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=self.hspace, wspace=self.wspace)
            plt.close(fig)
        return fig

    def __make_plot(self, ax, data):
        if self.type == 'line':
            x = None if self.x_var == 'index' else data[self.x_var]
            outliers = None if self.out_var is None else data[self.out_var]
            annotations = None if self.ant_var is None else data[self.ant_var]

            ax = self.__line_plot(
                ax, data[self.y_vars], outliers=outliers, out_alpha=self.out_alpha,
                lw=self.lw, marker=self.marker, x=x, annotations=annotations)
        elif self.type == 'scatter':
            ax = self.__scatter_plot(ax, data[self.x_var], data[self.y_vars],
                                     marker=self.marker, ms=self.ms, alpha=self.alpha)
        elif self.type == 'long_line':
            raise NotImplementedError()
        elif self.type == 'step':
            raise NotImplementedError()
        elif self.type == 'hist':
            ax = self.__histogram(ax, data[self.y_vars[0]], bins=self.bins, c=self.color, alpha=self.alpha)
        elif self.type == 'acf':
            ax = self.__acf(ax, data[self.y_vars[0]], lags=self.lags, ms=self.ms, c=self.color)
        elif self.type == 'pacf':
            ax = self.__pacf(ax, data[self.y_vars[0]], lags=self.lags, ms=self.ms, c=self.color)
        elif self.type == 'boxcox':
            ax = self.__boxcox_normplot(ax, data[self.y_vars[0]], self.la, self.lb, marker=self.marker,
                                        ms=self.ms, alpha=self.alpha)
        elif self.type == 'heatmap':
            ax = self.__corr_map(ax, data[self.y_vars])
        elif self.type == 'kde':
            ax = self.__kde_plot(ax, data[self.y_vars[0]], data[self.x_var])

        return ax

    def __line_plot(self, ax, data, **kwargs):
        # line width
        lw = kwargs.get('lw', 1.5)
        # line color
        lc = kwargs.get('lc', None)
        # if None then use date index on x-axis
        x = kwargs.get('x', None)
        marker = kwargs.get('marker', None)
        outliers = kwargs.get('outliers', None)
        out_alpha = kwargs.get('out_alpha', 1)
        annotations = kwargs.get('annotations', None)

        marker_size = lw * 10
        color = self.__get_color(1) if lc is None else self.__get_color(-1, color=lc)

        if x is None:
            x = data.index.to_pydatetime()

        if isinstance(data, pd.Series):
            y = data.values
            ax.plot(x, y, lw=lw, c=color, marker=marker, markersize=marker_size)
        else:
            n_lines = data.shape[1]
            if n_lines == 1:
                y = data.iloc[:, 0]
                ax.plot(x, y, lw=lw, c=color, marker=marker, markersize=marker_size)

                if annotations is not None:
                    for i, ano in enumerate(annotations):
                        ax.annotate(ano, xy=(x[i], y[i]), textcoords='data', color='b', alpha=0.6, size=self.fs)
            else:
                y = data.iloc[:, :n_lines]
                for j in range(0, n_lines):
                    ax.plot(x, data.iloc[:, j], lw=lw,
                            c=self.__get_color(j), marker=marker, markersize=marker_size)
                # fill area between lines
                if len(self.fill_between) == 2:
                    ax.fill_between(x, data[self.fill_between[0]], data[self.fill_between[1]],
                                    facecolor='gray', alpha=0.1)

        # highlight outliers by vertical lines
        if outliers is not None:
            ax = self.__add_vertical_lines(
                ax, x[outliers == 1], y, lw, self.__get_color(-1, color='g'), out_alpha)
        return ax

    def __histogram(self, ax, data, **kwargs):
        # color
        c = kwargs.get('c', None)
        bins = kwargs.get('bins', 10)
        alpha = kwargs.get('alpha', 0.75)

        color = self.__get_color(1) if c is None else self.__get_color(-1, color=c)

        ax.hist(data, bins=bins, color=color, alpha=alpha)
        return ax

    def __step_plot(self, ax, x, y, **kwargs):
        # line width
        lw = kwargs.get('lw', 1.5)
        # line color
        lc = kwargs.get('lc', None)

        color = self.__get_color(1) if lc is None else self.__get_color(-1, color=lc)
        ax.step(x, y, lw=lw, c=color)
        return ax

    def __scatter_plot(self, ax, x, y, **kwargs):
        marker = kwargs.get('marker', None)
        # marker size
        ms = kwargs.get('ms', 50)
        # marker color
        mc = kwargs.get('mc', None)
        alpha = kwargs.get('alpha', 0.75)

        color = self.__get_color(1) if mc is None else self.__get_color(-1, color=mc)
        ax.scatter(x, y, marker=marker, s=ms, c=[color], alpha=alpha)
        return ax

    def __acf(self, ax, y, lags, **kwargs):
        ms = kwargs.get('ms', 50)
        mc = kwargs.get('mc', None)

        color = self.__get_color(1) if mc is None else self.__get_color(-1, color=mc)
        plot_acf(y, ax=ax, lags=lags, title=None, markersize=ms, c=color)

        return ax

    def __pacf(self, ax, y, lags, **kwargs):
        ms = kwargs.get('ms', 50)
        mc = kwargs.get('mc', None)

        color = self.__get_color(1) if mc is None else self.__get_color(-1, color=mc)
        plot_pacf(y, ax=ax, lags=lags, title=None, markersize=ms, c=color)

        return ax

    def __boxcox_normplot(self, ax, y, la, lb, **kwargs):
        marker = kwargs.get('marker', None)
        ms = kwargs.get('ms', 50)
        alpha = kwargs.get('alpha', 0.75)

        # convert to strictly positives
        if y.min() <= 0:
            vals = y.values + abs(y.min()) + 1
        else:
            vals = y.values

        prob = stats.boxcox_normplot(vals, la, lb, plot=ax)
        ax.plot(prob[0], prob[1], color='b', marker=marker, markersize=ms, alpha=alpha)

        xt, max_log, interval = stats.boxcox(vals, alpha=0.05)
        ax.axvline(max_log, color='r')

        return ax

    def __corr_map(self, ax, corr):
        sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
        return ax

    def __kde_plot(self, ax, y, x):
        sns.kdeplot(x, y, cmap='Blues', shade=True, shade_lowest=True, ax=ax)
        return ax

    def __init_groups(self, data):
        values = [list(data[var].unique()) for var in self.group_vars]
        groups = [element for element in itertools.product(*values)]
        return groups

    def __init_dates(self, data):
        if self.date_var is None:
            return None
        elif self.date_var == 'index':
            return data.index
        return data[self.date_var].values

    def __init_title(self, data, group):
        if self.title is not None:
            title = self.title
        else:
            title = '( '
            for item in group:
                title += str(item) + ' '
            title += ')'
        return title

    def __init_subplots_grid(self, n_plots):
        self.n_rows = np.ceil(n_plots / 5)
        self.n_cols = min(n_plots, 5)

    def __set_props(self, ax, **kwargs):
        dates = kwargs.get('dates', None)
        title = kwargs.get('title', None)

        if self.type not in ['heatmap']:
            plt.tick_params(axis="both", which="both", bottom=False, top=False,
                            labelbottom=True, left=False, right=False, labelleft=True)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

        if dates is not None:
            if len(dates) > 72:
                if self.x_ticks_step is None:
                    step = max(int(np.round(len(dates) / (30 * 24)) * 24), 24)
                else:
                    step = self.x_ticks_step
                ax.set_xticks(dates[::step])
                if self.dates_fmt == 'auto':
                    ax.xaxis.set_major_formatter(plt_dates.DateFormatter('%d-%b'))
            else:
                intv = len(dates) // 6
                ax.xaxis.set_major_locator(plt_dates.HourLocator(byhour=range(0, len(dates), intv)))
                if self.dates_fmt == 'auto':
                    ax.xaxis.set_major_formatter(plt_dates.DateFormatter('%d-%b %H'))
            if self.dates_fmt != 'auto':
                ax.xaxis.set_major_formatter(plt_dates.DateFormatter(self.dates_fmt))
            if self.dates_fmt != '%H':
                plt.setp(ax.get_xticklabels(), rotation=30)
            ax.set_xlim([dates[0], dates[-1]])
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)

        if self.type not in ['heatmap']:
            ax.grid(c='b', ls='--', lw='0.5', alpha=0.3)

        plt.setp(ax.spines.values(), color='b', lw='0.5', alpha=0.3)
        plt.setp(ax.get_xticklabels(), color='b', alpha=0.6)
        plt.setp(ax.get_yticklabels(), color='b', alpha=0.6)
        plt.setp(ax.xaxis.get_label(), color='b', alpha=0.6)
        plt.setp(ax.yaxis.get_label(), color='b', alpha=0.6)

        if self.fs is not None:
            plt.setp(ax.get_xticklabels(), fontsize=self.fs)
            plt.setp(ax.get_yticklabels(), fontsize=self.fs)
            plt.setp(ax.xaxis.get_label(), fontsize=self.fs)
            plt.setp(ax.yaxis.get_label(), fontsize=self.fs)
        if title is not None:
            if self.title_inside:
                if self.fs is None:
                    ax.text(.01, .9, title, horizontalalignment='left',
                            transform=ax.transAxes, color='b', alpha=0.6)
                else:
                    ax.text(.01, .9, title, horizontalalignment='left',
                            transform=ax.transAxes, color='b', alpha=0.6, fontsize=self.fs)
            else:
                ax.set_title(title, loc='left', pad=10, color='b', alpha=0.6, fontsize=self.fs)
        if self.legend is not None:
            if self.fs is None:
                leg = ax.legend(self.legend, loc='best', edgecolor='b', framealpha=0.3)
            else:
                leg = ax.legend(self.legend, loc='best', edgecolor='b', framealpha=0.3, fontsize=self.fs)
            for text in leg.get_texts():
                text.set_color('b')
                text.set_alpha(0.6)
            leg.get_frame().set_linewidth(0.5)
        return ax

    @staticmethod
    def __add_vertical_lines(ax, x, y, lw, c, alpha):
        y_min = y.min().min()
        y_max = y.max().max()
        for index in x:
            ax.plot([index, index], [y_min, y_max], lw=lw, c=c, alpha=alpha)
        return ax

    @staticmethod
    def __get_color(index, **kwargs):
        color = kwargs.get('color', None)

        t = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (23, 190, 207),
             (255, 187, 120), (255, 127, 14), (152, 223, 138), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (174, 199, 232), (158, 218, 229)]
        if color == 'r':
            index = 0
        elif color == 'b':
            index = 1
        elif color == 'g':
            index = 14
        index = np.mod(index, len(t)) + 1
        return (t[index - 1][0] / 255., t[index - 1][1] / 255., t[index - 1][2] / 255.)

    @staticmethod
    def get_mae(x, y):
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        if sum(~pd.isnull(x)) == 0 or sum(~pd.isnull(y)) == 0:
            return np.nan
        try:
            mae = np.nanmean(np.abs(x - y))
        except TypeError:
            mae = np.nan
        return mae

    @staticmethod
    def get_mape(x, y, scaler):
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        if sum(~pd.isnull(x)) == 0 or sum(~pd.isnull(y)) == 0:
            return np.nan
        try:
            if scaler == 'max':
                mape = np.nanmean(np.abs(x - y)) / np.nanmax(x)
            else:
                mape = np.nanmean(np.abs(x - y)) / np.nanmean(x)
        except TypeError:
            mape = np.nan
        return mape * 100


