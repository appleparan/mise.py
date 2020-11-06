''' Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve stl_bokeh.py 

at your command prompt. Then navigate to the URL

    http://localhost:5006/stl_bokeh

if you run bokeh serve in WSL, create firewall rule for TCP 5006 port
and serve with address 0.0.0.0

    bokeh serve stl_bokeh.py --address 0.0.0.0

in your browser.

'''
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from pytz import timezone

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Select, CheckboxButtonGroup
from bokeh.models import Range1d, DateRangeSlider, RangeSlider
from bokeh.plotting import figure
from bokeh.palettes import RdBu4, Dark2_3

import glob
seoultz = timezone('Asia/Seoul')


def get_dataset(target="PM10", station_name="종로구"):
    fname = target + "_" + station_name + "_sea_decomp" + ".csv"

    dirpath = Path("mean_decomp/decomposition") / station_name / target / fname
    filepath = Path("mean_decomp/decomposition") / \
        station_name / target / fname

    raw_df = pd.read_csv(filepath,
                         index_col=[0],
                         parse_dates=[0])

    df = raw_df.copy()
    df.sort_index(inplace=True)

    return ColumnDataSource(data=df)

def get_yminmax(source):
    ymin_yr = min(source.data['year_res'])
    ymax_yr = max(source.data['year_res'])
    ymin_ys = min(source.data['year_sea'])
    ymax_ys = max(source.data['year_sea'])
    ymin_dr = min(source.data['day_res'])
    ymax_dr = max(source.data['day_res'])
    ymin_ds = min(source.data['day_sea'])
    ymax_ds = max(source.data['day_sea'])
    ymin_wr = min(source.data['week_res'])
    ymax_wr = max(source.data['week_res'])
    ymin_ws = min(source.data['week_sea'])
    ymax_ws = max(source.data['week_sea'])
    ymin = min([ymin_yr, ymin_ys, ymin_dr, ymin_ds])
    ymax = max([ymax_yr, ymax_ys, ymax_dr, ymax_ds])

    return ymin, ymax


def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    # https://stackoverflow.com/questions/48937900/round-time-to-nearest-hour-python
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + dt.timedelta(hours=t.minute//30))


target_select = Select(value="PM10", title='Target', options=['PM10', 'PM25'])

source = get_dataset(target="PM10")

xmin = min(source.data['date'])
xmax = max(source.data['date'])
xrange_slider = DateRangeSlider(start=xmin,
                                end=xmax,
                                value=(xmin, xmax),
                                step=1, title="Date Range")
ymin, ymax = get_yminmax(source)
yrange_slider = RangeSlider(start=np.floor(ymin),
                            end=np.ceil(ymax),
                            value=(np.floor(ymin), np.ceil(ymax)),
                            step=1, title="Y Range")

fit_button = Button(label="Fit Plot (All Range)", button_type="success")


def make_plot(_source, title):
    # Set up plot
    _xmin = min(_source.data['date'])
    _xmax = max(_source.data['date'])
    _ymin, _ymax = get_yminmax(_source)

    xrange_slider.update(start=_xmin, end=_xmax, value=(_xmin, _xmax))
    yrange_slider.update(start=_ymin, end=_ymax, value=(_ymin, _ymax))

    plot = figure(plot_height=600, plot_width=1080, title=title,
                  x_axis_type="datetime",
                  tools="crosshair,pan,reset,save,wheel_zoom",
                  x_range=[_xmin, _xmax], y_range=[_ymin, _ymax])
    #Blues5[3]
    plot.line('date', 'raw', source=_source, color=Dark2_3[0],
              line_width=3, line_alpha=0.7, legend_label="Raw")
    plot.line('date', 'day_sea', source=_source, color=RdBu4[0],
              line_width=3, line_alpha=0.7, legend_label="Daily Seasonality")
    plot.line('date', 'day_res', source=_source, color=RdBu4[1],
              line_width=3, line_alpha=0.7, legend_label="Daily Residual")
    plot.line('date', 'year_sea', source=_source, color=RdBu4[3],
              line_width=3, line_alpha=0.7, legend_label="Annual Seasonaliy")
    plot.line('date', 'year_res', source=_source, color=RdBu4[2],
              line_width=3, line_alpha=0.7, legend_label="Annual Residual")

    plot.xaxis.axis_label = "Dates"
    plot.axis.axis_label_text_font_style = "bold"
    plot.grid.grid_line_alpha = 0.5
    plot.legend.click_policy = "hide"

    return plot


def update_plot(attrname, old, new):
    _target = target_select.value
    plot.title.text = " Seasonality Decomposition of " + _target + " by mean"

    if isinstance(xrange_slider.value[0], int):
        _xmin = hour_rounder(dt.datetime.fromtimestamp(
            xrange_slider.value[0] / 1e3))
    elif isinstance(xrange_slider.value[0], dt.datetime):
        _xmin = hour_rounder(xrange_slider.value[0])
    elif isinstance(xrange_slider.value[0], np.datetime64):
        _xmin = hour_rounder(dt.datetime.utcfromtimestamp(
            xrange_slider.value[0].astype('O')/1e9))

    if type(xrange_slider.value[1]) == int:
        _xmax = hour_rounder(dt.datetime.fromtimestamp(
            xrange_slider.value[1] / 1e3))
    elif isinstance(xrange_slider.value[1], dt.datetime):
        _xmax = hour_rounder(xrange_slider.value[1])
    elif isinstance(xrange_slider.value[1], np.datetime64):
        _xmax = hour_rounder(dt.datetime.utcfromtimestamp(
            xrange_slider.value[1].astype('O')/1e9))

    # https://stackoverflow.com/a/61119600/743078
    plot.x_range.start = _xmin
    plot.x_range.end = _xmax
    plot.y_range.start = yrange_slider.value[0]
    plot.y_range.end = yrange_slider.value[1]

    src = get_dataset(target=_target)
    source.data.update(src.data)


def fit_plot():
    _xmin = min(source.data['date'])
    _xmax = max(source.data['date'])
    _ymin, _ymax = get_yminmax(source)

    xrange_slider.update(start=_xmin, end=_xmax, value=(_xmin, _xmax))
    yrange_slider.update(start=_ymin, end=_ymax, value=(_ymin, _ymax))

    plot.x_range.start = _xmin
    plot.x_range.end = _xmax
    plot.y_range.start = _ymin
    plot.y_range.end = _ymax


# default plot
plot = make_plot(source, "Seasonality Decompsition of PM10 by STL")

target_select.on_change('value', update_plot)
xrange_slider.on_change('value_throttled', update_plot)
yrange_slider.on_change('value_throttled', update_plot)

fit_button.on_click(fit_plot)

# Set up layouts and add to document
controls = column(target_select, xrange_slider, yrange_slider, fit_button)
curdoc().add_root(row(plot, controls))
curdoc().title = "Seasonality Decomposition"
