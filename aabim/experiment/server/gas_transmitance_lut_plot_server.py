import xarray as xr
import numpy as np

from bokeh.models import Slider
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column

import scipy as sp

gas_trans_nc = xr.open_dataset(
    "/reverie/data/surface/output/ACI12_gas.nc"
)

# Generate some sample data
sun_zenith = gas_trans_nc.variables["sol_zen"].values
view_zenith = gas_trans_nc.variables["view_zen"].values
relative_azimuth = gas_trans_nc.variables["relative_azimuth"].values
water = gas_trans_nc.variables["water"].values
ozone = gas_trans_nc.variables["ozone"].values
target_pressure = gas_trans_nc.variables["target_pressure"].values
sensor_altitude = gas_trans_nc.variables["sensor_altitude"].values
wavelength = gas_trans_nc.variables["wavelength"].values


# Set up initial values
selected_ths = sun_zenith[0]
selected_thv = view_zenith[0]
selected_azi = relative_azimuth[0]
selected_pressure = target_pressure[0]
selected_altitude = sensor_altitude[0]
selected_water = water[0]
selected_ozone = ozone[0]

# Set up widgets
ths_slider = Slider(
    start=sun_zenith.min(),
    end=sun_zenith.max(),
    value=selected_ths,
    step=1,
    title="sol_zenith",
)
thv_slider = Slider(
    start=view_zenith.min(),
    end=view_zenith.max(),
    value=selected_thv,
    step=1,
    title="view_zenith",
)
azi_slider = Slider(
    start=relative_azimuth.min(),
    end=relative_azimuth.max(),
    value=selected_azi,
    step=1,
    title="relative_azimuth",
)
pressure_slider = Slider(
    start=target_pressure.min(),
    end=target_pressure.max(),
    value=selected_pressure,
    step=100,
    title="target_pressure",
)
altitude_slider = Slider(
    start=sensor_altitude.min(),
    end=sensor_altitude.max(),
    value=selected_altitude,
    step=-0.5,
    title="sensor_altitude",
)
water_slider = Slider(
    start=water.min(), end=water.max(), value=selected_water, step=0.01, title="water_vapor"
)
ozone_slider = Slider(
    start=ozone.min(), end=ozone.max(), value=selected_ozone, step=0.01, title="ozone"
)

# Convert the lookup_table to a list of lists for JS to recognize it as array
# lookup_table_list = lookup_table.tolist()

# Set up plot
plot = figure(
    height=400,
    width=600,
    title="Lookup Table",
    tools="crosshair,pan,reset,save,box_zoom,wheel_zoom",
    x_range=[wavelength[0], wavelength[-1]],
    # y_range=[lookup_table.min(), lookup_table.max()],
)

# Create a lookup table with random data
lookup_table = gas_trans_nc.variables["tgs"].values

r = plot.line(
    x=wavelength, y=lookup_table[0, 0, 0, 0, 0, 0, 0, :], line_width=3, line_alpha=0.6
)

# add a text renderer to the plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
#            text_baseline="middle", text_align="center")

ds = r.data_source

# create a callback that adds a number in a random location
def callback(attr, old, new):
    # Get the current values of the sliders

    ths = ths_slider.value
    thv = thv_slider.value
    azi = azi_slider.value
    pressure = pressure_slider.value
    altitude = altitude_slider.value
    water_selected = water_slider.value
    ozone_selected = ozone_slider.value

    res = sp.interpolate.interpn(
        points=(
            sun_zenith,
            view_zenith,
            relative_azimuth,
            water,
            ozone,
            target_pressure,
            sensor_altitude,
            wavelength,
        ),
        values=lookup_table[:, :, :, :, :, :, :, :],
        xi=(
            ths,
            thv,
            azi,
            water_selected,
            ozone_selected,
            pressure,
            altitude,
            wavelength[()],
        ),
    )

    # BEST PRACTICE --- update .data in one step with a new dict
    ds.data = {
        "x": wavelength,
        "y": res,
    }


# add a button widget and configure with the call back
# Attach the callback to the sliders
ths_slider.on_change("value", callback)
thv_slider.on_change("value", callback)
azi_slider.on_change("value", callback)
water_slider.on_change("value", callback)
ozone_slider.on_change("value", callback)
pressure_slider.on_change("value", callback)
altitude_slider.on_change("value", callback)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        column(ths_slider, thv_slider, azi_slider, water_slider, ozone_slider, pressure_slider, altitude_slider),
        column(plot),
    )
)
