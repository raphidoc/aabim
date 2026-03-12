import xarray as xr
import numpy as np

from bokeh.models import Slider
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column

import scipy as sp

rho_path_nc = xr.open_dataset(
    "/home/raphael/PycharmProjects/acolite/data/LUT/ACOLITE-LUT-202110/ACOLITE-LUT-202110-MOD2-1013mb.nc"
)

new_rho_path_nc = xr.Dataset(
    data_vars=dict(
        atmospheric_radiance_at_sensor=(
            [
                "wavelength",
                "relative_azimuth",
                "view_zenith",
                "sun_zenith",
                "aot",
            ],
            rho_path_nc.lut[16, :, :, :, :, 0, :].values,
        )
    ),
    coords=dict(
        wavelength=(["wavelength"], rho_path_nc.attrs["wave"] * 1e3),
        relative_azimuth=(["relative_azimuth"], rho_path_nc.attrs["azi"]),
        view_zenith=(["view_zenith"], rho_path_nc.attrs["thv"]),
        sun_zenith=(["sun_zenith"], rho_path_nc.attrs["ths"]),
        aot=(["aot"], rho_path_nc.attrs["tau"]),
    ),
    attrs=dict(description="Atmospheric path reflectance look up table computed with 6S for REVERIE"),
)

new_rho_path_nc = new_rho_path_nc.sel(wavelength=slice(330, 999))

# Generate some sample data
wavelength = new_rho_path_nc.variables["wavelength"].values
relative_azimuth = new_rho_path_nc.variables["relative_azimuth"].values
view_zenith = new_rho_path_nc.variables["view_zenith"].values
sun_zenith = new_rho_path_nc.variables["sun_zenith"].values
aot = new_rho_path_nc.variables["aot"].values

# Create a lookup table with random data
lookup_table = new_rho_path_nc.variables["atmospheric_radiance_at_sensor"].values

{"Lt":6, "dic":{"lt":6}}

# Set up initial values
selected_azi = relative_azimuth[0]
selected_thv = view_zenith[0]
selected_ths = sun_zenith[0]
selected_tau = aot[0]

# Set up widgets
azi_slider = Slider(
    start=relative_azimuth.min(),
    end=relative_azimuth.max(),
    value=selected_azi,
    step=0.1,
    title="Azi",
)
thv_slider = Slider(
    start=view_zenith.min(),
    end=view_zenith.max(),
    value=selected_thv,
    step=0.1,
    title="Thv",
)
ths_slider = Slider(
    start=sun_zenith.min(),
    end=sun_zenith.max(),
    value=selected_ths,
    step=0.1,
    title="Ths",
)
tau_slider = Slider(
    start=aot.min(), end=aot.max(), value=selected_tau, step=0.01, title="Tau"
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

r = plot.line(
    x=wavelength, y=lookup_table[:, 0, 0, 0, 0], line_width=3, line_alpha=0.6
)

# add a text renderer to the plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
#            text_baseline="middle", text_align="center")

ds = r.data_source


# create a callback that adds a number in a random location
def callback(attr, old, new):
    # Get the current values of the sliders

    azi = azi_slider.value
    thv = thv_slider.value
    ths = ths_slider.value
    tau = tau_slider.value

    # azi_index = list(relative_azimuth).index(azi)
    # thv_index = list(view_zenith).index(thv)
    # ths_index = list(sun_zenith).index(ths)
    # wind_index = list(wind_speed).index(wind)
    # tau_index = list(aot).index(tau)

    # breakpoint()

    res = sp.interpolate.interpn(
        points=(
            wavelength[()],
            relative_azimuth[()],
            view_zenith,
            sun_zenith[()],
            aot[()],
        ),
        values=lookup_table[:, :, :, :, :],
        xi=(
            wavelength[()],
            azi,
            thv,
            ths,
            tau,
        ),
    )

    # BEST PRACTICE --- update .data in one step with a new dict
    ds.data = {
        "x": wavelength,
        "y": res,
    }


# add a button widget and configure with the call back
# Attach the callback to the sliders
azi_slider.on_change("value", callback)
thv_slider.on_change("value", callback)
ths_slider.on_change("value", callback)
tau_slider.on_change("value", callback)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        column(azi_slider, thv_slider, ths_slider, tau_slider),
        column(plot),
    )
)
