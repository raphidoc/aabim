import xarray as xr

from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Select

rho_sky_nc = xr.open_dataset(
    "/D/Documents/PhD/Thesis/Chapter2/Data/ACOLITE-RSKY-202102-82W-MOD2.nc"
)

new_rho_sky_nc = xr.Dataset(
    data_vars=dict(
        rho_sky=(
            [
                "wavelength",
                "relative_azimuth",
                "view_zenith",
                "sun_zenith",
                "wind_speed",
                "aot",
            ],
            rho_sky_nc.lut[:, :, :, :, :, :].values,
        )
    ),
    coords=dict(
        wavelength=(["wavelength"], rho_sky_nc.attrs["wave"] * 1e3),
        relative_azimuth=(["relative_azimuth"], rho_sky_nc.attrs["azi"]),
        view_zenith=(["view_zenith"], rho_sky_nc.attrs["thv"]),
        sun_zenith=(["sun_zenith"], rho_sky_nc.attrs["ths"]),
        wind_speed=(["wind_speed"], rho_sky_nc.attrs["wind"]),
        aot=(["aot"], rho_sky_nc.attrs["tau"]),
    ),
    attrs=dict(description="Sky glint look up table computed with OSOAA for ACOLITE"),
)

new_rho_sky_nc = new_rho_sky_nc.sel(wavelength=slice(330, 999))

# Generate some sample data
wavelength = new_rho_sky_nc.variables["wavelength"].values
relative_azimuth = new_rho_sky_nc.variables["relative_azimuth"].values
view_zenith = new_rho_sky_nc.variables["view_zenith"].values
sun_zenith = new_rho_sky_nc.variables["sun_zenith"].values
wind_speed = new_rho_sky_nc.variables["wind_speed"].values
aot = new_rho_sky_nc.variables["aot"].values

# Create a lookup table with random data
lookup_table = new_rho_sky_nc.variables["rho_sky"].values

# Set up initial values
selected_azi = relative_azimuth[0]
selected_thv = view_zenith[0]
selected_ths = sun_zenith[0]
selected_wind = wind_speed[0]
selected_tau = aot[0]

# Convert the lookup_table to a list of lists for JS to recognize it as array
lookup_table_list = lookup_table.tolist()

# Set up the data source
source = ColumnDataSource(
    data=dict(
        x=wavelength, y=lookup_table[:, 0, 0, 0, 0, 0], lookup_table=lookup_table_list
    )
)

# Set up plot
plot = figure(
    height=400,
    width=600,
    title="Lookup Table",
    tools="crosshair,pan,reset,save,box_zoom,wheel_zoom",
    x_range=[wavelength[0], wavelength[-1]],
    #y_range=[lookup_table.min(), lookup_table.max()],
)

plot.line("x", "y", source=source, line_width=3, line_alpha=0.6)

# Set up widgets
azi_select = Select(
    title="relative_azimuth:", value=str(selected_azi), options=[str(x) for x in relative_azimuth]
)
thv_select = Select(
    title="view_azimuth:", value=str(selected_thv), options=[str(x) for x in view_zenith]
)
ths_select = Select(
    title="sun_azimuth:", value=str(selected_ths), options=[str(x) for x in sun_zenith]
)
wind_select = Select(
    title="Wind speed:", value=str(selected_wind), options=[str(x) for x in wind_speed]
)
tau_select = Select(
    title="aot(550):", value=str(selected_tau), options=[str(x) for x in aot]
)

# Set up callbacks
callback = CustomJS(
    args=dict(
        source=source,
        azi_select=azi_select,
        thv_select=thv_select,
        ths_select=ths_select,
        wind_select=wind_select,
        tau_select=tau_select,
        relative_azimuth=relative_azimuth,
        view_zenith=view_zenith,
        sun_zenith=sun_zenith,
        wind_speed=wind_speed,
        aot=aot,
    ),
    code="""
    name data = source.data;
    name relative_azimuth = relative_azimuth.indexOf(parseFloat(azi_select.value));
    name view_zenith = view_zenith.indexOf(parseFloat(thv_select.value));
    name sun_zenith = sun_zenith.indexOf(parseFloat(ths_select.value));
    name wind_speed = wind_speed.indexOf(parseFloat(wind_select.value));
    name aot = aot.indexOf(parseFloat(tau_select.value));

    console.log('LUT is array: ', Array.isArray(data['lookup_table']));
    console.log('lookup_table shape:', data['lookup_table'].length, data['lookup_table'][0].length);

    function getShape(arr) {
        name shape = [];
        while(Array.isArray(arr)) {
            shape.push(arr.length);
            arr = arr[0];
        }
        return shape;
    }

    console.log('indices:', relative_azimuth, view_zenith, sun_zenith, wind_speed, aot);

    console.log('LUT shape:',getShape(data['lookup_table']));

    if (relative_azimuth === -1 || view_zenith === -1 || sun_zenith === -1 || wind_speed === -1 || aot === -1) {
        // Handle the case where the selected value is not found in the array
        console.error('Selected value not found in array');
    } else {
        name new_y = [];
        name wavelengths = data['lookup_table'].length;

        for (name i = 0; i < wavelengths; i++) {
            name value = data['lookup_table'][i][relative_azimuth][view_zenith][sun_zenith][wind_speed][aot];
            new_y.push(value);
        }

        console.log('new y shape: ', getShape(new_y));
        console.log('new y values: ', new_y);
        source.data['y'] = new_y;
        source.change.emit();
    }
""",
)

azi_select.js_on_change("value", callback)
thv_select.js_on_change("value", callback)
ths_select.js_on_change("value", callback)
wind_select.js_on_change("value", callback)
tau_select.js_on_change("value", callback)

#Set up layouts and add to document
inputs = column(azi_select, thv_select, ths_select, wind_select, tau_select)

# from bokeh.models import Slider
# Set up widgets
# azi_slider = Slider(
#     start=relative_azimuth.min(),
#     end=relative_azimuth.max(),
#     value=selected_azi,
#     step=1,
#     title="Azi",
# )
# thv_slider = Slider(
#     start=view_zenith.min(),
#     end=view_zenith.max(),
#     value=sel_thv,
#     step=1,
#     title="Thv",
# )
# ths_slider = Slider(
#     start=sun_zenith.min(),
#     end=sun_zenith.max(),
#     value=sel_ths,
#     step=1,
#     title="Ths",
# )
# wind_slider = Slider(
#     start=wind_speed.min(),
#     end=wind_speed.max(),
#     value=selected_wind,
#     step=1,
#     title="Wind",
# )
# tau_slider = Slider(
#     start=aot.min(), end=aot.max(), value=selected_tau, step=0.01, title="Tau"
# )
#
# # Set up callbacks
# callback = CustomJS(
#     args=dict(
#         source=source,
#         azi_select=azi_select,
#         thv_select=thv_select,
#         ths_select=ths_select,
#         wind_select=wind_select,
#         tau_select=tau_select,
#         relative_azimuth=relative_azimuth,
#         view_zenith=view_zenith,
#         sun_zenith=sun_zenith,
#         wind_speed=wind_speed,
#         aot=aot,
#     ),
#     code="""
#     name data = source.data;
#     name relative_azimuth = relative_azimuth.indexOf(parseFloat(azi_select.value));
#     name view_zenith = view_zenith.indexOf(parseFloat(thv_select.value));
#     name sun_zenith = sun_zenith.indexOf(parseFloat(ths_select.value));
#     name wind_speed = wind_speed.indexOf(parseFloat(wind_select.value));
#     name aot = aot.indexOf(parseFloat(tau_select.value));
#
#     console.log('LUT is array: ', Array.isArray(data['lookup_table']));
#     console.log('lookup_table shape:', data['lookup_table'].length, data['lookup_table'][0].length);
#
#     function getShape(arr) {
#         name shape = [];
#         while(Array.isArray(arr)) {
#             shape.push(arr.length);
#             arr = arr[0];
#         }
#         return shape;
#     }
#
#     console.log('indices:', relative_azimuth, view_zenith, sun_zenith, wind_speed, aot);
#
#     console.log('LUT shape:',getShape(data['lookup_table']));
#
#     if (relative_azimuth === -1 || view_zenith === -1 || sun_zenith === -1 || wind_speed === -1 || aot === -1) {
#         // Handle the case where the selected value is not found in the array
#         console.error('Selected value not found in array');
#     } else {
#         name new_y = [];
#         name wavelengths = data['lookup_table'].length;
#
#         for (name i = 0; i < wavelengths; i++) {
#             name value = data['lookup_table'][i][relative_azimuth][view_zenith][sun_zenith][wind_speed][aot];
#             new_y.push(value);
#         }
#
#         console.log('new y shape: ', getShape(new_y));
#         console.log('new y values: ', new_y);
#         source.data['y'] = new_y;
#         source.change.emit();
#     }
# """,
# )
#
# azi_slider.js_on_change("value", callback)
# thv_slider.js_on_change("value", callback)
# ths_slider.js_on_change("value", callback)
# wind_slider.js_on_change("value", callback)
# tau_slider.js_on_change("value", callback)
#
# # Set up layouts and add to document
# inputs = column(azi_slider, thv_slider, ths_slider, wind_slider, tau_slider)

from bokeh.plotting import output_file, save

# Set the output file
output_file("lookup_table_plot.html")

# Add the plot to the document
save(row(inputs, plot, width=800))

# html = file_html(row(plot, column(amp, freq, phase, offset)), CDN, "my plot")

# with open("/home/raphael/PycharmProjects/reverie/reverie/6S/out.txt", "w") as f:
#     f.writelines(html)
