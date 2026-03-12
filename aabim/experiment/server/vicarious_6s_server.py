import concurrent.futures
import json
import os
import subprocess
import time

import pandas as pd

from bokeh.models import Slider
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column


# Function to run model and accumulate results
def run_model_and_accumulate(start, end, commands, y):
    global counter  # Declare counter as global

    for i in range(start, end):
        command = commands[i]
        process = subprocess.run(command, shell=True, capture_output=True)

        temp = json.loads(process.stdout)

        y[i] = (float(temp["atmospheric_radiance_at_sensor_[W m-2 sr-1 um-1]"]) + float(
            temp["pixel_radiance_at_sensor_[W m-2 sr-1 um-1]"]
        )) * 1e-1

        counter += 1

    return


def format_estimated_time(estimated_time):
    if estimated_time >= 60.0 * 60 * 24 * 365 * 100:
        return str(estimated_time / (60.0 * 60 * 24 * 365 * 100)) + " centuries"
    elif estimated_time >= 60.0 * 60 * 24 * 365:
        return str(estimated_time / (60.0 * 60 * 24 * 365)) + " years"
    elif estimated_time >= 60 * 60 * 24 * 30:
        return str(estimated_time / (60 * 60 * 24 * 30)) + " months"
    elif estimated_time >= 60 * 60 * 24 * 7:
        return str(estimated_time / (60 * 60 * 24 * 7)) + " weeks"
    elif estimated_time >= 60 * 60 * 24:
        return str(estimated_time / (60 * 60 * 24)) + " days"
    elif estimated_time >= 60 * 60:
        return str(estimated_time / (60 * 60)) + " hours"
    elif estimated_time >= 60:
        return str(estimated_time / 60) + " minutes"
    else:
        return str(estimated_time) + " seconds"


input_data = pd.read_csv(
    "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/radiance_at_sensor_rho_w_plus_sky_merged.csv"
)

# 37ba3e42-73bb-11ee-8b3c-8cec4bb24221 or d20f1ac6-7203-11ee-9369-3cf011fcb4b7

input_data = input_data[input_data["UUID"] == "5f23b1ec-dfd7-11ee-9e2e-3cf011fcb4b7"]

# Define the parameter range on which 6s can be run
# 20 node wavelength of 6s
# wavelength = [
#     0.350,
#     0.400,
#     0.412,
#     0.443,
#     0.470,
#     0.488,
#     0.515,
#     0.550,
#     0.590,
#     0.633,
#     0.670,
#     0.694,
#     0.760,
#     0.860,
#     # 1.240,
#     # 1.536,
#     # 1.650,
#     # 1.950,
#     # 2.250,
#     # 3.750,
# ]

# Set up initial values
sel_ths = input_data.sun_zenith_mean.unique()[0]
sel_phis = input_data.sun_azimuth_mean.unique()[0]
sel_thv = input_data.view_zenith_mean.unique()[0]
sel_phiv = input_data.view_azimuth_mean.unique()[0]
sel_water_vapor = 0
sel_ozone = 0.3
# aerosol type 2, maritime
sel_dust_aer = 0.0  # volumetric % of dust-like
sel_water_soluble_aer = 0.05  # volumetric % of water-soluble
sel_oceanic_aer = 0.95  # volumetric % of oceanic
sel_soot_aer = 0.0  # volumetric % of soot
sel_aot550 = 0.4
sel_pressure = 1100

# Convert positive meters altitude in negative kilometers
sel_altitude = -input_data.z_mean.unique()[0] / 1000


wavelength = (input_data.Wavelength.unique() * 1e-3).tolist()

surface_reflectance = input_data.rhow_plus_glint.tolist()

# Set up widgets
ths_slider = Slider(
    start=0,
    end=90,
    value=sel_ths,
    step=1,
    title="sol_zenith",
)
phis_slider = Slider(
    start=0,
    end=360,
    value=sel_phis,
    step=1,
    title="sol_azimuth",
)
thv_slider = Slider(
    start=0,
    end=90,
    value=sel_thv,
    step=1,
    title="view_zenith",
)
phiv_slider = Slider(start=0, end=360, value=sel_phiv, step=1, title="view_azimuth")

# Gas concentration
water_slider = Slider(
    start=0,
    end=4,
    value=sel_water_vapor,
    step=0.01,
    title="water_vapor",
)
ozone_slider = Slider(start=0, end=0.8, value=sel_ozone, step=0.01, title="ozone")

# Aerosol type and concentration
dust_aer_slider = Slider(
    start=0, end=1, value=sel_dust_aer, step=0.1, title="dust_aerosol"
)
water_soluble_aer_slider = Slider(
    start=0, end=1, value=sel_water_soluble_aer, step=0.1, title="water_soluble_aerosol"
)
oceanic_aer_slider = Slider(
    start=0, end=1, value=sel_oceanic_aer, step=0.1, title="oceanic_aerosol"
)
soot_aer_slider = Slider(
    start=0, end=1, value=sel_soot_aer, step=0.1, title="soot_aerosol"
)
aot550_slider = Slider(start=0, end=4, value=sel_aot550, step=0.001, title="aot550")

pressure_slider = Slider(
    start=0,
    end=1100,
    value=sel_pressure,
    step=1,
    title="target_pressure",
)
altitude_slider = Slider(
    start=-20,
    end=-0.1,
    value=sel_altitude,
    step=-0.1,
    title="sensor_altitude",
)

# Create initial data value
# Declare the global counter
counter = 0

y = [0] * len(wavelength)

commands = []
for i in range(len(wavelength)):
    command = (
        'echo "\n0 # IGEOM\n'
        + f"{sel_ths} {sel_phis} {sel_thv} {sel_phiv} 1 1 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
        + "8 # IDATM user\n"
        + f"{sel_water_vapor}\n"
        + f"{sel_ozone}\n"
        + "4 # IAER user\n"
        + f"{sel_dust_aer}\n"
        + f"{sel_water_soluble_aer}\n"
        + f"{sel_oceanic_aer}\n"
        + f"{sel_soot_aer}\n"
        + "0 # visibility\n"
        + f"{sel_aot550}\n"
        + f"{sel_pressure} # XPS pressure at terget\n"
        + f"{sel_altitude} # XPP sensor altitude\n"
        + "-1.0 -1.0 # UH20 UO3 below sensor\n"
        + "-1.0 # taer550 below sensor\n"
        + "-1 # IWAVE monochromatic\n"
        + f"{wavelength[i]} # wavelength\n"
        + "0 # INHOMO\n"
        + "0 # IDIREC\n"
        + "0 # IGROUN 0 = rho\n"
        + f"{surface_reflectance[i]} # surface reflectance\n"
        + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1'
    )

    commands.append(command)

    # process = subprocess.run(command, shell=True, capture_output=True)
    #
    # temp = json.loads(process.stdout)
    #
    # y[i] = float(temp["atmospheric_radiance_at_sensor_[W m-2 sr-1 um-1]"]) + float(
    #     temp["pixel_radiance_at_sensor_[W m-2 sr-1 um-1]"]
    # )


# Determine the number of workers to use
num_workers = 6  # os.cpu_count()

print(f"Running on {num_workers} threads")

# Calculate the number of iterations per worker
iterations_per_worker = len(wavelength) // num_workers

print(f"{iterations_per_worker} iteration per threads")

start_time = time.perf_counter()

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Start each worker
    futures = []
    for i in range(num_workers):
        # Calculate the start and end indices for this worker
        start = i * iterations_per_worker
        end = start + iterations_per_worker if i != num_workers - 1 else len(wavelength)

        # Start the worker
        futures.append(
            executor.submit(run_model_and_accumulate, start, end, commands, y)
        )

    while counter < len(wavelength):
        time.sleep(1)  # Sleep for a second

        now = time.perf_counter()
        elapsed = now - start_time

        if elapsed > 0 and counter > 0:  # To avoid division by zero
            print(f"\r({counter}/{len(wavelength)}) | ", end="")

            iterations_per_second = counter / elapsed
            print(f"Iterations per second: {iterations_per_second}", end="")

            estimated_total_time = (len(wavelength) - counter) / iterations_per_second
            print(
                f" | Estimated time upon completion: {format_estimated_time(estimated_total_time)}",
                end="",
                flush=True,
            )

    # Wait for all workers to finish
    concurrent.futures.wait(futures)

# Set up plot
plot = figure(
    height=400,
    width=600,
    title="atmospheric radiance at sensor",
    tools="crosshair,pan,reset,save,box_zoom,wheel_zoom",
    x_range=[wavelength[0], wavelength[-1]],
    # y_range=[lookup_table.min(), lookup_table.max()],
)

# Plot measured radiance value
plot.line(
    x=wavelength,
    y=input_data.radiance_at_sensor_mean.tolist(),
    line_width=3,
    line_alpha=1,
    line_color="blue",
)

r = plot.line(x=wavelength, y=y, line_width=3, line_alpha=1, line_color="red")

ds = r.data_source


# create a callback that adds a number in a random location
def callback(event):
    # Get the current values of the sliders

    sel_ths = ths_slider.value
    sel_thv = thv_slider.value
    sel_phis = phis_slider.value
    sel_phiv = phiv_slider.value
    sel_water_vapor = water_slider.value
    sel_ozone = ozone_slider.value
    sel_dust_aer = dust_aer_slider.value
    sel_water_soluble_aer = water_soluble_aer_slider.value
    sel_oceanic_aer = oceanic_aer_slider.value
    sel_soot_aer = soot_aer_slider.value
    sel_aot550 = aot550_slider.value

    sel_pressure = pressure_slider.value
    sel_altitude = altitude_slider.value

    # res = [0] * len(wavelength)
    #
    # start_time = time.perf_counter()
    #
    # counter = 0
    #
    # for i in range(len(wavelength)):
    #     command = (
    #         'echo "\n0 # IGEOM\n'
    #         + f"{sel_ths} {sel_phis} {sel_thv} {sel_phiv} 1 1 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
    #         + "8 # IDATM no gas\n"
    #         + f"{sel_water_vapor}\n"
    #         + f"{sel_ozone}\n"
    #         + "4 # IAER maritime\n"
    #         + f"{sel_dust_aer}\n"
    #         + f"{sel_water_soluble_aer}\n"
    #         + f"{sel_oceanic_aer}\n"
    #         + f"{sel_soot_aer}\n"
    #         + "0 # visibility\n"
    #         + f"{sel_aot550}\n"
    #         + f"{sel_pressure} # XPS pressure at terget\n"
    #         + f"{sel_altitude} # XPP sensor altitude\n"
    #         + "-1.0 -1.0 # UH20 UO3 below sensor\n"
    #         + "-1.0 # taer550 below sensor\n"
    #         + "-1 # IWAVE monochromatic\n"
    #         + f"{wavelength[i]} # wavelength\n"
    #         + "0 # INHOMO\n"
    #         + "0 # IDIREC\n"
    #         + "0 # IGROUN 0 = rho\n"
    #         + f"{surface_reflectance[i]} # surface reflectance\n"
    #         + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1'
    #     )
    #
    #     process = subprocess.run(command, shell=True, capture_output=True)
    #
    #     temp = json.loads(process.stdout)
    #
    #     res[i] = float(temp["atmospheric_radiance_at_sensor_[W m-2 sr-1 um-1]"]) + float(
    #     temp["pixel_radiance_at_sensor_[W m-2 sr-1 um-1]"]
    # )
    #
    #     counter += 1
    #
    # end_time = time.perf_counter()
    # print(f"Time taken: {end_time - start_time} seconds")
    # print(f"iteration per seconds: {counter / (end_time - start_time)}")

    counter = 0

    y = [0] * len(wavelength)

    commands = []
    for i in range(len(wavelength)):
        command = (
            'echo "\n0 # IGEOM\n'
            + f"{sel_ths} {sel_phis} {sel_thv} {sel_phiv} 1 1 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
            + "8 # IDATM user\n"
            + f"{sel_water_vapor}\n"
            + f"{sel_ozone}\n"
            + "4 # IAER user\n"
            + f"{sel_dust_aer}\n"
            + f"{sel_water_soluble_aer}\n"
            + f"{sel_oceanic_aer}\n"
            + f"{sel_soot_aer}\n"
            + "0 # visibility\n"
            + f"{sel_aot550}\n"
            + f"{sel_pressure} # XPS pressure at terget\n"
            + f"{sel_altitude} # XPP sensor altitude\n"
            + "-1.0 -1.0 # UH20 UO3 below sensor\n"
            + "-1.0 # taer550 below sensor\n"
            + "-1 # IWAVE monochromatic\n"
            + f"{wavelength[i]} # wavelength\n"
            + "0 # INHOMO\n"
            + "0 # IDIREC\n"
            + "0 # IGROUN 0 = rho\n"
            + f"{surface_reflectance[i]} # surface reflectance\n"
            + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1'
        )

        commands.append(command)

        # process = subprocess.run(command, shell=True, capture_output=True)
        #
        # temp = json.loads(process.stdout)
        #
        # y[i] = float(temp["atmospheric_radiance_at_sensor_[W m-2 sr-1 um-1]"]) + float(
        #     temp["pixel_radiance_at_sensor_[W m-2 sr-1 um-1]"]
        # )

    # Determine the number of workers to use
    num_workers = 6  # os.cpu_count()

    print(f"Running on {num_workers} threads")

    # Calculate the number of iterations per worker
    iterations_per_worker = len(wavelength) // num_workers

    print(f"{iterations_per_worker} iteration per threads")

    start_time = time.perf_counter()

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start each worker
        futures = []
        for i in range(num_workers):
            # Calculate the start and end indices for this worker
            start = i * iterations_per_worker
            end = (
                start + iterations_per_worker
                if i != num_workers - 1
                else len(wavelength)
            )

            # Start the worker
            futures.append(
                executor.submit(run_model_and_accumulate, start, end, commands, y)
            )

        # For some reason enter in a deadlock situation when trying to print the progress from the callback
        # It seems that counter never gets updated by the workers even so the y list gets updated ...
        # while counter < len(wavelength):
        #     time.sleep(1)  # Sleep for a second
        #
        #     now = time.perf_counter()
        #     elapsed = now - start_time
        #
        #     if elapsed > 0 and counter > 0:  # To avoid division by zero
        #         print(f"\r({counter}/{len(wavelength)}) | ", end="")
        #
        #         iterations_per_second = counter / elapsed
        #         print(f"Iterations per second: {iterations_per_second}", end="")
        #
        #         estimated_total_time = (
        #             len(wavelength) - counter
        #         ) / iterations_per_second
        #         print(
        #             f" | Estimated time upon completion: {format_estimated_time(estimated_total_time)}",
        #             end="",
        #             flush=True,
        #         )

        # Wait for all workers to finish
        concurrent.futures.wait(futures)

        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"iteration per seconds: {counter / (end_time - start_time)}")

        print("Done")

    # BEST PRACTICE --- update .data in one step with a new dict
    ds.data = {
        "x": wavelength,
        "y": y,
    }


# add a button widget and configure with the call back
# Attach the callback to the sliders
# ths_slider.on_change("value", callback)
# phis_slider.on_change("value", callback)
# thv_slider.on_change("value", callback)
# phiv_slider.on_change("value", callback)
# water_slider.on_change("value", callback)
# ozone_slider.on_change("value", callback)
# dust_aer_slider.on_change("value", callback)
# water_soluble_aer_slider.on_change("value", callback)
# oceanic_aer_slider.on_change("value", callback)
# soot_aer_slider.on_change("value", callback)
# aot550_slider.on_change("value", callback)
# pressure_slider.on_change("value", callback)
# altitude_slider.on_change("value", callback)

from bokeh.events import ButtonClick
from bokeh.models import Button

button = Button(label="Run", button_type="success")

# Attach the callback function to the button's click event
button.on_click(callback)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        column(
            ths_slider,
            phis_slider,
            thv_slider,
            phiv_slider,
            water_slider,
            ozone_slider,
            dust_aer_slider,
            water_soluble_aer_slider,
            oceanic_aer_slider,
            soot_aer_slider,
            aot550_slider,
            pressure_slider,
            altitude_slider,
            button,
        ),
        column(plot),
    )
)
