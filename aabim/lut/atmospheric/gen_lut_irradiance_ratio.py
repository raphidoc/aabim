import concurrent.futures
import itertools
import os
import subprocess
import time
import csv

import netCDF4
import numpy as np
from p_tqdm import p_uimap
import json


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


def create_ir_output_nc(filename, coords, compression=None, complevel=None):
    # Create the output netcdf file
    nc = netCDF4.Dataset(
        filename,
        "w",
        format="NETCDF4",
    )
    # Create the dims

    nc.createDimension("sol_zen", len(coords["sol_zen"]))
    nc.createDimension("water", len(coords["water"]))
    nc.createDimension("ozone", len(coords["ozone"]))
    nc.createDimension("aod_550", len(coords["aod_550"]))
    nc.createDimension("target_pressure", len(coords["target_pressure"]))
    nc.createDimension("wavelength", len(coords["wavelength"]))

    sol_zen_nc = nc.createVariable("sol_zen", "f4", ("sol_zen",))
    sol_zen_nc.standard_name = "solar_zenith_angle"
    sol_zen_nc.units = "degree"
    sol_zen_nc[:] = coords["sol_zen"]

    water_nc = nc.createVariable("water", "f4", ("water",))
    water_nc.standard_name = "atmosphere_mass_content_of_water_vapor"
    water_nc.units = "g cm-2"
    water_nc[:] = coords["water"]

    ozone_nc = nc.createVariable("ozone", "f4", ("ozone",))
    ozone_nc.standard_name = "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
    ozone_nc.units = "atm-cm"
    ozone_nc[:] = coords["ozone"]

    pressure_nc = nc.createVariable("aod_550", "f4", ("aod_550",))
    pressure_nc.units = "NA"
    pressure_nc[:] = coords["aod_550"]

    pressure_nc = nc.createVariable("target_pressure", "f4", ("target_pressure",))
    pressure_nc.units = "mbar"
    pressure_nc[:] = coords["target_pressure"]

    # Create the coordinates variables
    wavelength_nc = nc.createVariable("wavelength", "f4", ("wavelength",))
    wavelength_nc.standard_name = "radiation_wavelength"
    wavelength_nc.units = "nm"
    wavelength_nc[:] = coords["wavelength"]

    # Create the data variables
    dimensions = (
        "sol_zen",
        "water",
        "ozone",
        "aod_550",
        "target_pressure",
        "wavelength",
    )

    nc.createVariable(
        "i_direct", "f4", dimensions, compression=compression, complevel=complevel
    )
    nc.createVariable(
        "i_diffuse", "f4", dimensions, compression=compression, complevel=complevel
    )

    return nc


# Function to run model and accumulate results
def run_model_and_accumulate(start, end, commands, i_direct, i_diffuse):
    global counter  # Declare counter as global

    for i in range(start, end):
        command = commands[i]
        process = subprocess.run(command, shell=True, capture_output=True)

        temp = json.loads(process.stdout)


        i_direct[i] = float(temp["direct_solar_irradiance_at_target_[W m-2 um-1]"])
        i_diffuse[i] = float(temp["diffuse_atmospheric_irradiance_at_target_[W m-2 um-1]"])
        counter += 1

    return


# Declare the global counter
counter = 0


if __name__ == "__main__":
    # Function to generate cartesian product of input iterables
    def cartesian_product(dimensions):
        return list(itertools.product(*dimensions))

    # TODO, run only for the 20 node wavelength of 6S
    #  they use liner interpolation for the rest of the spectrun anyway
    # Define your dims here
    dimensions = [
        np.arange(0, 81, 1).tolist(),  # sun zenith
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # H2O g/cm2
        [0.3, 0.5],  # Ozone cm-atm https://gml.noaa.gov/ozwv/dobson/papers/wmobro/ozone.html
        [
            0.001,
            0.01,
            0.02,
            0.05,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.7,
            1,
            1.3,
            1.6,
            2,
            3,
            5,
        ],  # aot550
        [500, 750.0, 1013.0, 1100.0],  # pressure at target mb
        np.arange(0.34, 1.1, 0.01).tolist(),  # wavelength
    ]

    combination = cartesian_product(dimensions)

    print("number of combination: ", len(combination))

    i_direct = [0] * len(combination)
    i_diffuse = [0] * len(combination)

    # Create commands
    commands = []
    for i in range(len(combination)):
        command = (
            'echo "\n0 # IGEOM\n'
            + f"{combination[i][0]} 0.0 0.0 0.0 1 1 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
            + "8 # IDATM\n"
            + f"{combination[i][1]}\n"
            + f"{combination[i][2]}\n"
            + "2 # IAER maritime\n"
            + "0 # visibility\n"
            + f"{combination[i][3]}\n"
            + f"{combination[i][4]} # XPS pressure at terget\n"
            + f"0 # XPP sensor altitude\n"
            # + "-1.0 -1.0 # UH20 UO3 below sensor\n"
            # + "-1.0 # taer550 below sensor\n"
            + "-1 # IWAVE monochromatic\n"
            + f"{combination[i][5]} # wavelength\n"
            + "0 # INHOMO\n"
            + "0 # IDIREC\n"
            + "0 # IGROUN 0 = rho\n"
            + "0 # surface reflectance\n"
            + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1'
        )
        commands.append(command)

    lut_dir = "/home/raphael/PycharmProjects/reverie/reverie/data/lut"

    output_dir = os.path.join(lut_dir, "output")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Determine the number of workers to use
    num_workers = os.cpu_count()

    print(f"Running on {num_workers} threads")

    # Calculate the number of iterations per worker
    iterations_per_worker = len(combination) // num_workers

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
                else len(combination)
            )

            # Start the worker
            futures.append(
                executor.submit(
                    run_model_and_accumulate, start, end, commands, i_direct, i_diffuse
                )
            )

        while counter < len(combination):
            time.sleep(1)  # Sleep for a second

            now = time.perf_counter()
            elapsed = now - start_time

            if elapsed > 0:  # To avoid division by zero
                print(f"\r({counter}/{len(combination)}) | ", end="")

                iterations_per_second = counter / elapsed
                print(f"Iterations per second: {iterations_per_second}", end="")

                estimated_total_time = (
                    len(combination) - counter
                ) / iterations_per_second
                print(
                    f" | Estimated time upon completion: {format_estimated_time(estimated_total_time)}",
                    end="",
                    flush=True,
                )

        # Wait for all workers to finish
        concurrent.futures.wait(futures)

        now = time.perf_counter()
        elapsed = now - start_time
        print(f"Time elapsed: {elapsed}")

        coords = {
            "sol_zen": dimensions[0],
            "water": dimensions[1],
            "ozone": dimensions[2],
            "aod_550": dimensions[3],
            "target_pressure": dimensions[4],
            "wavelength": dimensions[5],
        }

    # Write the LUT data to a CSV file
    csv_output_path = os.path.join(output_dir, "lut_ir.csv")

    import numpy as np

    # Combine data into a structured NumPy array
    data = np.hstack((combination, np.array(i_direct).reshape(-1, 1), np.array(i_diffuse).reshape(-1, 1)))

    # Define the header
    header = "sol_zen,aod_550,target_pressure,wavelength,i_direct,i_diffuse"

    # Write the data to a CSV file
    np.savetxt(csv_output_path, data, delimiter=",", header=header, comments="", fmt="%.6f")

    # Write the output to a nc file

    nc = create_ir_output_nc(os.path.join(output_dir, "lut_ir.nc"), coords)

    # Get the shape of the dims
    shape = [len(dimension) for dimension in dimensions]

    i_direct_reshaped = np.reshape(i_direct, shape)
    i_diffuse_reshaped = np.reshape(i_diffuse, shape)

    # Write the results to the file
    nc.variables["i_direct"][
        :,
        :,
        :,
        :,
        :,
        :,
    ] = i_direct_reshaped

    nc.variables["i_diffuse"][
        :,
        :,
        :,
        :,
        :,
        :,
    ] = i_diffuse_reshaped

    nc.close()
