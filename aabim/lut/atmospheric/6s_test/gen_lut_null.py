import concurrent.futures
import itertools
import os
import subprocess
import time

import math

import netCDF4
import numpy as np
import pandas as pd

# from p_tqdm import p_uimap
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

# Function to run model and accumulate results
def run_model_and_accumulate(
    start,
    end,
    commands,
    results
):
    global counter  # Declare counter as global

    # print(f"Running task with start={start}, end={end}")

    for i in range(start, end):
        counter += 1

        command = commands[i]
        process = subprocess.run(command, shell=True, capture_output=True)

        # print(f"Subprocess exited with status {process.returncode}")

        results[i] = json.loads(process.stdout)

    return


# Declare the global counter
counter = 0

if __name__ == "__main__":
    # Function to generate cartesian product of input
    def cartesian_product(dimensions):
        return list(itertools.product(*dimensions))

    # TODO, run only for the 20 node wavelength of 6S ?
    # 20 node wavelength of 6s
    # wavelength_6s = [
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

    # Define your dims here
    # dims = [
    #     np.arange(10, 61, 10).tolist(),  # sun zenith
    #     np.arange(0, 31, 10).tolist(),  # view zenith
    #     np.arange(0, 181, 10).tolist(),  # relative azimuth
    #     [1.0, 2.0, 3.0], # H2O g/cm2
    #     [0.3, 0.5],  # Ozone cm-atm https://gml.noaa.gov/ozwv/dobson/papers/wmobro/ozone.html
    #     [
    #         1.0e-3,
    #         1.0e-2,
    #         2.0e-2,
    #         5.0e-2,
    #         1.0e-1,
    #         1.5e-1,
    #         2.0e-1,
    #         3.0e-1,
    #         5.0e-1,
    #         7.0e-1,
    #         1.0,
    #         1.3,
    #         1.6,
    #         2.0,
    #         3.0,
    #         5.0
    #     ], # AOD 555
    #     [750., 1013.],  # pressure at target mb
    #     [-1, -3, -4],  # sensor altitude -km
    #     np.arange(0.34, 1.1, 0.01).tolist(),  # wavelength
    # ]

    wavelength = [379.21, 383.63, 388.05, 392.48, 396.9, 401.32, 405.74, 410.16,
    414.58, 419., 423.42, 427.84, 432.26, 436.68, 441.1, 445.52,
    449.93, 454.35, 458.77, 463.18, 467.6, 472.01, 476.43, 480.84,
    485.25, 489.67, 494.08, 498.49, 502.9, 507.32, 511.73, 516.14,
    520.55, 524.96, 529.37, 533.78, 538.19, 542.6, 547., 551.41,
    555.82, 560.23, 564.63, 569.04, 573.45, 577.85, 582.26, 586.66,
    591.07, 595.47, 599.88, 604.28, 608.69, 613.09, 617.49, 621.9,
    626.3, 630.7, 635.1, 639.5, 643.91, 648.31, 652.71, 657.11,
    661.51, 665.91, 670.31, 674.71, 679.11, 683.51, 687.9, 692.3,
    696.7, 701.1, 705.5, 709.89, 714.29, 718.69, 723.09, 727.48,
    731.88, 736.28, 740.67, 745.07, 749.46, 753.86, 758.25, 762.65,
    767.04, 771.44, 775.83, 780.23, 784.62, 789.01, 793.41, 797.8,
    802.19, 806.59, 810.98, 815.37, 819.77, 824.16, 828.55, 832.94,
    837.34, 841.73, 846.12, 850.51, 854.9, 859.3, 863.69, 868.08,
    872.47, 876.86, 881.25, 885.64, 890.03, 894.42, 898.81, 903.21,
    907.6, 911.99, 916.38, 920.77, 925.16, 929.55, 933.94, 938.33,
    942.72, 947.11, 951.5, 955.89, 960.28, 964.67, 969.06, 973.45,
    977.83, 982.22, 986.61, 991.]

    dimensions = [
        [32],  # sun zenith
        [7],  # view zenith
        [89],  # relative azimuth
        [1007],  # pressure at target mb
        [-3],  # sensor altitude -km
        [w * 0.001 for w in wavelength]
    ]

    combination = cartesian_product(dimensions)

    print("number of combination: ", len(combination))

    # Create commands
    commands = []
    for i in range(len(combination)):
        command = (
            'echo "\n0 # IGEOM\n'
            + f"{combination[i][0]} 0.0 {combination[i][1]} {combination[i][2]} 7 5 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
            + "0 # IDATM\n"
            # + f"{combination[i][3]}\n"
            # + f"{combination[i][4]}\n"
            + "0 # IAER maritime\n"
            + f"-1 # visibility\n"
            # + f"{combination[i][5]} # aot(555)\n"
            + f"{combination[i][3]} # XPS pressure at target\n"
            + f"{combination[i][4]} # XPP sensor altitude\n"
            + "-1.0 -1.0 # UH20 UO3 below sensor\n"
            + "-1.0 # taer550 below sensor\n"
            # + "-1000 \n"
            + "-1 # IWAVE monochromatic\n"
            + f"{combination[i][5]} # wavelength\n"
            + "0 # INHOMO\n"
            + "0 # IDIREC\n"
            + "0 # IGROUN 0 = rho\n"
            + "0 # surface reflectance\n"
            + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1-json'
        )
        commands.append(command)

    # Determine the number of workers to use
    num_workers = os.cpu_count()

    results = [None] * len(combination)

    print(f"Running on {num_workers} threads")

    # Calculate the number of iterations per worker
    iterations_per_worker = len(combination) // num_workers

    print(f"{iterations_per_worker} iteration per worker")

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
                    run_model_and_accumulate,
                    start,
                    end,
                    commands,
                    results
                )
            )

        # Iterate over the Future objects as they complete
        # Allow to avoid the global counter object
        # for future in concurrent.futures.as_completed(futures):
        #     # The result of the Future is the return value of the function
        #     result = future.result()
        #
        #     # Increment the counter
        #     counter += 1
        #
        #     # Print the progress
        #     print(f"Completed {counter} out of {len(combination)} tasks")

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

    # Convert list of dicts to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df["test"] = "null"

    output_dir = "/D/Documents/phd/thesis/3_chapter/data/wise/6S/6s_test/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir, "test_null.csv"), index=False)
