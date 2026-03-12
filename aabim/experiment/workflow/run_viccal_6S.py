import subprocess
import pandas as pd
from tqdm import tqdm


def run_viccal_6s(params: str):
    exec_6sv2 = "/home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1"
    process = subprocess.run(exec_6sv2, input=params, text=True, capture_output=True)

    # Check if the process completed successfully
    if process.returncode != 0:
        raise Exception(
            "6S execution failed with error code %s" % process.returncode,
            process.stderr,
        )
    else:
        return process.stdout


if __name__ == "__main__":
    input_data = pd.read_csv(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/radiance_at_sensor_rho_w_plus_sky_merged.csv"
    )

    temp = pd.DataFrame()
    for i in tqdm(range(len(input_data))):
        kwargs = {
            "sol_zen": input_data["sun_zenith_mean"][i],
            "sol_azi": input_data["sun_azimuth_mean"][i],
            "view_zen": input_data["view_zenith_mean"][i],
            "view_azi": input_data["view_azimuth_mean"][i],
            "aot550": 0.12,
            "sensor_altitude": -(input_data["z_mean"][i]/1000),
            "wavelength": input_data["Wavelength"][i] * 1e-3,
            "water_reflectance": input_data["rhow_plus_glint"][i],
        }

        input_str = "\n".join(
            [
                # Geometrical conditions
                "0 # Geometrical conditions igeom",
                "%(sol_zen).2f %(sol_azi).2f %(view_zen).2f %(view_azi).2f 1 1"
                % kwargs,
                "8 ",  # Atmospheric conditions, gas (idatm)
                "%(water_vapor)" % kwargs,  # water vapor in g/cm2
                "%(ozone)" % kwargs,  # Ozone content (cm-atm),
                # Aerosol type and concentration
                "2 # Maritime aerosol model",
                "0 # Visibility (km)",
                "%(aot550).2f" % kwargs,
                # Target altitude as pressure in mb
                "0 # target altitude (xps)",
                # Sensor altitude in - km relative to target altitude
                "%(sensor_altitude).4f # Sensor altitude (xpp)" % kwargs,
                # If aircraft simulation (> -110 < 0) enter water vapor, ozone content
                # Aircraft water vapor, ozone and aot550
                "-1.0 -1.0",
                "-1.0",
                # Spectrum selection for simulation
                "-1 # Spectrum monochromatic selection (iwave)",
                "%(wavelength).3f" % kwargs,
                # Ground reflectance
                "0 # Ground reflectance (inhomo)",
                "0",
                "0",
                "%(water_reflectance).6f" % kwargs,
                # Atmospheric correction
                "-1 # Atmospheric correction (irapp)",
            ]
        )

        # with open("/reverie/6S/in.txt", "w") as f:
        #     f.writelines(input_str)

        res = run_viccal_6s(input_str)

        import json

        test = json.loads(res)

        test = pd.DataFrame([test])
        test.index = [i]
        #
        # # Create a new DataFrame with the same columns as `input_data`
        # test = pd.DataFrame([test], columns=input_data.columns)
        #
        # # Append `test` to `input_data`
        # input_data = input_data.append(test, ignore_index=True)

        temp = pd.concat([temp, test], axis=0)

    input_data = pd.concat([input_data, temp], axis=1)

    input_data.to_csv(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/6S/6S_Lt_rhow_merged.csv"
    )

    # with open("/home/raphael/PycharmProjects/reverie/reverie/6S/out.txt", "w") as f:
    #     f.writelines(res)
