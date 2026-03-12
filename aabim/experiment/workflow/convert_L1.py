import os
from reverie.converter import wise

if __name__ == "__main__":
    data_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.dpix",
        #"ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.dpix",
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.dpix",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.dpix",
        # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.dpix",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.dpix",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.dpix",
        "MC-11A/190820_MC-11A-WI-1x1x1_v02-L1CG.dpix",
    ]

    for image in images:
        l1 = wise.read_pix.Pix(
            pix_dir=os.path.join(data_dir + image),
        )

        l1.to_reve_nc()
