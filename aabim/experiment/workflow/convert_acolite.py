import os
from reverie.converter import prisma

if __name__ == "__main__":
    data_dir = "/D/Documents/phd/training/sls_2024/practicals/presentation/"

    images = [
        # "PRISMA_2021_06_29_15_39_37_L2R.nc"
        "PRS_L2C_STD_20210629153937_20210629153942_0001.he5"
    ]

    for image in images:
        l1 = prisma.read_he5.PrismaHe5(
            he5_file=os.path.join(data_dir + image),
        )

        l1.to_reve_nc()