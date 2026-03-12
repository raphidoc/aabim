import os
import numpy as np
from pandas import Series

from reverie.utils.helper import read_envi_hdr


class ErrorFlightLine(Exception):
    def __init__(
        self,
        message="Initialize FlightLine from WISE files failed,L1A or L1A-GLU header is needed",
    ):
        self.message = message
        super().__init__(self.message)


class FlightLine:
    @classmethod
    def from_wise_file(cls, nav_sum_log, glu_hdr):
        """
        initialize WISE FlightLing based on navigation log file, L1A header or L1A-GLU header
        :param nav_sum_log:  subfixed with '-Navcor_sum.log'
        :param glu_hdr: l1a header subfixed with '-L1A.pix.hdr, -L1A.glu.hdr'
        """

        # _logger.info("Initializing Flight Line from WISE navigation sum log file and L1A header:")
        # _logger.info("{},{}".format(nav_sum_log,glu_hdr))
        if not os.path.exists(glu_hdr):
            # _logger.error("{} doesn't exist,initilation of flight line failed".format(glu_hdr))
            raise ErrorFlightLine()

        def read():
            header = read_envi_hdr(glu_hdr)

            ncols = int(header["samples"])
            nrows = int(header["lines"])
            x, y = header["pixel size"].split(",")
            resolution_x, resolution_y = float(x), float(y)

            with open(nav_sum_log, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith("Average Value"):
                    line_strip = (
                        line.replace("Average Value", "")
                        .replace("+ ground pixel size", "")
                        .replace("+", "")
                        .strip()
                    )

                    # _logger.debug(line_strip.split(' '))
                    values = [
                        float(item)
                        for item in line_strip.split(" ")
                        if item.strip() != ""
                    ] + [ncols, nrows, resolution_x, resolution_y]
                    return Series(
                        data=values,
                        index=[
                            "Roll",
                            "Pitch",
                            "Heading",
                            "Distance",
                            "Height",
                            "Easting",
                            "Norhting",
                            "Samples",
                            "Lines",
                            "ResX",
                            "ResY",
                        ],
                    )

        s = read()
        return cls(
            s["Height"], s["Heading"], int(s["Samples"]), int(s["Lines"]), s["ResX"]
        )

    def __init__(self, height, heading, samples, lines, resolution_x, **kwargs):
        """
        :param height:  fly height (m)
        :param heading: attitude of flight refering to NORTH (degree)
        :param samples: number of pixels in each scanning line (int)
        :param lines:  number of scanning lines
        :param resolution_x:  resolution of the x direction (m)
        :param kwargs:   unit of distance is meter, unit of angle is degree
        """
        self.height = height
        self.heading = heading
        self.samples = samples
        self.lines = lines
        self.resolution_x = resolution_x

        self.roll = 0.0 if "roll" not in kwargs else kwargs["roll"]
        self.pitch = 0.0 if "pitch" not in kwargs else kwargs["pitch"]
        self.center_x = (
            self.samples / 2 if "center_x" not in kwargs else kwargs["center_x"]
        )
        self.surface_altitude = (
            0.0 if "surface_altitude" not in kwargs else kwargs["surface_altitude"]
        )

        self.sample_center = [
            self.resolution_x * s + self.resolution_x / 2 for s in range(samples)
        ]

    def _cal_nadir_x(self):
        """
        calculate the Nadir position in each scanning line
        :return:  nadir point (pixel), nadir point (meter)
        """
        distance_nadir2center = self.height * np.tan(np.deg2rad(self.roll))
        nadir = self.center_x * self.resolution_x - distance_nadir2center
        nadir_x = int(nadir / self.resolution_x)
        return nadir_x, nadir

    def cal_view_geom(self):
        """
        calculate viewing zenith and azimuth angle
        :return: zenith, azimuth
        """

        nadir_x, nadir = self._cal_nadir_x()
        vz = np.rad2deg(
            np.arctan(np.abs(nadir - np.asarray(self.sample_center)) / self.height)
        )

        # Convert heading to 0, 360 range
        azimuth = self.heading % 360

        # Right Wing
        va_ = (azimuth + 90) % 360
        va = np.full_like(vz, va_)

        # Left Wing
        va[nadir_x:] = (azimuth - 90) % 360

        return np.tile(vz, (self.lines, 1)), np.tile(va, (self.lines, 1))
