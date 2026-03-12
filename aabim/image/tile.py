from rasterio.windows import Window


# Window().
class Tile:
    """
    tile is a rectangle which is used to seperate an image
    """

    @classmethod
    def FromWindow(cls, window: Window):
        """
        creat an instance from a instance of Window
        :param window:
        :return:
        """
        return cls(
            window.row_off,
            window.row_off + window.height,
            window.col_off,
            window.col_off + window.width,
        )

    def __init__(self, sline, eline, spixl, epixl):
        self.sline = sline
        self.spixl = spixl
        self.eline = eline
        self.epixl = epixl
        self.xsize = epixl - spixl
        self.ysize = eline - sline
        self.data = (self.sline, self.eline, self.spixl, self.epixl)

    def to_window(self):
        """
        convert into an instance of Window of rasterio
        :return:
        """
        return Window.from_slices((self.sline, self.eline), (self.spixl, self.epixl))

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "sline:{},eline:{},spixl:{},epixl:{}".format(
            self.sline, self.eline, self.spixl, self.epixl
        )

    def __repr__(self):
        return "sline:{},eline:{},spixl:{},epixl:{}".format(
            self.sline, self.eline, self.spixl, self.epixl
        )

    def __add__(self, other):
        if not isinstance(other, Tile):
            raise TypeError("can only two Tiles do addition")
        _sline = min(self.sline, other.sline)
        _eline = max(self.eline, other.eline)
        _spixl = min(self.spixl, other.epixl)
        _epixl = max(self.epixl, other.epixl)
        _tile = (_sline, _eline, _spixl, _epixl)

        return _tile

    def __sub__(self, other):
        """
        to do
        :param other:
        :return:
        """
        pass

    def expand(self, size):
        self.eline += size
        self.epixl += size
        self.spixl -= size
        self.epixl -= size

    def shrink(self, size):
        self.eline -= size
        self.epixl -= size
        self.spixl += size
        self.epixl += size
