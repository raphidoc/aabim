class TilingStrategy:
    @staticmethod
    def create(image, window_size=256):
        return image.create_windows(window_size)