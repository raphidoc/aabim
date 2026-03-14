from dataclasses import dataclass, field
import numpy as np

@dataclass
class Sensor:
    name:        str
    wavelengths: np.ndarray
    srf:         np.ndarray | None = None   # spectral response functions, optional
    metadata:    dict = field(default_factory=dict)