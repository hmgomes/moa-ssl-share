import numpy as np
import river.stream as rs
from pyssl.skmf import convert_river_to_skmf
from skmultiflow.data.file_stream import FileStream
from conftest import WEATHER_CSV

def test_conversion():
    skmf_stream = FileStream(WEATHER_CSV)
    river_stream = rs.iter_csv(WEATHER_CSV, target="target")

    i = 0
    while skmf_stream.has_more_samples() and i < 1_000:
        skmf_x, skmf_y = skmf_stream.next_sample()
        river_x, river_y = next(river_stream)
        x, y  = convert_river_to_skmf(river_x, river_y)
        i += 1
    
        assert np.allclose(x, skmf_x)
        assert y == skmf_y
