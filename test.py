from psd_tools import PSDImage

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)  

test = PSDImage.open("tests/psd_files/adjustments/exposure_grayscale.psd")
im = test.composite(layer_filter=lambda l: l.is_visible())
im.save("out.png")
print("guardada")