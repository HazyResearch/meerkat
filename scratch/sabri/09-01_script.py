import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))


import meerkat as mk

network = mk.interactive_mode()


dp = mk.get("rfw")
dp.gui.table(id_column="image_id")
