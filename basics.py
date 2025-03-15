import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    no_preview = True # deactivate preview
except:
    if os.getenv("SIONNA_NO_PREVIEW"):
        no_preview = True
    else:
        no_preview = False

resolution = [480,320] # increase for higher quality of renderings

# Define magic cell command to skip a cell if needed
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

sionna.config.seed = 42




import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths, RIS, r_hat, normalize
from sionna import PI

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement


# Load integrated scene
scene = load_scene("oulu_downtown_scene/untitled.xml") # Try also sionna.rt.scene.etoile
scene.frequency = 3e9 # Carrier frequency [Hz]
scene.frequency = 3e9 # Carrier frequency [Hz]
scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso", "V")
scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso", "V")

# Place a transmitter 
tx = Transmitter("tx", position=[-25,35,12])
scene.add(tx)

# Place receivers
rx1 = Receiver("rx1", position=[60,-50,3])
scene.add(rx1)
rx2 = Receiver("rx2", position=[109,-43,3])
scene.add(rx2)

# Place RIS
ris1 = RIS(name="ris1",
          position=[64,55,30],
          num_rows=100,
          num_cols=100,
          num_modes=2,
          look_at=(tx.position+rx1.position)/2) # Look in between TX and RX1
scene.add(ris1)

ris2 = RIS(name="ris2",
          position=[-15,-50,20],
          num_rows=100,
          num_cols=100,
          num_modes=2,
          look_at=(tx.position+rx2.position)/2) # Look in between TX and RX2
scene.add(ris2)


if scene.get("cam") is None:
    scene.add(Camera("cam",
                        position=[50,-50,130],
                        look_at=[0,0,0]))
scene.render(camera="cam", num_samples=512)



scene.preview(show_orientations=True)