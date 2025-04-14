import numpy as np
import time
import rasterio
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

with rasterio.open("/project/macs30123/landsat8/LC08_B4.tif") as band4:
    red_orig = band4.read(1).astype('float32')
with rasterio.open("/project/macs30123/landsat8/LC08_B5.tif") as band5:
    nir_orig = band5.read(1).astype('float32')

assert red_orig.shape == nir_orig.shape

#OpenCL context and kernel
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

ndvi_kernel = ElementwiseKernel(
    ctx,
    "float *nir, float *red, float *ndvi",
    "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i] + 1e-10)",
    "ndvi_kernel"
)

#Scaling factors
scales = [20, 50, 100, 150]

print(f"{'Scale':<10}{'Shape':<20}{'CPU Time (s)':<15}{'GPU Time (s)':<15}")

for scale in scales:
    #Tile image to simulate larger workloads
    red = np.tile(red_orig, (scale, 1))
    nir = np.tile(nir_orig, (scale, 1))
    shape = red.shape

    #CPU
    start_cpu = time.time()
    ndvi_cpu = (nir - red) / (nir + red + 1e-10)
    end_cpu = time.time()

    #GPU
    red_flat = red.ravel()
    nir_flat = nir.ravel()
    red_dev = cl_array.to_device(queue, red_flat)
    nir_dev = cl_array.to_device(queue, nir_flat)
    ndvi_dev = cl_array.empty_like(red_dev)

    start_gpu = time.time()
    ndvi_kernel(nir_dev, red_dev, ndvi_dev)
    queue.finish()
    end_gpu = time.time()

    print(f"{scale:<10}{str(shape):<20}{end_cpu - start_cpu:<15.4f}{end_gpu - start_gpu:<15.4f}")
