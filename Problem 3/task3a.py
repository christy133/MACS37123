import numpy as np
import time
import rasterio
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

#Load bands from the provided path
with rasterio.open("/project/macs30123/landsat8/LC08_B4.tif") as band4:
    red = band4.read(1).astype('float32')
with rasterio.open("/project/macs30123/landsat8/LC08_B5.tif") as band5:
    nir = band5.read(1).astype('float32')

assert red.shape == nir.shape
shape = red.shape
size = red.size

start_cpu = time.time()
ndvi_cpu = (nir - red) / (nir + red + 1e-10)
end_cpu = time.time()

#Setup OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

red_dev = cl_array.to_device(queue, red.ravel()) 
nir_dev = cl_array.to_device(queue, nir.ravel())
ndvi_dev = cl_array.empty_like(red_dev)

# Elementwise
ndvi_kernel = ElementwiseKernel(
    ctx,
    "float *nir, float *red, float *ndvi",
    "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i] + 1e-10)", 
    "ndvi_kernel"
)

start_gpu = time.time()
ndvi_kernel(nir_dev, red_dev, ndvi_dev)
queue.finish()
end_gpu = time.time()

ndvi_gpu = ndvi_dev.get().reshape(shape)

print(f"CPU time: {end_cpu - start_cpu:.4f} seconds")
print(f"GPU time: {end_gpu - start_gpu:.4f} seconds")
mse = np.mean((ndvi_cpu - ndvi_gpu) ** 2)
print(f"Mean Squared Error between CPU and GPU NDVI: {mse:.6f}")
