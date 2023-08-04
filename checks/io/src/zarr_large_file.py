import zarr
import numpy as np
import timeit
import os

def setup():

    global chunk_size, num_chunks, num_dsets, buf
    chunk_size = 512*1024
    num_chunks = 256
    num_dsets = 2

    buf = np.random.random(chunk_size)

def write_zarr():

    store = 'large.zip'

    with zarr.open(store, mode='w') as z:

        for n in range(num_dsets):

            dset = z.create_dataset('dset_'+str(n), shape=(num_chunks, chunk_size), dtype='f8', chunks= (1, chunk_size))

            for i in range(num_chunks):
                dset[i,:] = buf
    #z.close()

write_time = timeit.timeit(write_zarr, setup=setup, number=1)

print('write time', write_time)

#os.remove(store)
