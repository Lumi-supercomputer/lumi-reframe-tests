import h5py
import numpy as np
import timeit
import os

def setup():

    global chunk_size, num_chunks, num_dsets, buf
    chunk_size = 512*1024
    num_chunks = 256
    num_dsets = 2

    buf = np.random.random(chunk_size)

def write_h5():

    with h5py.File('large.hdf5', 'w-', driver='sec2') as hf:

        for n in range(num_dsets):

            dset = hf.create_dataset('dset_'+str(n), (num_chunks, chunk_size), dtype='f8', chunks= (1, chunk_size))

            for i in range(num_chunks):
                dset[i,:] = buf
    hf.close()

write_time = timeit.timeit(write_h5, setup=setup, number=1)

print('write time', write_time)

os.remove('large.hdf5')
