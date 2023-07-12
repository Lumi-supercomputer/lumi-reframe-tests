import h5py
import numpy as np
import timeit
import os
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
nranks = MPI.COMM_WORLD.size

chunk_size = 512*1024
num_chunks = 256
num_dsets = 2

buf = np.random.random(chunk_size)

start_time = timeit.default_timer()

with h5py.File('large_parallel.hdf5', 'w-', driver='mpio', comm=MPI.COMM_WORLD) as hf:

    for n in range(num_dsets):

        dset = hf.create_dataset('dset_'+str(n), (num_chunks, chunk_size), dtype='f8', chunks= (1, chunk_size))

        for i in range(0,num_chunks,nranks):
            dset[i+rank,:] = buf

hf.close()

write_time = timeit.default_timer() - start_time

if rank == 0:

    print('write time', write_time)
    os.remove('large_parallel.hdf5')
