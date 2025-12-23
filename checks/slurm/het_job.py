import os
import json

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError


@rfm.simple_test
class HeterogeneousJob(rfm.RunOnlyRegressionTest):

    # Variables to control the hint and binding options on the launcher.
    valid_systems = ['lumi:gpu', 'lumi:cpu']
    valid_prog_environs = ['builtin']
    maintainers = ['mszpindler']
    modules = ['lumi-CPEtools']

    tags = {'production', 'lumi'}

    exclusive_access= True
    time_limit = 180
    omp_bind = 'cores'
    omp_proc_bind = 'close'

    ntasks = [8,2]
    nthreads = [4,16]

    @run_before('run')
    def set_het_groups(self):
        self.job.options = [f'--ntasks={self.ntasks[0]}'] 
        self.job.options += [f'--cpus-per-task={self.nthreads[0]}']
        self.job.options += ['hetjob']
        self.job.options += [f'--ntasks={self.ntasks[1]}']
        self.job.options += [f'--cpus-per-task={self.nthreads[1]}']
        self.job.options += self.current_partition.access
        self.job.launcher.options = [f'--het-group=0 --cpus-per-task={self.nthreads[0]} {self.executable} : --het-group=1 --cpus-per-task={self.nthreads[1]}']

    @run_before('run')
    def set_omp_vars(self):
        self.env_vars = {
            'OMP_PLACES': self.omp_bind,
            'OMP_PROC_BIND': self.omp_proc_bind,
        }

    @run_after('init')
    def set_executable(self):
        self.executable = 'hybrid_check'
        #self.executable_opts = ['-r']

    @sanity_function
    def check_het_groups(self):
        nranks = sn.extractsingle(r'Running\s+(\S+)\s+MPI\s+ranks\s+with\s+between\s+\S+\s+and\s+\S+\s+threads\s+.*',self.stdout, 1, int)
        #Running 10 MPI ranks with between 4 and 16 threads each
        t0 = sn.extractsingle(r'Running\s+\S+\s+MPI\s+ranks\s+with\s+between\s+(\S+)+\s+and\s+\S+\s+threads\s+.*',self.stdout, 1, int)
        t1 = sn.extractsingle(r'Running\s+\S+\s+MPI\s+ranks\s+with\s+between\s+\S+\s+and\s+(\S+)+\s+threads\s+.*',self.stdout, 1, int)
        return sn.all([
            sn.assert_eq(nranks, self.ntasks[0]+self.ntasks[1]),
            sn.assert_eq(t0, self.nthreads[0]), 
            sn.assert_eq(t1, self.nthreads[1])
        ])
