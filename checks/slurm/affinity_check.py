import os
import json

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError


class AffinityTestBase(rfm.RunOnlyRegressionTest):
    '''Base class for the affinity checks.'''

    # Variables to control the hint and binding options on the launcher.
    multithread = parameter([True, False])
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    maintainers = ['mszpindler']
    tags = {'production'}


class AffinityOpenMPBase(AffinityTestBase):
    '''Extend affinity base with OMP hooks.

    The tests derived from this class book the full node and place the
    threads acordingly based exclusively on the OMP_BIND env var. The
    number of total OMP_THREADS will vary depending on what are we
    binding the OMP threads to (e.g. if we bind to sockets, we'll have as
    many threads as sockets).
    '''

    omp_bind = variable(str)
    omp_proc_bind = variable(str, value='close')
    num_tasks = 1

    @run_before('run')
    def set_num_cpus_per_task(self):
        self.job.launcher.options = ['--cpus-per-task=$SLURM_CPUS_ON_NODE']

    @run_before('run')
    def set_omp_vars(self):
        self.variables = {
            'OMP_PLACES': self.omp_bind,
            'OMP_PROC_BIND': self.omp_proc_bind,
        }

@rfm.simple_test
class HybridCheck(AffinityOpenMPBase):

    exclusive_access= True
    omp_bind = 'cores'
    modules = ['lumi-CPEtools']

    @run_after('init')
    def set_executable(self):
        self.executable = 'hybrid_check'
        self.executable_opts = ['-r']

    @run_after('init')
    def set_threading(self):
        if self.multithread:
            self.use_multithreading = True
            self.variables['OMP_PLACES'] = 'threads' 
        else:
            self.use_multithreading = False
            self.variables['OMP_PLACES'] = 'cores' 
 

    @run_before('run')
    def set_omp_vars(self):
        self.variables['OMP_PROC_BIND'] = 'close'
            #'OMP_PLACES': 'cores',
            #'OMP_PROC_BIND': 'close',

    @sanity_function
    def check_thread_pin(self):
        nthreads = sn.extractsingle(r'Running\s+(\S+)\s+threads\s+.*',self.stdout, 1, int)  
        return sn.assert_eq(nthreads, sn.count(sn.findall(r'hybrid_check', self.stdout)))
