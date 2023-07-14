import os
import json

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError


@rfm.simple_test
class HeterogeneousJob(rfm.RunOnlyRegressionTest):

    # Variables to control the hint and binding options on the launcher.
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    maintainers = ['mszpindler']
    modules = ['lumi-CPEtools']
    tags = {'production'}

    exclusive_access= False
    time_limit = 180
    omp_bind = 'cores'
    omp_proc_bind = 'close'

    @run_before('run')
    def set_het_groups(self):
        self.job.options = ['--cpus-per-task=2 --ntasks-per-node=8'] 
        self.job.options += ['hetjob']
        self.job.options += ['--cpus-per-task=8 --ntasks-per-node=2']
        self.job.options += self.current_partition.access

    @run_before('run')
    def set_omp_vars(self):
        self.variables = {
            'OMP_PLACES': self.omp_bind,
            'OMP_PROC_BIND': self.omp_proc_bind,
        }

    @run_after('init')
    def set_executable(self):
        self.executable = 'hybrid_check'
        #self.executable_opts = ['-r']

    @run_before('run')
    def set_het_launch_opts(self):
        self.job.launcher.options = ['--het-group=0,1']

    @sanity_function
    def check_het_groups(self):
        ngroups = sn.extractall(r'Running\s+(\S+)\s+threads\s+.*',self.stdout, 1, int)  
        return sn.assert_eq(sn.count(ngroups), 2)
