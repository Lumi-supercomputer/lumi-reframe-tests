import os
import json

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError


class AffinityTaskBase(rfm.RunOnlyRegressionTest):
    '''Base class for the affinity checks.'''

    # Variables to control the hint and binding options on the launcher.
    multithread = parameter([True, False])
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    maintainers = ['mszpindler']
    tags = {'production'}

    exclusive_access= True
    modules = ['lumi-CPEtools']

    @run_after('init')
    def set_executable(self):
        self.executable = 'hybrid_check'
        self.executable_opts = ['-r']

    @run_before('run')
    def set_omp_vars(self):
        self.variables = {
            'OMP_PROC_BIND': 'close'
        }
        if self.multithread:
            self.use_multithreading = True
            self.variables['OMP_PLACES'] = 'threads' 
        else:
            self.use_multithreading = False
            self.variables['OMP_PLACES'] = 'cores' 

@rfm.simple_test
class SingleTask(AffinityTaskBase):

    num_tasks = 1

    @run_before('run')
    def set_num_cpus_per_task(self):
        self.job.launcher.options = ['--cpus-per-task=$SLURM_CPUS_ON_NODE']

 
    @sanity_function
    def check_thread_pin(self):
        nthreads = sn.extractsingle(r'Running\s+(\S+)\s+threads\s+.*',self.stdout, 1, int)  
        return sn.assert_eq(nthreads, sn.count(sn.findall(r'hybrid_check', self.stdout)))

@rfm.simple_test
class TaskPerGPU(AffinityTaskBase):

    num_gpus = 8
    num_gpus_per_node = num_gpus 
    num_tasks = num_gpus
    num_threads = num_tasks - 1

    @run_before('run')
    def set_num_cpus_per_task(self):
        self.job.launcher.options = [f'--cpus-per-task={self.num_threads}']

    @sanity_function
    def check_thread_pin(self):
        nranks = sn.extractsingle(r'Running\s+(\S+)\s+MPI\s+ranks\s+.*',self.stdout, 1, int)  
        nthreads = sn.extractsingle(r'Running\s+\S+\s+MPI\s+ranks\s+with\s+(\S+)\s+.*',self.stdout, 1, int)
        return sn.assert_eq(nranks, self.num_tasks) 

@rfm.simple_test
class TaskPerGPU_CPUMap(TaskPerGPU):

    @run_before('run')
    def set_cpu_mask(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']
