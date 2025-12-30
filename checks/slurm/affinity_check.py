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
    valid_systems = ['lumi:gpu', 'lumi:cpu']
    valid_prog_environs = ['builtin']
    maintainers = ['mszpindler']
    tags = {'production', 'lumi'}

    exclusive_access= True
    modules = ['lumi-CPEtools']

    @run_after('init')
    def set_executable(self):
        self.executable = 'hybrid_check'
        self.executable_opts = ['-r']

    @run_before('run')
    def set_omp_vars(self):
        self.env_vars = {
            'OMP_PROC_BIND': 'close'
        }
        if self.multithread:
            self.use_multithreading = True
            self.env_vars['OMP_PLACES'] = 'threads' 
        else:
            self.use_multithreading = False
            self.env_vars['OMP_PLACES'] = 'cores' 

@rfm.simple_test
class SingleTask_Check(AffinityTaskBase):

    # Tests single MPI task pins threads to unique CPUs

    num_tasks = 1

    @run_before('run')
    def set_num_cpus_per_task(self):
        self.job.launcher.options = ['--cpus-per-task=$SLURM_CPUS_ON_NODE']

    @sanity_function
    def check_thread_pin(self):
        nthreads = sn.extractsingle(r'Running\s+(\S+)\s+threads\s+.*',self.stdout, 1, int)  
        thread_mask = sn.extractall(r'mask\s+\((?P<mask>\S+)\)', self.stdout, 'mask', int)
        return sn.assert_eq(nthreads, sn.count_uniq(thread_mask))

@rfm.simple_test
class HybridTask_Check(AffinityTaskBase):

    # Tests hybrid MPI job pins task's threads to the same NUMA domain

    dom_size = 16

    @run_before('run')
    def set_task_and_threads(self):
        if self.current_partition.name in ['gpu']:
            self.num_tasks = 8
            self.num_threads = 7 
        elif self.current_partition.name in ['cpu']:
            self.num_tasks = 16
            self.num_threads = 8 
        if self.multithread:
            self.num_threads *= 2

    @run_before('run')
    def set_num_cpus_per_task(self):
        if self.current_partition.name in ['gpu']:
            if self.multithread:
                cpu_bind_mask = '0xfe00000000000000fe000000000000,0xfe00000000000000fe00000000000000,0xfe00000000000000fe0000,0xfe00000000000000fe000000,0xfe00000000000000fe,0xfe00000000000000fe00,0xfe00000000000000fe00000000,0xfe00000000000000fe0000000000'
            else:
                cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
            self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']
        elif self.current_partition.name in ['cpu']:
            self.job.launcher.options = [f'--cpus-per-task={self.num_threads}']

    @sanity_function
    def check_thread_pin(self):
        for rank in range(self.num_tasks):
           thread_mask = sn.extractall(rf'MPI rank\s+{rank}\/{self.num_tasks}.+?mask\s+\((?P<mask>\S+)\)', self.stdout, 'mask', int)
           if self.multithread:
              thread_mask = [t % 64 for t in thread_mask]
           thread_dom = [mask // self.dom_size for mask in thread_mask]
           if not(len(thread_dom) == self.num_threads and thread_dom.count(thread_dom[0]) == len(thread_dom)):
               return False
        return True

@rfm.simple_test
class GPUPerTask_Check(AffinityTaskBase):

    # Tests for Slum CPU binding with one GPU per MPI task (without manual GPU selection) 
    # GPU-centric NUMA ordering
    # https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/#mpi-based-job

    num_gpus = 8
    num_gpus_per_node = num_gpus 
    num_tasks = num_gpus
    valid_systems = ['lumi:gpu']

    @run_after('init')
    def set_executable(self):
        self.executable = 'gpu_check'
        self.executable_opts = ['-l']

    @run_before('run')
    def set_num_gpus_per_task(self):
        self.job.options = [f'--gpus-per-task=1']

    @run_before('run')
    def set_cpu_map(self):
        self.job.launcher.options = ['--cpu-bind="map_cpu:1,9,17,25,33,41,49,57"']

    @sanity_function
    def check_cpu_gpu_numa_bind(self):
        cpu_bind = sn.extractall(r'\(CCD(?P<number>\S+)\)', self.stdout, 'number', int)

        gpu_bind = sn.extractall(r'\(GCD\S+\/CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        return sn.assert_eq(cpu_bind, gpu_bind)

@rfm.simple_test
class Hybrid_GPUSelect_Check(AffinityTaskBase):

    # Tests for Slurm CPU binding with a custom mask and GPU manual per task selection 
    # GPU-centric NUMA ordering
    # https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/#hybrid-mpiopenmp-job

    num_gpus = 8
    num_gpus_per_node = num_gpus 
    num_tasks = num_gpus
    num_threads = num_tasks - 1
    valid_systems = ['lumi:gpu']

    @run_after('init')
    def set_executable(self):
        self.executable = 'gpu_check'
        self.executable_opts = ['-l']

    @run_before('run')
    def set_cpu_mask(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

    @run_before('run')
    def select_gpu(self):
        self.prerun_cmds = ['cat << EOF > select_gpu', '#!/bin/bash', 'export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID', 'exec \$*', 'EOF']
        self.prerun_cmds += ['chmod +x ./select_gpu']
        self.executable = f'./select_gpu {self.executable}'  

    @sanity_function
    def check_cpu_gpu_numa_bind(self):
        cpu_bind = sn.extractall(r'\(CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        gpu_bind = sn.extractall(r'\(GCD\S+\/CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        return sn.assert_eq(cpu_bind, gpu_bind)

@rfm.simple_test
class Hybrid_GPUBind_Check(AffinityTaskBase):

    # Tests for Slurm CPU binding with mask and GPU binding with a custom mapping 
    # CPU-centric NUMA ordering

    num_gpus = 8
    num_gpus_per_node = num_gpus 
    num_tasks = num_gpus
    num_threads = num_tasks - 1
    valid_systems = ['lumi:gpu']

    @run_after('init')
    def set_executable(self):
        self.executable = 'gpu_check'
        self.executable_opts = ['-l']

    @run_before('run')
    def set_cpu_gpu_bind(self):
        cpu_bind_mask = '0xfe,0xfe00,0xfe0000,0xfe000000,0xfe00000000,0xfe0000000000,0xfe000000000000,0xfe00000000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']
        self.job.launcher.options += ['--gpu-bind=map_gpu:0,1,2,3,4,5,6,7']

    @sanity_function
    def check_cpu_gpu_numa_bind(self):
        cpu_bind = sn.extractall(r'\(CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        gpu_bind = sn.extractall(r'\(GCD\S+\/CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        return sn.assert_eq(cpu_bind, gpu_bind)
