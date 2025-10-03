# This script derives from CSCS NAMD test
# available at:
# https://github.com/reframe-hpc/cscs-reframe-tests/blob/main/checks/apps/namd/namd_check.py

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class lumi_namd_stmv(rfm.RunOnlyRegressionTest):

    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']

    gpu_mode = parameter(['offload', 'resident'])

    use_multithreading = False
    exclusive_access = True

    num_nodes = parameter([1,2], loggable=True)
    num_gpus_per_node = 8
    time_limit = '10m'

    executable = 'namd3'
    tags = {'benchmark', 'contrib', 'gpu', 'performance'}

    perf_relative = variable(float, value=0.0, loggable=True)
    allref = {
        1: {
            'resident': (40.0, -0.05, None, 'ns/day'), # gpu resident mode
            'offload': (8.0, -0.05, None, 'ns/day'),   # offload mode
        },
        2: {
            'offload': (10.0, -0.05, None, 'ns/day'),   # offload mode
        },
    }

    @run_after('init')
    def set_module_environ(self):
        match self.gpu_mode:
            case 'resident':
                self.modules = ['NAMD/3.0.2-cpeGNU-24.03-rocm-gpu-resident']
            case 'offload':
                self.modules = ['NAMD/3.0.2-cpeGNU-24.03-rocm-gpu-offload'] 

    @run_before('run')
    def prepare_test(self):
        self.desc = "NAMD STMV benchmark"
        bench_dir_path= os.path.join(self.current_system.resourcesdir,
                                       'datasets', 'namd', 'stmv_gpu')
        self.prerun_cmds += [
            f'ln -s {bench_dir_path} .'
        ]

    @run_after('init')
    def setup_runtime(self):
        if self.gpu_mode == 'resident':
            self.executable_opts = ['+p56', '+pmepes 1', '+setcpuaffinity', '+devices 4,5,2,3,6,7,0,1', 'stmv_gpures_npt.namd']
            self.num_cpus_per_task = 56
            self.num_tasks = 1
        elif self.gpu_mode == 'offload':
            #self.executable_opts = ['+ignoresharing', '+devices 0', '+p 6', 'stmv_gpuoff_npt.namd']
            self.executable_opts = ['+p6', '+devices 4,5,2,3,6,7,0,1', 'stmv_gpuoff_npt.namd']
            self.num_cpus_per_task = 7
            self.num_tasks_per_node = 8
            self.num_tasks = self.num_nodes*self.num_tasks_per_node

    #@run_before('run')
    #def set_cpu_mask(self):
    #    cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
    #    self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

    #@run_before('run')
    #def add_select_gpu_wrapper(self):
    #    self.prerun_cmds += [
    #        'cat << EOF > select_gpu',
    #        '#!/bin/bash',
    #        'export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID',
    #        'exec \$*',
    #        'EOF',
    #        'chmod +x ./select_gpu'
    #    ]
    #    self.executable = './select_gpu ' + self.executable

    @sanity_function
    def validate_energy(self):
        energy = sn.avg(sn.extractall(
            r'ENERGY:([ \t]+\S+){10}[ \t]+(?P<energy>\S+)',
            self.stdout, 'energy', float)
        )
        energy_reference = -2801000.0
        energy_diff = sn.abs(energy - energy_reference)
        return sn.all([
            sn.assert_found(r'\S+\s+End of program', self.stdout),
            sn.assert_lt(energy_diff, 1999)
        ])

    @performance_function('ns/day')
    def perf(self):
        return sn.avg(sn.extractall(
            r'Info: Benchmark time: \S+ CPUs \S+ s/step (?P<ns_per_day>\S+) ns/day \S+ MB memory',
            self.stdout, 'ns_per_day', float))

    @run_after('performance')
    def higher_the_better(self):
        perf_var = 'perf'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

    @run_after('init')
    def setup_run(self):
        try: 
            found = self.allref[self.num_nodes][self.gpu_mode]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) and GPU '
                      f'{self.gpu_mode!r} mode is not supported')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.gpu_mode]
            }
        }
