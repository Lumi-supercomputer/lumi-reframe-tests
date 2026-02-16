import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

@rfm.simple_test
class HipBandwidth(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['rocm']
    build_system = 'Make'
    sourcesdir = 'https://github.com/amd/rocm-examples'
    maintainers = ['mszpindler']
    num_tasks = 8
    num_gpus_per_node = 8
    exclusive_access = True

    binding = parameter(['closest', 'optimal'])

    tags = {'production', 'craype'}

    @run_after('init')
    def add_select_gpu_wrapper(self):
        if self.binding == 'closest':
            wrapper = 'gpu-affinity-localid.sh'
        elif self.binding == 'optimal':
            wrapper = 'gpu-affinity.sh'
        wrapper_path = os.path.join(self.current_system.resourcesdir, 'reframe_resources', 'gpu_wrappers', wrapper)
        self.prerun_cmds += [f'ln -s {wrapper_path} ./select_gpu.sh']
        

    @run_before('compile')
    def pre_compile(self):
        self.prebuild_cmds = ['cd HIP-Basic/bandwidth/']
        self.build_system.flags_from_environ = False

    @run_before('run')
    def set_exec(self):
        self.executable = './select_gpu.sh ./HIP-Basic/bandwidth/hip_bandwidth'
        if self.binding == 'closest':
            cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
            self.job.launcher.options += [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']
        self.executable_opts = ['-memory pinned -trials 100 -memcpy htod -start 1073741824 -end 1077936128']   
        self.job.launcher.options += ['--cpu-bind=verbose', '--cpus-per-task=7']

    @sanity_function
    def validate_test(self):
        num_devices = sn.count(sn.findall(r'^Device ID', self.stdout))
        return sn.assert_eq(num_devices, self.num_gpus_per_node)
