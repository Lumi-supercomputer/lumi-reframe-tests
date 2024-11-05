import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

@rfm.simple_test
class MultiLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu', 'lumi:small']
    valid_prog_environs = ['builtin']
    executable = 'wait'
    num_tasks_per_node = 3
    num_nodes = 3
    num_tasks = num_nodes*num_tasks_per_node
    # Required to mitigate error-configuring-interconnect
    exclusive_access = True

    tags = {'production', 'lumi'}

    @run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        background_cmd = 'hostname'
        self.prerun_cmds = [
            f'{cmd} --overlap --exact -N {self.num_nodes} -n {self.num_tasks_per_node} {background_cmd} &'
            for n in range(1, self.num_nodes+1)
        ]

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def validate_test(self):
        return sn.assert_eq(
            sn.count(sn.extractall(r'^nid\d+', self.stdout)), self.num_tasks
        )

@rfm.simple_test
class MultiLaunchGPUTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    executable = 'wait'
    num_tasks_per_node = 1
    num_gpus_per_node = 8
    num_nodes = 1
    num_tasks = 8
    exclusive_access = True
    modules = ['lumi-CPEtools']

    tags = {'production', 'lumi'}

    @run_after('init')
    def add_select_gpu_wrapper(self):
        self.prerun_cmds += [
            'cat << EOF > select_step_gpu',
            '#!/bin/bash',
            'CPU_MAP="49-55 57-63 17-23 25-31 1-7 9-15 33-39 41-47"',
            'CPUS=(\${CPU_MAP})',
            'export ROCR_VISIBLE_DEVICES=\$SLURM_STEPID',
            'exec numactl --physcpubind=\${CPUS[\$SLURM_STEPID]} \$*',
            'EOF',
            'chmod +x ./select_step_gpu',
        ]

    @run_before('run')
    def pre_launch(self):
        self.job.options += ['--cpus-per-task=56']
        cmd = self.job.launcher.run_command(self.job)
        background_cmd = 'gpu_check -l'
        self.prerun_cmds += [
            f'{cmd} -n 1 --overlap --exact ./select_step_gpu {background_cmd} &'
            for n in range(0, self.num_tasks-1)
        ]

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def check_cpu_gpu_numa_bind(self):
        cpu_bind = sn.extractall(r'\(CCD(?P<number>\S+)\)', self.stdout, 'number', int)

        gpu_bind = sn.extractall(r'\(GCD\S+\/CCD(?P<number>\S+)\)', self.stdout, 'number', int)
        return sn.assert_eq(cpu_bind, gpu_bind)
