import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

# This tries to fill up a node with multiple sub-job steps
# launched with multiple srun calls interleaved
@rfm.simple_test
class MultiLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:small', 'lumi:gpu']
    valid_prog_environs = ['builtin']
    executable = 'wait'
    num_nodes = 1
    exclusive_access = True
    keep_files = ['rfm_step_*.out']

    tags = {'production', 'lumi'}

    @run_before('run')
    def job_alloc_opts(self):
        self.job.options = ['--distribution=block:block:block'] # block distribution across CCDs
        if self.current_partition.name in ['gpu']:
            self.job.options += ['--gpu-bind=closest'] #, '--gpus-per-node=8']

    @run_before('run')
    def step_alloc_opts(self):
        if self.current_partition.name in ['gpu']:
            self.num_tasks_per_node = 7 # occupy one CCD (7 cores, one disabled) per step
            num_cpus = 56
        else:
            self.num_tasks_per_node = 8 # occupy one CCD (8 cores) per step
            num_cpus = 128
        self.num_steps = num_cpus // self.num_tasks_per_node
        self.num_tasks = self.num_nodes*self.num_tasks_per_node
        cmd = self.job.launcher.run_command(self.job)
        background_cmd = "bash -c 'echo $(hostname):$(taskset -pc $$ | cut -d: -f2); sleep 5'"
        self.prerun_cmds = [
            f'{cmd} --exact --cpu-bind=verbose --output=rfm_step_%s.out -N {self.num_nodes} -n {self.num_tasks_per_node} {background_cmd} &'
            for n in range(self.num_steps)
        ]

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def validate_test(self):
        step_bindings = []
        for s in range(self.num_steps):
             step_bindings += sn.extractall(r'nid(?P<step_bindings>\S+:\s\S+)', f'rfm_step_{s}.out')
        return sn.assert_eq(
            sn.count_uniq(step_bindings), self.num_tasks*self.num_steps, 'Step bindings are overlapping'
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

# This tries to overcommit nodes with multiple job steps
# launched with multiple srun calls overlapped
@rfm.simple_test
class OverlapLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:small']
    valid_prog_environs = ['builtin']
    executable = 'wait'
    num_tasks_per_node = 1
    cpus_per_task = 128
    num_nodes = 2
    num_tasks = num_nodes*num_tasks_per_node
    num_steps = 2
    exclusive_access = True

    tags = {'production', 'lumi'}

    @run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        background_cmd = './wrap.sh'
        self.prerun_cmds = [
            f'{cmd} --overlap {background_cmd} &'
            for n in range(1, self.num_steps+1)
        ]

    @run_before('run')
    def set_vni_opts(self):
        self.job.options += ['--overcommit']
        self.job.options += ['--network=job_vni,def_acs=1']

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def validate_test(self):
        return sn.assert_eq(
            sn.count(sn.extractall(r'nid\d+', self.stdout)), self.num_tasks*self.num_steps
        )
