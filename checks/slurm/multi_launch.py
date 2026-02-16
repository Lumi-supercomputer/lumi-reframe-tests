import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

# This tries to fill up a node with multiple sub-job steps
# launched with multiple srun calls interleaved
class SlurmMultiLaunch(rfm.RunOnlyRegressionTest):
    executable = 'wait'
    num_nodes = 1
    exclusive_access = True
    use_multithreading = False
    keep_files = ['rfm_step_*.out']
    tags = {'production', 'lumi'}

    @run_before('run')
    def job_alloc_opts(self):
        self.job.options = ['--distribution=block:block:block'] # block distribution across CCDs

    @sanity_function
    def validate_test(self):
        step_bindings = []
        for s in range(self.num_steps):
             step_bindings += sn.extractall(r'nid(?P<step_bindings>\S+:\s\S+)', f'rfm_step_{s}.out')
        return sn.assert_eq(
            sn.count_uniq(step_bindings), self.num_tasks_per_node*self.num_steps, 'Step bindings are overlapping'
        )


@rfm.simple_test
class MultiLaunchTest(SlurmMultiLaunch):
    valid_systems = ['lumi:cpu']
    valid_prog_environs = ['builtin']

    @run_before('run')
    def step_alloc_opts(self):
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

@rfm.simple_test
class MultiLaunchGPUTest(SlurmMultiLaunch):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']

    modules = ['lumi-CPEtools']

    @run_before('run')
    def job_gpu_alloc_opts(self):
        self.job.options += ['--gpus-per-task=8'] 

    @run_before('run')
    def step_alloc_opts(self):
        self.num_tasks_per_node = 1 
        num_cpus = 56 # occupy eight CCDs (7 cores, one disabled), mask cores per step
        self.num_cpus_per_step = 7 
        self.num_cpus_per_task = num_cpus

        self.num_steps = num_cpus // self.num_cpus_per_step
        self.num_tasks = self.num_nodes*self.num_tasks_per_node
        cmd = self.job.launcher.run_command(self.job)

        binding_mask = [
            '0xfe000000000000',
            '0xfe00000000000000',
            '0xfe0000',
            '0xfe000000',
            '0xfe',
            '0xfe00',
            '0xfe00000000',
            '0xfe0000000000'  
        ]
        for n in range(self.num_steps):
            background_cmd = f"bash -c 'echo $(hostname):$(taskset -pc $$ | cut -d: -f2); export ROCR_VISIBLE_DEVICES={n}; gpu_check -l; sleep 5'"
            self.prerun_cmds += [
                f'{cmd} --overlap --exact --cpu-bind=verbose,mask_cpu:{binding_mask[n]} --output=rfm_step_%s.out -N {self.num_nodes} -n {self.num_tasks_per_node} {background_cmd} &'
            ]

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

# This tries to overcommit nodes with multiple job steps
# launched with multiple srun calls overlapped
@rfm.simple_test
class OverlapLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:cpu']
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
