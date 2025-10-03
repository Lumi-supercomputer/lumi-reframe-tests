import os
import reframe as rfm
import reframe.utility.sanity as sn


# Codes used for the check are taken from: https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/communication commit 8e4cdd8

class deepspeed_comm(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    exclusive_access = True
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8

    perf_relative = variable(float, value=0.0, loggable=True)

    reference = {
        'lumi:gpu': {
            'throughput': (815, -0.1, None, 'Gbps'),
            # We are interested in a single variable per test
            #'duration': (10.5, -0.1, None, 'ms')
        }
    }

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using network AWS Libfabric', self.stdout),
            sn.assert_found(r'Selected Provider is cxi', self.stdout),
            sn.assert_found(r'Performance of', self.stdout)  
        ])

    #@performance_function('ms')
    #def duration(self):
    #    return sn.extractsingle(
    #        r'512\.0 MB\s+\S+\s+(?P<duration>\S+)\s+ms',
    #        self.stdout, 'duration', float
    #    )

    @performance_function('Gbps')
    def throughput(self):
        return sn.extractsingle(
            r'512\.0 MB\s+\S+\s+\S+\s+\S+\s+(?P<throughput>\S+)\s+\S+',
            self.stdout, 'throughput', float
        )

    @run_after('performance')
    def higher_the_better(self):
        perf_var = 'throughput'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

@rfm.simple_test
class ds_comm_all_reduce(deepspeed_comm):
    modules = ['PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410']
    dist_mode = parameter(['deepspeed', 'torch'])

    tags = {'python', 'contrib'}

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = '$SIFPYTORCH'
        self.container_platform.command = f'bash conda-python-distributed.sh -u communication/all_reduce.py --scan --dist="{self.dist_mode}"'
        self.container_platform.env_vars = {'NCCL_DEBUG': 'INFO'}

