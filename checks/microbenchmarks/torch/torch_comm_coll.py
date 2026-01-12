import os
import reframe as rfm
import reframe.utility.sanity as sn


# Codes used for the check are taken from: https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/communication commit 8e4cdd8

class deepspeed_comm(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    exclusive_access = True

    perf_relative = variable(float, value=0.0, loggable=True)

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Performance of', self.stdout),
            sn.assert_eq(sn.count( sn.findall('x4', self.stdout) ), 23),
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
class torch_comm_coll_test(deepspeed_comm):
    container_platform = 'Singularity'
    
    coll_type  = parameter(['all_reduce', 'all_gather'])
    run_mode   = parameter(['native', 'torchrun'])
    cont_image = parameter([
        'rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9', #'rocm-6.2.4-python-3.12-pytorch-v2.6.0',
        'rocm-6.2.4-python-3.12-pytorch-v2.7.0-dockerhash-2a550b31226f', #'rocm-6.2.4-python-3.12-pytorch-v2.7.0',
        'rocm-6.2.4-python-3.12-pytorch-v2.7.1-dockerhash-0d479e852886', #'rocm-6.2.4-python-3.12-pytorch-v2.7.1',
    ])

    tags = {'python', 'performance'}

    allref = {
                'all_reduce': (840, -0.1, None, 'Gbps'),
                'all_gather': (775, -0.1, None, 'Gbps') 
    }

    @run_before('run')
    def set_cpu_and_task_binding(self):
        if self.run_mode == 'native':
            self.num_tasks = 16
            self.num_tasks_per_node = 8
            self.num_gpus_per_node = 8
            self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']
        if self.run_mode == 'torchrun':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_gpus_per_node = 8
            self.num_cpus_per_task = 56
        if self.coll_type == 'all_gather':
            self.extra_resources = {
               'memory': {'mem_per_node': '100G'}
           }

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = os.path.join(
            '/appl/lumi/containers/easybuild-sif-images/',
            f'lumi-pytorch-{self.cont_image}.sif',
        )
        py_script = 'communication/' + self.coll_type + '.py --scan --dist="torch"'
        if self.run_mode == 'native':
            self.container_platform.command = 'bash python-distributed.sh -u ' + py_script
        if self.run_mode == 'torchrun':
            self.container_platform.command = 'bash torch-distributed.sh ' + py_script
        self.env_vars = {
            'NCCL_DEBUG': 'WARN',
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'MASTER_ADDR': '$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)',
            'MASTER_PORT':'29500',
            f'WORLD_SIZE': self.num_tasks,
            'SINGULARITYENV_OMP_NUM_THREADS': '7',
        }

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.coll_type]
        except KeyError:
            self.skip('Missing reference value for throughput')

        self.reference = {
            '*': {
                'throughput': self.allref[self.coll_type]
            }
        }
