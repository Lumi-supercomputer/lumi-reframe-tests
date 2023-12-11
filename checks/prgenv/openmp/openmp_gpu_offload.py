import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class vAdd_ompGPU(rfm.RegressionTest):
    descr = 'Checks OpenMP target offload on GPU'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-amd', 'ROCm']
    modules = ['rocm']
    num_gpus_per_node = 1
    build_system = 'Make'
    sourcesdir = 'https://code.ornl.gov/olcf/vector_addition.git'
    prebuild_cmds = ['cd ompGPU/']
    executable = 'ompGPU/vAdd_ompGPU'

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}

    @run_before('compile')
    def set_compiler_opts(self):
        self.build_system.cflags = ['-fopenmp']

    @run_before('run')
    def set_env_vars(self):
        if self.current_environ.name == 'PrgEnv-cray':
            self.env_vars = {'CRAY_ACC_DEBUG': '2',} # 'LD_LIBRARY_PATH': '$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH'}

    @sanity_function
    def validate_solution(self):
        return sn.extractsingle(r'passed', self.stdout)
