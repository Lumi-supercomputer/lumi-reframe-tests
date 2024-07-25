import os
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class OpenMPRuntime(rfm.RunOnlyRegressionTest):
    descr = ''
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU', 'cpeCray', 'cpeAMD']
    num_tasks_per_node = 1
    num_cpus_per_task = 7
    lumi_cpe_tools_ver = '1.1'
    executable = 'omp_check'
    use_multithreading= None

    maintainers = ['mszpindler']
    tags = {'craype'}

    @run_after('init')
    def set_executable(self):
        self.executable_opts = ['-r']

    @run_before('run')
    def set_module(self):
        lumi_stack_ver = os.getenv('LUMI_STACK_VERSION')
        self.modules = [f'lumi-CPEtools/{self.lumi_cpe_tools_ver}-{self.current_environ.name}-{lumi_stack_ver}']


    @run_before('run')
    def set_omp_vars(self):
        self.env_vars = {
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
            'OMP_DISPLAY_ENV': 'verbose',
        }

    @sanity_function
    def validate_omp_env(self):
        a = sn.count(sn.findall('OMP_', self.stdout))
        b = sn.count(sn.findall('OMP_', self.stderr))
        return sn.assert_gt(a+b, 0)
