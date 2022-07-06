import reframe as rfm
import reframe.utility.sanity as sn


class cp2k_check(rfm.RunOnlyRegressionTest):
    executable = 'cp2k.psmp'
    maintainers = ['mszpindler']
    strict_check = False
    pe_release = '21.08'
    cp2k_version = '9.1'
    modules = ['CP2K-{cp2k_version}']

    @run_after('init')
    def set_input(self):
        self.prerun_cmds = [
                f'curl -LJO https://raw.githubusercontent.com/cp2k/cp2k/master/benchmarks/QS/H2O-256.inp'
            ]
        self.executable_opts = ['H2O-256.inp']

    @run_after('init')
    def set_prgenv(self):
        if self.current_system.name in ['lumi']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @sanity_function
    def assert_energy_diff(self):
        energy = sn.extractsingle(
            r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
            r'energy [\[\(]a\.u\.[\]\)]:\s+(?P<energy>\S+)',
            self.stdout, 'energy', float, item=-1
        )
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        return sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?i)(?P<step_count>STEP NUMBER)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                self.stdout, 'perf', float)


@rfm.simple_test
class lumi_cp2k_cpu_check(cp2k_check):
    scale = parameter(['small'])
    valid_systems = ['lumi:cpu', 'lumi:gpu']
    refs_by_scale = {
        'small': {
            'lumi:cpu': {'time': (152.644, None, 0.05, 's')},
        },
        'large': {
            #
        }
    }

    @run_after('init')
    def setup_by_scale(self):
        self.descr = f'CP2K CPU check (version: {self.scale})'
        self.tags |= {'maintenance', 'production'}
        if self.scale == 'small':
            if self.current_system.name in ['lumi']:
                self.num_tasks = 256
                self.num_tasks_per_node = 128

        self.reference = self.refs_by_scale[self.scale]

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']
        self.job.options = ['--time=5']

@rfm.simple_test
class lumi_cp2k_gpu_check(cp2k_check):
    # Fix it: GPU enabled CP2K instance (module)
    scale = parameter(['small'])
    valid_systems = ['lumi:gpu']
    refs_by_scale = {
        'small': {
            'lumi:gpu': {'time': (182.0, None, 0.05, 's')},
        },
        'large': {
            #
        }
    }

    @run_after('init')
    def setup_by_scale(self):
        self.descr = f'CP2K GPU check (version: {self.scale})'
        if self.scale == 'small':
            if self.current_system.name in ['lumi']:
                self.num_tasks = 64
                self.num_tasks_per_node = 64
                self.num_gpus_per_node = 1
                self.num_cpus_per_task = 2
                self.variables = {
                    'OMP_NUM_THREADS': str(self.num_cpus_per_task)
                }

        self.reference = self.refs_by_scale[self.scale]
        self.tags |= {'maintenance', 'production'}
