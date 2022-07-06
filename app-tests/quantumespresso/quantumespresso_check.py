import reframe as rfm
import reframe.utility.sanity as sn


class quantumespresso_check(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    pe_release = '21.12'
    qe_version = '7.0`'
    modules = ['QuantumESPRESSO-{qe_version}']
    executable = 'pw.x'
    strict_check = False
    maintainers = ['mszpindler']

    @run_after('init')
    def prepare_test(self):
        self.prerun_cmds = [
            f'curl -LJO https://raw.githubusercontent.com/QEF/benchmarks/master/AUSURF112/ausurf.in',
            f'curl -LJO https://raw.githubusercontent.com/QEF/benchmarks/master/AUSURF112/Au.pbe-nd-van.UPF'
        ]
        self.executable_opts += ['-in', 'ausurf.in', '-pd', '.true.']

    @run_after('init')
    def set_prog_envs_and_tags(self):
        if self.current_system.name in ['lumi']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']


    @sanity_function
    def assert_simulation_success(self):
        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-self.energy_reference)
        return sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                self.stdout, 'wtime', float)


@rfm.simple_test
class lumi_quantumespresso_cpu_check(quantumespresso_check):
    energy_tolerance = 1.0e-6

    @run_after('init')
    def setup_test(self):
        self.descr = (f'QuantumESPRESSO CPU check (version: {self.scale})')
        if self.scale == 'small':
            self.valid_systems = ['lumi:cpu']
            self.energy_reference = -11427.09017218
            if self.current_system.name in ['lumi']:
                self.num_tasks = 256
                self.num_tasks_per_node = 128
                self.num_cpus_per_task = 1
        else:
            self.energy_reference = -11427.09017152
            self.valid_systems = []

    @run_before('performance')
    def set_reference(self):
        # Fix reference values
        references = {
            'small': {
                'dom:mc': {'time': (110.0, None, 0.05, 's')},
                'daint:mc': {'time': (127.0, None, 0.10, 's')},
                'eiger:mc': {'time': (66.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (66.0, None, 0.10, 's')}
            },
            'large': {
                'daint:mc': {'time': (171.0, None, 0.10, 's')},
                'eiger:mc': {'time': (53.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (53.0, None, 0.10, 's')}
            }
        }
        self.reference = references[self.scale]

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    #@run_before('run')
    #def set_cpu_binding(self):
        #self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class lumi_quantumespresso_cpu_check(quantumespresso_check):
    # Fix it: GPU enabled CP2K instance (module)
    valid_systems = ['lumi:gpu']
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12
    energy_tolerance = 1.0e-7

    @run_after('init')
    def setup_test(self):
        self.descr = (f'QuantumESPRESSO GPU check (version: {self.scale})')
        if self.scale == 'small':
            self.valid_systems += ['lumi:gpu']
            self.num_tasks = 4
            self.energy_reference = -11427.09017168
        else:
            self.num_tasks = 8
            self.energy_reference = -11427.09017179

    @run_before('performance')
    def set_reference(self):
        # Fix reference values
        references = {
            'small': {
                'dom:gpu': {'time': (59.0, None, 0.05, 's')},
                'daint:gpu': {'time': (59.0, None, 0.05, 's')}
            },
            'large': {
                'daint:gpu': {'time': (40.0, None, 0.05, 's')}
            }
        }
        self.reference = references[self.scale]
