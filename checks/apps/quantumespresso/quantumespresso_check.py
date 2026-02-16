import reframe as rfm
import reframe.utility.sanity as sn


class quantumespresso_check(rfm.RunOnlyRegressionTest):
    mode = parameter(['mpi', 'mpi_omp'])
    modules = ['QuantumESPRESSO']
    executable = 'pw.x'
    executable_opts += ['-in', 'ausurf.in', '-pd', '.true.']
    maintainers = ['mszpindler']

    tags = {'contrib', 'performance'}

    perf_relative = variable(float, value=0.0, loggable=True)

    @sanity_function
    def assert_simulation_success(self):
        energy = sn.extractsingle(
                # It seems as QE prefixes the final total energy with a `!`.
                r'\s*!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                self.stdout,
                'energy',
                float
        )
        energy_diff = sn.abs(energy-self.energy_reference)
        return sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_found(r'\s+JOB DONE', self.stdout),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                self.stdout, 'wtime', float)

    @run_after('performance')
    def lower_the_better(self):
        perf_var = 'time'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][1]-self.perfvalues[key_str][0])/self.perfvalues[key_str][1])

@rfm.simple_test
class lumi_quantumespresso_cpu_check(quantumespresso_check):
    valid_systems = ['lumi:cpu']
    valid_prog_environs = ['cpeGNU']
    descr = f'QuantumESPRESSO CPU check'
    time_limit = '15m'

    # It seems to be independent of the number of nodes/ranks.
    energy_reference = -11427.09017152
    energy_tolerance = 1.0e-6

    @run_after('init')
    def setup_test(self):
        self.descr = f'QuantumESPRESSO CPU check mode "{self.mode}"'

        # In the CSCS test the option `num_tasks_per_core` is defined, however,
        #  on LUMI this leads to a quite significant drop in performance.
        if self.mode == 'mpi':
            self.num_tasks = 256
            self.num_tasks_per_node = 128
            self.num_cpus_per_task = 1

        else:
            self.num_tasks = 128
            self.num_tasks_per_node = 64
            self.num_cpus_per_task = 2
            self.env_vars = {
                'OMP_NUM_THREADS': self.num_cpus_per_task,
            }

    @run_before('performance')
    def set_perf_reference(self):
        references = {
            'mpi': {
                'lumi:cpu': {
                    'time': (75.0, None, 0.05, 's')
                },
            },
            'mpi_omp': {
                'lumi:cpu': {
                    'time': (56.00, None, 0.05, 's')
                },
            },
        }
        self.reference = references[self.mode]
