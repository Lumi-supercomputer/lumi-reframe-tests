import reframe as rfm
import reframe.utility.sanity as sn


class quantumespresso_check(rfm.RunOnlyRegressionTest):
    modules = ['QuantumESPRESSO']
    executable = 'pw.x'
    executable_opts += ['-in', 'ausurf.in', '-pd', '.true.']
    maintainers = ['mszpindler']

    @sanity_function
    def assert_simulation_success(self):
        energy = sn.extractsingle(r'total energy\s+\=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-self.energy_reference)
        return sn.all([
            sn.assert_found(r'\s+JOB DONE', self.stdout),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                self.stdout, 'wtime', float)


@rfm.simple_test
class lumi_quantumespresso_cpu_check(quantumespresso_check):
    valid_systems = ['lumi:small']
    valid_prog_environs = ['cpeGNU']
    descr = f'QuantumESPRESSO CPU check'
    num_tasks = 256
    num_tasks_per_node = 128
    num_cpus_per_task = 1
    time_limit = '15m'

    energy_tolerance = 1.0e-6
    energy_reference = -11423.49032755
    reference = {
        'lumi:small': {'time': (110.0, None, 0.05, 's')},
    } 
