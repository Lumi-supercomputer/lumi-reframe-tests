import reframe as rfm
import reframe.utility.sanity as sn


class quantumespresso_check(rfm.RunOnlyRegressionTest):
    modules = ['QuantumESPRESSO']
    executable = 'pw.x'
    executable_opts += ['-in', 'ausurf.in', '-pd', '.true.']
    maintainers = ['mszpindler']

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


@rfm.simple_test
class lumi_quantumespresso_cpu_check(quantumespresso_check):
    valid_systems = ['lumi:small']
    valid_prog_environs = ['cpeGNU']
    descr = f'QuantumESPRESSO CPU check'
    num_tasks = 256
    num_tasks_per_node = 128
    num_cpus_per_task = 1
    time_limit = '15m'

    # It seems that the reference value of the energy also depends on the node
    #  configuration. The value selected below is the one used by CSCS in their
    #  "large" configuration, which also has 256 nodes.
    energy_reference = -11427.09017152
    energy_tolerance = 1.0e-6

    # The time was obtained by running it multiple times and then compute
    #  the average over these runs.
    reference = {
        'lumi:small': {'time': (56.16, None, 0.05, 's')},
    } 
