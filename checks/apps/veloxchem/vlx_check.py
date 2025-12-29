import reframe as rfm
import reframe.utility.sanity as sn

class vlx_check(rfm.RunOnlyRegressionTest):
    modules = ['VeloxChem']
    executable = 'vlx'
    output_file = 'vlx.out'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']
    maintainers = ['mszpindler']
    descr = 'VeloxChem GPU test'
    exclusive_access = True
    keep_files = [output_file]
    
    energy_tolerance = 1.0e-12
    
    @run_before('run')
    def setup_job(self):
        self.num_gpus_per_node = 8
        self.num_tasks_per_node = 1
        self.job.launcher.options = ['--cpus-per-task=56']
        self.env_vars = {
            'SRUN_CPUS_PER_TASK': self.num_gpus_per_node,
            'OMP_NUM_THREADS': self.num_gpus_per_node,
            'KMP_AFFINITY': 'verbose,proclist=[49,57,17,25,1,9,33,41],explicit',
        }

    @sanity_function
    def assert_simulation_success(self):
        energy = sn.extractsingle(
                r'\s+Total Energy\s+\:\s+(?P<energy>\S+) a.u.',
                self.output_file,
                'energy',
                float
        )
        energy_diff = sn.abs((energy-self.energy_reference)/self.energy_reference)
        return sn.all([
            sn.assert_found(f'SCF converged in {self.num_iter} iterations', self.output_file),
            sn.assert_found(r'!\s+VeloxChem execution completed', self.output_file),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'!\s+Total execution time is\s+(?P<wtime>\S+)',
                                self.output_file, 'wtime', float)


@rfm.simple_test
class veloxchem_g_quad(vlx_check):
    num_tasks = 8
    time_limit = '13m'

    references = {
        'lumi:gpu': {
            'time': (745, None, 0.05, 's')
        }
    }

    energy_reference = -30517.3986504615
    num_iter = 11

    @run_after('init')
    def setup_run(self):
        job_name = 'g-quad-neutral'
        input_file = f'{job_name}.inp'
        self.executable_opts = [input_file, self.output_file]

@rfm.simple_test
class veloxchem_bp_6(vlx_check):
    num_tasks = 1
    time_limit = '10m'

    references = {
        'lumi:gpu': {
            'time': (575, None, 0.05, 's')
        }
    }

    energy_reference = -16618.0888623844
    num_iter = 6

    @run_after('init')
    def setup_run(self):
        job_name = 'bp-6'
        input_file = f'{job_name}.inp'
        self.executable_opts = [input_file, self.output_file]
