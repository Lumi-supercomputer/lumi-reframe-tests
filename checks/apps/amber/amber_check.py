import os
import contextlib
import reframe as rfm
import reframe.utility.sanity as sn

class amber_nve20_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Amber NVE test.

    `Amber <https://ambermd.org/>`__ is a suite of biomolecular simulation
    programs. It began in the late 1970's, and is maintained by an active
    development community.

    This test is parametrized over the benchmark type (see
    :attr:`benchmark_info`) and the variant of the code (see :attr:`variant`).
    Each test instance executes the benchmark, validates numerically its output
    and extracts and reports a performance metric.

    '''

    #: The output file to pass to the Amber executable.
    #:
    #: :type: :class:`str`
    #: :required: No
    #: :default: ``'amber.out'``
    output_file = variable(str, value='amber.out')

    #: The input file to use.
    #:
    #: This is set to ``mdin.CPU`` or ``mdin.GPU`` depending on the test
    #: variant during initialization.
    #:
    #: :type: :class:`str`
    #: :required: Yes
    input_file = variable(str)

    #: Parameter pack encoding the benchmark information.
    #:
    #: The first element of the tuple refers to the benchmark name,
    #: the second is the energy reference and the third is the
    #: tolerance threshold.
    #:
    #: :type: `Tuple[str, float, float]`
    #: :values:
    #:     .. code-block:: python
    #:
    benchmark_info = parameter([
        ('Cellulose_production_NVE_4fs', -394000.0, 1.0E-03),
        #('FactorIX_production_NVE', -234188.0, 1.0E-04),
        #('JAC_production_NVE_4fs', -44810.0, 1.0E-03),
        #('JAC_production_NVE', -58138.0, 5.0E-04)
    ], fmt=lambda x: x[0])

    # Parameter encoding the variant of the test.
    #
    # :type:`str`
    # :values: ``['mpi', 'cuda']``
    variant = parameter(['mpi', 'cuda', 'rocm'], loggable=True)

    # Test tags
    #
    # :required: No
    # :default: ``{'sciapp', 'chemistry'}``
    tags = {'sciapp', 'chemistry'}

    #: See :attr:`~reframe.core.pipeline.RegressionTest.num_tasks`.
    #:
    #: The ``mpi`` variant of the test requires ``num_tasks > 1``.
    #:
    #: :required: Yes
    num_tasks = required

    @loggable
    @property
    def bench_name(self):
        '''The benchmark name.

        :type: :class:`str`
        '''

        return self.__bench

    @property
    def energy_ref(self):
        '''The energy reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref

    @property
    def energy_tol(self):
        '''The energy tolerance value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_tol

    @run_after('init')
    def prepare_test(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'Amber NVE {self.bench_name} benchmark ({self.variant})'

        params = {
            'mpi':  ('mdin.CPU', 'pmemd.MPI'),
            'cuda': ('mdin.GPU', 'pmemd.cuda.MPI'),
            'rocm': ('mdin.GPU', 'pmemd.hip')
        }
        try:
            self.input_file, self.executable = params[self.variant]
        except KeyError:
            raise ValueError(
                f'test not set up for platform {self.variant!r}'
            ) from None

        benchmark_suite_name = 'Amber20_Benchmark_Suite'
        benchmark_suite_dir = os.path.join(self.current_system.resourcesdir, 'amber', benchmark_suite_name)
        input_file_path = os.path.join(benchmark_suite_dir, 'PME', self.bench_name, self.input_file)
        top_file_path = os.path.join(benchmark_suite_dir, 'PME', 'Topologies', self.bench_name.split('_')[0] + '.prmtop')
        crd_file_path = os.path.join(benchmark_suite_dir, 'PME', 'Coordinates', self.bench_name.split('_')[0] + '.inpcrd')
        self.prerun_cmds = [
            f'ln -s {input_file_path} .',
            f'ln -s {top_file_path} .',
            f'ln -s {crd_file_path} . '
        ]
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file,
                                '-p', top_file_path,
				'-c', crd_file_path,
                               ]
        self.keep_files = [self.output_file]

    @performance_function('ns/day')
    def perf(self):
        '''The performance of the benchmark expressed in ``ns/day``.'''
        return sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                self.output_file, 'perf', float, item=1)

    @sanity_function
    def assert_energy_readout(self):
        '''Assert that the obtained energy meets the required tolerance.'''

        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_diff = sn.abs(energy - self.energy_ref)
        ref_ener_diff = sn.abs(self.energy_ref *
                               self.energy_tol)
        return sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])


@rfm.simple_test
class lumi_amber_check(amber_nve20_check):
    modules = ['Amber']
    valid_prog_environs = ['PrgEnv-gnu']
    tags |= {'maintenance', 'production'}
    maintainers = ['mszpindler']
    num_nodes = parameter([1, 2], loggable=True)
    allref = {
        1: {
            'gfx90a': {
                'Cellulose_production_NVE_4fs': (115.0, -0.10, None, 'ns/day'),
            }
        },
        4: {
            'zen3': {
                'Cellulose_production_NVE_4fs': (10.0, -0.30, None, 'ns/day'),
                #'FactorIX_production_NVE': (7.0, -0.30, None, 'ns/day'),
                #'JAC_production_NVE': (30.0, -0.30, None, 'ns/day'),
                #'JAC_production_NVE_4fs': (45.0, -0.30, None, 'ns/day')
            }
        },
    }

    tags = {'contrib/22.08'}

    @run_after('init')
    def scope_systems(self):
        valid_systems = {
            'rocm': {1: ['lumi:gpu']},
            'mpi': {
                2: ['lumi:small'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.variant][self.num_nodes]
        except KeyError:
            self.valid_systems = []

    @run_after('init')
    def set_num_gpus_per_node(self):
        if self.variant == 'rocm':
            self.num_gpus_per_node = 1

    @run_after('setup')
    def set_num_tasks(self):
        if self.variant == 'rocm':
            self.num_tasks_per_node = 1
        else:
            proc = self.current_partition.processor
            pname = self.current_partition.fullname
            self.num_tasks_per_node = proc.num_cores

        self.num_tasks = self.num_nodes * self.num_tasks_per_node

    @run_before('performance')
    def set_perf_reference(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if pname in ('lumi:gpu'):
            arch = 'gfx90a'
        else:
            arch = proc.arch

        with contextlib.suppress(KeyError):
            self.reference = {
                pname: {
                    'perf': self.allref[self.num_nodes][arch][self.bench_name]
                }
            }
