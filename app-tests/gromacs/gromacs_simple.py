import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check


@rfm.simple_test
class simple_gromacs_check(gromacs_check):
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    pe_release = '21.12'
    gromacs_version = '2021.5'
    num_nodes = 1

    benchmark_info = parameter([
        ('benchMEM', -1.08735e+06, 0.001),
        ('benchPEP', -1.08735e+06, 0.001), # Fix it
    ], fmt=lambda x: x[0], loggable=True)

    allref = {
	1: {
            'cascadelake': {  # Copied from Cascade Lake results 
                'benchMEM': (92.043, None, None, 'ns/day'),
                'benchPEP': (0.561, None, None, 'ns/day'), 
            }
        }
    }

    @run_after('init')
    def setup_modules(self):
        if self.current_system.name in ('lumi'):
            if self.nb_impl == 'cpu':
                self.modules = ['GROMACS-{gromacs_version}-cpeGNU-{pe_release}-CPU']
                self.valid_prog_environs = ['cpeGNU']
            # Add module information for GPU enabled version    
            #elif self.nb_impl == 'gpu':
            #    self.modules = ['GROMACS-{gromacs_version}-cpeGNU-{pe_release}-GPU']
            #    self.valid_prog_environs = ['cpeGNU']

    @run_after('init')
    def prepare_test(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'GROMACS {self.__bench} benchmark (NB: {self.nb_impl})'
        self.prerun_cmds = [
            f'curl -LJO https://www.mpinat.mpg.de/{self.__bench} && unzip {self.__bench}.zip && ln -s {self.__bench}.tpr benchmark.tpr'
        ]
        self.executable_opts += ['-nb', self.nb_impl, '-s benchmark.tpr']

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
    def setup_nb(self):
        valid_systems = {
            'cpu': {
                1: ['lumi:cpu'],
            },
            'gpu': {
                1: ['lumi:gpu'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.nb_impl][self.num_nodes]
        except KeyError:
            self.valid_systems = []

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)

    @deferrable
    def energy(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @sanity_function
    def assert_energy_readout(self):
        #energy_diff = sn.abs(energy - self.energy_ref)
        return sn.assert_found('Finished mdrun', 'md.log')


    @run_before('run')
    def setup_run(self):
        self.skip_if_no_procinfo()

        # Setup GPU run
        if self.nb_impl == 'gpu':
            self.num_gpus_per_node = 1

        proc = self.current_partition.processor

        # Choose arch; we set explicitly the GPU arch, since there is no
        # auto-detection
        arch = proc.arch

        try:
            found = self.allref[self.num_nodes][arch][self.bench_name]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][arch][self.bench_name]
            }
        }

        # Setup parallel run
        self.num_tasks_per_node = proc.num_cores
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
