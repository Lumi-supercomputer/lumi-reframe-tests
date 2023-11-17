import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

@rfm.simple_test
class lumi_gromacs_pep_h(gromacs_check):
    benchmark_info = parameter([
        ('benchPEP-h', 1.27e-04, 1e-05), 
    ], fmt=lambda x: x[0], loggable=True)
    update_mode = parameter(['gpu', 'cpu'])
    nb_impl = parameter(['gpu'])
    bonded_impl = parameter(['gpu'])
    maintainers = ['mszpindler']
    use_multithreading = False
    exclusive_access = True
    num_nodes = parameter([1,2], loggable=True)
    num_gpus_per_node = 8
    num_tasks_per_node = 8
    time_limit = '15m'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']
    fft_variant = parameter(['heffte', 'vkfft'])

    allref = {
        1: {
            'gfx90a': {
                'benchPEP-h': (7.3, -0.05, 0.05, 'ns/day'),
            },
        },
        2: {
            'gfx90a': {
                'benchPEP-h': (13.2, -0.05, 0.05, 'ns/day'),
            },
        },
    }

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
    def prepare_ztest(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'GROMACS {self.__bench} GPU benchmark (update mode: {self.update_mode}, bonded: {self.bonded_impl}, non-bonded: {self.nb_impl})'
        bench_file_path = os.path.join(self.current_system.resourcesdir, 'gromacs-benchmarks', 'www.mpinat.mpg.de', self.__bench, f'{self.__bench}.tpr')
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]
        npme_ranks = 2*self.num_nodes
        self.executable_opts += [
            '-nsteps 5000',
            '-nstlist 400',
            '-noconfout',
            '-nb', self.nb_impl,
            '-npme', f'{npme_ranks}',
            '-pme', 'gpu', 
            '-update', self.update_mode,
            '-bonded', self.bonded_impl,
            '-pin on',
            '-s benchmark.tpr'
        ]
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'OMP_NUM_THREADS': '7',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1'
        }

    @run_after('init')
    def set_modules(self):
        match self.fft_variant:
            case 'heffte':
                self.modules = ['GROMACS/2023.2-cpeGNU-22.12-HeFFTe-GPU']
            case 'vkfft':
                self.modules = ['GROMACS/2023.2-cpeGNU-22.12-VkFFT-GPU']
            case _:
                self.skip('FFT library not defined')

    @run_before('run')
    def set_cpu_mask(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

    @run_after('init')
    def add_select_gpu_wrapper(self):
        self.prerun_cmds += [
            'cat << EOF > select_gpu',
            '#!/bin/bash',
            'export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID',
            'exec \$*',
            'EOF',
            'chmod +x ./select_gpu'
        ]
        self.executable = './select_gpu ' + self.executable


    @run_after('init')
    def setup_nb(self):
    # Fix it !
        valid_systems = {
            'gpu': {
                1: ['lumi:gpu'],
                2: ['lumi:gpu'],
            },
            'cpu': {
                1: ['lumi:gpu'],
                2: ['lumi:gpu'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.update_mode][self.num_nodes]
        except KeyError:
            self.valid_systems = []


    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)
    @deferrable
    def energy_drift(self):
        return sn.extractsingle(r'\s+Conserved\s+energy\s+drift\:\s+(\S+)', 'md.log', 1, float)
    
    @sanity_function
    def assert_energy_readout(self):
        return sn.and_(sn.assert_found('Finished mdrun', 'md.log'), sn.assert_le(sn.abs(self.energy_drift() - self.energy_ref), self.energy_tol))

    @run_before('run')
    def setup_run(self):
        proc = self.current_partition.processor
        arch = 'gfx90a'

        try:
            found = self.allref[self.num_nodes][arch][self.bench_name]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        # Refactor this (replace arch) !
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][arch][self.bench_name]
            }
        }
        self.num_tasks = self.num_tasks_per_node*self.num_nodes
