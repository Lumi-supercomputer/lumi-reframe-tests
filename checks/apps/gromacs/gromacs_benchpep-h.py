import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

@rfm.simple_test
class lumi_gromacs_pep_h(gromacs_check):
    # Here purpose of second and third parameters changes to total energy 
    # at step 0 and energy drift; tolerances are now in readout functions
    benchmark_info = parameter([
        ('benchPEP-h', [-1.43526e+08,7.50e-05], [0.001, 0.033]), 
    ], fmt=lambda x: x[0], loggable=True)
    update_mode = parameter(['gpu', 'cpu'])
    nb_impl = parameter(['gpu'])
    bonded_impl = parameter(['gpu'])
    fft_variant = parameter(['heffte', 'vkfft'])
    maintainers = ['mszpindler']
    use_multithreading = False
    exclusive_access = True
    num_nodes = parameter([2, 4], loggable=True)
    num_gpus_per_node = 8
    time_limit = '15m'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']

    allref = {
        1: {
            'gpu': { # update=gpu, gpu resident mode
                'benchPEP-h': (7.3, -0.05, 0.05, 'ns/day'),
            },
            'cpu': { # update=cpu, force offload mode
                'benchPEP-h': (4.6, -0.05, 0.05, 'ns/day'),
            },
        },
        2: {
            'gpu': { # update=gpu, gpu resident mode
                'benchPEP-h': (13.2, -0.075, 0.075, 'ns/day'),
            },
            'cpu': { # update=cpu, force offload mode
                'benchPEP-h': (9.5, -0.05, 0.05, 'ns/day'),
            },
        },
        4: {
            'gpu': { # update=gpu, gpu resident mode
                'benchPEP-h': (13.2, -0.05, None, 'ns/day'),
            },
            'cpu': { # update=cpu, force offload mode
                'benchPEP-h': (9.5, -0.05, 0.05, 'ns/day'),
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
    def energy_step0_ref(self):
        '''The energy reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref[0]

    @property
    def energy_drift_ref(self):
        '''The energy drift reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref[1]

    @property
    def energy_step0_tol(self):
        return self.__nrg_tol[0]

    @property
    def energy_drift_tol(self):
        return self.__nrg_tol[1]

    @run_after('init')
    def prepare_test(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'GROMACS {self.__bench} GPU benchmark (update mode: {self.update_mode}, bonded: {self.bonded_impl}, non-bonded: {self.nb_impl})'
        bench_file_path = os.path.join(self.current_system.resourcesdir, 'gromacs-benchmarks', 'www.mpinat.mpg.de', self.__bench, f'{self.__bench}.tpr')
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]

    @run_after('init')
    def setup_fft_variant(self):
        match self.fft_variant:
            case 'heffte':
                self.modules = ['GROMACS/2023.2-cpeAMD-22.12-HeFFTe-GPU']
                self.num_tasks_per_node = 8
                npme_ranks = 2*self.num_nodes
            case 'vkfft':
                self.modules = ['GROMACS/2023.2-cpeAMD-22.12-VkFFT-GPU']
                self.num_tasks_per_node = 8
                npme_ranks = 1
            case _:
                self.skip('FFT library variant not defined')

        self.executable_opts += [
            '-nsteps 20000',
            '-nstlist 400',
            '-noconfout',
            '-nb', self.nb_impl,
            '-npme', f'{npme_ranks}',
            '-pme', 'gpu', 
            '-update', self.update_mode,
            '-bonded', self.bonded_impl,
            '-s benchmark.tpr'
        ]
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
            'OMP_NUM_THREADS': '7',
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
            'OMP_DISPLAY_ENV': '1',
            'OMP_DISPLAY_AFFINITY': 'TRUE',
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1'
        }

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
        valid_systems = {
            'heffte': {
                2: ['lumi:gpu'],
                4: ['lumi:gpu'],
            },
            'vkfft': {
                1: ['lumi:gpu'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.fft_variant][self.num_nodes]
        except KeyError:
            self.valid_systems = []


    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)
    @deferrable
    def energy_drift(self):
        return sn.extractsingle(r'\s+Conserved\s+energy\s+drift\:\s+(\S+)', 'md.log', 1, float)

    @deferrable
    def energy_step0(self):
        return sn.extractsingle(r'\s+Kinetic En\.\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\n'
                                r'(\s+\S+)\s+(?P<energy>\S+)(\s+\S+){3}\n'
                                r'\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)
    
    @sanity_function
    def assert_energy_readout(self):
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log', 'Run failed to complete'), 
            sn.assert_reference(self.energy_step0(), self.energy_step0_ref, -self.energy_step0_tol, self.energy_step0_tol, 'Failed to meet reference value for total energy at step 0'),
            sn.assert_reference(self.energy_drift(), self.energy_drift_ref, -self.energy_drift_tol, self.energy_drift_tol, 'Failed to meet reference value for conserved energy drift')
        ])

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes][self.update_mode][self.bench_name]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.update_mode][self.bench_name]
            }
        }
        self.num_tasks = self.num_tasks_per_node*self.num_nodes

