import os
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class lumi_gromacs_pep_h(rfm.RunOnlyRegressionTest):
    '''GROMACS PEP-h benchmark.
    Dept. of Theoretical and Computational Biophysics, Max Planck Institute for Multidisciplinary Sciences, Göttingen, https://www.mpinat.mpg.de/grubmueller/bench
    
    Direct access to the data set: https://www.mpinat.mpg.de/benchPEP-h
    '''
    benchmark_info = {
        'name': 'benchPEP-h',
        'energy_step0_ref': -1.43526e+08,
        'energy_step0_tol': 0.001,
    }

    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']
    module_ver = parameter([
        '2024.1-cpeAMD-23.09-HeFFTe-2.4-AdaptiveCpp-23.10.0-rocm-5.4.6',
    ], loggable=True)
    maintainers = ['mszpindler']
    use_multithreading = False
    exclusive_access = True
    num_nodes = parameter([2, 4], loggable=True)
    num_gpus_per_node = 8
    time_limit = '15m'
    nb_impl = parameter(['gpu'])
    update_mode = parameter(['gpu', 'cpu'])
    executable = 'gmx_mpi mdrun'
    tags = {'benchmark', 'contrib', 'gpu'}
    keep_files = ['md.log']

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

    @run_after('init')
    def prepare_test(self):
        self.descr = f"GROMACS {self.benchmark_info['name']} GPU benchmark (update mode: {self.update_mode}, non-bonded: {self.nb_impl})"
        bench_file_path = os.path.join(self.current_system.resourcesdir, 
                                      'datasets',
                                      'gromacs',
                                      self.benchmark_info['name'],
                                      f"{self.benchmark_info['name']}.tpr")
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]

    @run_after('init')
    def apply_module_ver(self):
        module = f'GROMACS/{self.module_ver}'
        self.modules = [module]

    @run_after('init')
    def setup_runtime(self):
        self.num_tasks_per_node = 8
        self.num_tasks = self.num_tasks_per_node*self.num_nodes
        if self.num_nodes > 1:
           npme_ranks = 2*self.num_nodes
        else:
           npme_ranks = 1

        self.executable_opts += [
            '-nsteps 20000',
            '-nstlist 400',
            '-noconfout',
            '-nb', self.nb_impl,
            '-npme', f'{npme_ranks}',
            '-pme', 'gpu', 
            '-update', self.update_mode,
            '-bonded', 'gpu',
            '-s benchmark.tpr'
        ]
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
            'OMP_NUM_THREADS': '7',
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1',
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

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)

    @deferrable
    def energy_step0(self):
        return sn.extractsingle(r'\s+Kinetic En\.\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\n'
                                r'(\s+\S+)\s+(?P<energy>\S+)(\s+\S+){3}\n'
                                r'\s+Constr\. rmsd',
                                'md.log', 'energy', float)

    @deferrable
    def energy_drift(self):
        return sn.extractsingle(r'\s+Conserved\s+energy\s+drift\:\s+(\S+)', 'md.log', 1, float)

    @deferrable
    def verlet_buff_tol(self):
        return sn.extractsingle(r'\s+verlet-buffer-tolerance\s+\=\s+(\S+)', 'md.log', 1, float)
    
    @sanity_function
    def assert_energy_readout(self):
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log', 'Run failed to complete'), 
            sn.assert_reference(
                self.energy_step0(),
                self.benchmark_info['energy_step0_ref'],
                -self.benchmark_info['energy_step0_tol'],
                self.benchmark_info['energy_step0_tol']
            ),
            sn.assert_lt(
                self.energy_drift(),
                2*self.verlet_buff_tol()
           ),
        ])

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes][self.update_mode][self.benchmark_info['name']]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.update_mode][self.benchmark_info['name']]
            }
        }
