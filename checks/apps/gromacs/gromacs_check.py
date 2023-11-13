import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

@rfm.simple_test
class lumi_gromacs_tmpi(gromacs_check):
    descr = """This is to test single node, threadMPI version of GROMACS. 
    Multi-gpu version runs from AMD's container 
    available at: https://www.amd.com/en/technologies/infinity-hub/gromacs""" 
    maintainers = ['mszpindler']
    use_multithreading = False
    valid_prog_environs = ['builtin', 'cpeGNU']
    container_platform = 'Singularity'
    num_nodes = 1
    exclusive_access = True
    time_limit = '15m'

    benchmark_info = parameter([
        ('StandardMD/benchMEM', -1.08735e+06, 0.001), 			# atoms 81,743
        ('HECBioSim/Crambin', -204107.0, 0.001),      			# atoms 19,605
        ('HECBioSim/Glutamine-Binding-Protein', -724598.0, 0.001),	# atoms 61,153
        ('HECBioSim/hEGFRDimer', -3.32892e+06, 0.001),			# atoms 465,399
        ('HECBioSim/hEGFRDimerPair', -1.20733e+07, 0.001),		# atoms 1,403,182
        ('HECBioSim/hEGFRtetramerPair', -2.09831e+07, 0.001)		# atoms 2,997,924
    ], fmt=lambda x: x[0], loggable=True)
    
    executable = 'gmx mdrun'

    reference = {
        'lumi:gpu': {
            'HECBioSim/Crambin': (210.0, None, None, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (198.0, None, None, 'ns/day'),  
            'HECBioSim/hEGFRDimer': (53.0, None, None, 'ns/day'),
            'HECBioSim/hEGFRtetramerPair': (12.5, None, None, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (35.4, None, None, 'ns/day'),
            'StandardMD/benchMEM': (180.0, None, None, 'ns/day'),
        },
        'lumi:small': { 
            # Needs to be updated
            'HECBioSim/Crambin': (320.0, None, None, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (115.503, None, None, 'ns/day'),  
            'HECBioSim/hEGFRDimer': (18.938, None, None, 'ns/day'),
            'HECBioSim/hEGFRDimerSmallerPL': (33.785, None, None, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (7.759, None, None, 'ns/day'),
        },
    }

    @deferrable
    def energy_standardmd_benchpep(self):
        return sn.extractsingle(r'\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\s+Constr\. rmsd\n'
                                r'\s+(?P<energy>\S+)\s+.*',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_standardmd_benchmem(self):
        return sn.extractsingle(r'\s+Kinetic En\.\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\n'
                                r'(\s+\S+)\s+(?P<energy>\S+)\s+.*',
                                'md.log', 'energy', float, item=-1)

    @run_after('init')
    def setup_run(self):
        if self.nb_impl == 'gpu':
            self.valid_systems = ['lumi:gpu']
            self.valid_prog_environs = ['builtin']
            self.num_gpus_per_node = 8
            self.num_cpus_per_gpu = 6
        elif self.nb_impl == 'cpu': 
            self.valid_systems = ['lumi:small']
            self.valid_prog_environs = ['cpeGNU']
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 48

    @run_after('init')
    def overwrite_cmds(self):
    # Need to overwrite because of downloads being not possible on compute nodes
        self.prerun_cmds = []
        if self.nb_impl == 'gpu':
            self.env_vars = {
                'SINGULARITYENV_GMX_GPU_DD_COMMS': 'true', 
                'SINGULARITYENV_GMX_GPU_PME_PP_COMMS': 'true', 
                'SINGULARITYENV_GMX_FORCE_UPDATE_DEFAULT_GPU': 'true', 
                'SINGULARITYENV_OMP_NUM_THREADS': f'{self.num_cpus_per_gpu}', 
            }
        if self.nb_impl == 'cpu':
            self.env_vars = {
                'OMP_NUM_THREADS': '6', 
                'OMP_PROC_BIND': 'master',
                'OMP_PLACES': '{0:5}:8:8',
            }
        
        self.executable_opts = [
            '-notunepme', 
            '-resethway',
            '-pin', 'on',
            '-pinstride', '1',
            '-bonded', 'cpu',
            '-nb', f'{self.nb_impl}',
            '-pme', f'{self.nb_impl}',
            '-v',
            '-s', 
            f'{self.bench_name}/benchmark.tpr'
        ]

        if self.nb_impl == 'gpu':
            self.executable_opts += ['-npme', '1',]
            self.executable_opts += ['-ntomp', f'{self.num_cpus_per_gpu}',]
            self.executable_opts += ['-ntmpi', f'{self.num_gpus_per_node}',]
            self.executable_opts += ['-pinoffset', '1',]
            self.executable_opts += ['-gpu_id', '4,5,2,3,6,7,0,1']
            # alternative gpu mapping option: 
            #self.executable_opts += ['-gputasks', '45236701']
        if self.nb_impl == 'cpu':
            self.executable_opts += ['-npme', '-1',]
            self.executable_opts += ['-dlb', 'yes',]
            self.executable_opts += ['-ntomp', '6',]
            self.executable_opts += ['-ntmpi', '8']

    @run_before('run')
    def set_cpu_binding(self):
        if self.nb_impl == 'gpu':
            self.job.options = [f'--cpus-per-gpu={self.num_cpus_per_gpu}']

    @run_before('run')
    def set_runtime(self):
        exec_cmd = ' '.join([self.executable, *self.executable_opts])
        if self.nb_impl == 'gpu':
            self.container_platform.image = 'docker://amdih/gromacs:2022.3.amd1_174'
            self.container_platform.command = exec_cmd
        elif self.nb_impl == 'cpu':
            self.modules = ['GROMACS']

@rfm.simple_test
class lumi_gromacs_mpi(gromacs_check):
    descr = """This is to test multi-node, MPI version of GROMACS.""" 
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    valid_prog_environs = ['cpeGNU']
    valid_systems = ['lumi:standard']
    modules = ['GROMACS']
    time_limit = '15m'

    num_nodes = parameter([2, 4], loggable=True)

    nb_impl = parameter(['cpu'])
    
    allref = {
        2: {
            #'lumi:standard': { # Results collected with GROMACS 2023.2 and cpeGNU/22.12
            'HECBioSim/Crambin': (632.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (237.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimer': (26.8, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (13.4,-0.1, 0.1, 'ns/day'),
            #},
        },
        4: {
            #'lumi:standard': { # Results collected with GROMACS 2023.2 and cpeGNU/22.12
            'HECBioSim/Crambin': (576.5, -0.05, 0.05, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (375.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimer': (47.5, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (24.5, -0.1, 0.1, 'ns/day'),
            'HECBioSim/hEGFRtetramerPair': (11.1, -0.1, 0.1, 'ns/day'),
            #},
        },
    }

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes][self.bench_name]
        except KeyError:
            self.skip(f'No reference performance results for {self.bench_name!r} with {self.num_nodes}')
        self.num_tasks_per_node = 128
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.bench_name]
            }
        }
