import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

# This is to test single node, multi-gpu performance with AMD's HIP port of GROMACS
# available at: https://www.amd.com/en/technologies/infinity-hub/gromacs
# docker pull amdih/gromacs:2022.3.amd1_174

@rfm.simple_test
class lumi_gromacs_containerized(gromacs_check):
    maintainers = ['mszpindler']
    use_multithreading = True
    valid_prog_environs = ['builtin']
    container_platform = 'Singularity'
    num_nodes = 1
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    num_cpus_per_task = 6
    valid_systems = ['lumi:gpu']
    exclusive_access = True
    time_limit = '15m'

    nb_impl = parameter(['gpu'])

    gromacs_image = parameter([
        'docker://amdih/gromacs:2022.3.amd1_174',
    ])

    benchmark_info = parameter([
        ('StandardMD/benchMEM', -1.08735e+06, 0.001), 			# atoms 81,743
    #    ('StandardMD/benchPEP', -1.43527e+08, 0.001), 			# atoms 12,495,503
        ('HECBioSim/Crambin', -204107.0, 0.001),      			# atoms 19,605
        ('HECBioSim/Glutamine-Binding-Protein', -724598.0, 0.001),	# atoms 61,153
        ('HECBioSim/hEGFRDimer', -3.32892e+06, 0.001),			# atoms 465,399
    #    ('HECBioSim/hEGFRDimerSmallerPL', -3.27080e+06, 0.001),		# atoms 465,399
        ('HECBioSim/hEGFRDimerPair', -1.20733e+07, 0.001),		# atoms 1,403,182
        ('HECBioSim/hEGFRtetramerPair', -2.09831e+07, 0.001)		# atoms 2,997,924
    ], fmt=lambda x: x[0], loggable=True)
    
    executable = 'gmx mdrun'

    reference = {
        'lumi:gpu': {
            'HECBioSim/Crambin': (320.0, None, None, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (115.503, None, None, 'ns/day'),  
            'HECBioSim/hEGFRDimer': (18.938, None, None, 'ns/day'),
            'HECBioSim/hEGFRDimerSmallerPL': (33.785, None, None, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (7.759, None, None, 'ns/day'),
            'StandardMD/benchMEM': (92.043, None, None, 'ns/day'),
        }
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
    def overwrite_cmds(self):
    # Need to overwrite because of downloads being not possible on compute nodes
        self.prerun_cmds = []
        self.variables = {'SINGULARITYENV_GMX_GPU_DD_COMMS': 'true', 'SINGULARITYENV_GMX_GPU_PME_PP_COMMS': 'true', 'SINGULARITYENV_GMX_FORCE_UPDATE_DEFAULT_GPU': 'true', 'SINGULARITYENV_OMP_NUM_THREADS': '6', 'SINGULARITYENV_OMP_PROC_BIND': 'close', 'SINGULARITYENV_OMP_PLACES': 'cores', 'SLURM_CPU_BIND': 'map_cpu:1,9,17,25,33,49,57,62'}
        self.executable_opts = [
	    '-notunepme', 
            '-noconfout', 
            '-resethway',
            '-ntomp', f'{self.num_cpus_per_task}',
            '-ntmpi', f'{self.num_tasks_per_node}',
            '-pin', 'on',
            '-pinoffset', '1',
            '-npme', '1',
            '-bonded', 'cpu',
            '-nb', 'gpu', 
            '-pme', 'gpu', 
            '-gpu_id', '4,5,2,3,6,7,0,1',
            #'-gputasks', '45236701', 
            '-v',
            '-s', 
            f'{self.bench_name}/benchmark.tpr'
        ]


    @run_before('run')
    def set_container_variables(self):
        exec_cmd = ' '.join([self.executable, *self.executable_opts])
        self.container_platform.image = self.gromacs_image
        self.container_platform.command = exec_cmd
