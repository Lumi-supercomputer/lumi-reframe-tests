import reframe as rfm
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check


@rfm.simple_test
class lumi_gromacs_check(gromacs_check):
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    valid_prog_environs = ['cpeGNU']
    modules = ['GROMACS']
    time_limit = '15m'

    num_nodes = parameter([1, 2, 4], loggable=True)
    allref = {
        1: {
            'gfx90a': { 
            },
            'zen3': { 
                'HECBioSim/Crambin': (320.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (115.503, None, None, 'ns/day'),  
                'HECBioSim/hEGFRDimer': (18.938, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (33.785, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (7.759, None, None, 'ns/day'),
            },
        },
        2: {
            'gfx90a': { 
            },
            'zen3': { # Collected initial performance numbers after SS11 upgrade
                'HECBioSim/Crambin': (280.267, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (210.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (31.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (30.571, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (14.269, None, None, 'ns/day'),
            },
        },
        4: {
            'gfx90a': { 
            },
            'zen3': { # Collected initial performance numbers after SS11 upgrade
                'HECBioSim/Crambin': (295.263, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (190.459, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (32.232, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (34.619, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (21.66, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (9.58, None, None, 'ns/day'),
            },
        },
    }

    @run_after('init')
    def overwrite_cmds(self):
    # Need to overwrite because of downloads being not possible on compute nodes
        self.prerun_cmds = []
        self.executable_opts = ['-nb', self.nb_impl, '-s', 
            f'{self.bench_name}/benchmark.tpr'
        ]


    @run_after('init')
    def setup_filtering_criteria(self):
        # Update test's description
        self.descr += f' ({self.num_nodes} node(s))'

        # Setup system filtering
        valid_systems = {
            'cpu': {
                1: ['lumi:small'],
                2: ['lumi:small'],
                4: ['lumi:small'],
            },
            'gpu': {
                1: ['lumi:gpu'],
                2: ['lumi:gpu'],
                4: ['lumi:gpu'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.nb_impl][self.num_nodes]
        except KeyError:
            self.valid_systems = []


    @run_before('run')
    def setup_run(self):
        # Setup GPU run
        if self.nb_impl == 'gpu':
            self.num_gpus_per_node = 1

        proc = self.current_partition.processor

        # Choose arch; we set explicitly the GPU arch, since there is no
        # auto-detection
        arch = proc.arch
        if self.current_partition.fullname in ('lumi:gpu'):
            arch = 'gfx90a'

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
