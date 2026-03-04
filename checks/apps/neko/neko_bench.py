# adapted from Dardel (author: javier Aguilar Fruto)

import reframe as rfm
import reframe.utility.sanity as sn
import os
import csv
import string

class NekoError(Exception):
    pass

class MakeNeko(rfm.core.buildsystems.BuildSystem):
    srcfile = variable(str, type(None), value=None)

    def __init__(self):
        self.makeneko = 'makeneko'

    def emit_build_commands(self, environ):
        if not self.srcfile:
            raise NekoError('Source file required')

        return [f'{self.makeneko} "{self.srcfile}"']

class lumi_make_neko(MakeNeko, rfm.CompileOnlyRegressionTest):
    case = variable(str)
    modules = ['Neko/1.0.1-cpeCray-25.03-rocm']

    @run_after('setup')
    def set_build(self):
        self.build_system = MakeNeko()
        self.sourcepath = f'{self.case}.f90'

    @sanity_function
    def check_build(self):
        #return sn.assert_found('Building user NEKO ... done!', self.stdout)
        return sn.assert_found(' Done!', self.stdout)

class NekoTGVBase(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeCray']
    exclusive_access = True
    time_limit = '15m'

    # tested on the TDS, not working on the system with 25.03 
    modules = ['Neko/1.0.1-cpeCray-25.03-rocm']
    case = 'tgv'

    makeneko = fixture(lumi_make_neko, scope='environment', variables={'case': case})

    num_nodes = parameter([1,2,4,8,16])
    num_gpus_per_node = 8

    size = parameter([32768, 262144], loggable=True)
    
    mesh_file = variable(str, value='')

    perf_relative = variable(float, value=0.0, loggable=True)

    @run_before('compile')
    def set_mesh_file(self):
        self.mesh_file = f'{str(self.size)}.nmsh'
        mesh_file_path = os.path.join(self.current_system.resourcesdir, 
                                       'datasets',
                                       'neko', 
                                       'examples',
                                       'tgv',
                                       self.mesh_file)

        self.prerun_cmds += [
            f'ln -s {mesh_file_path} .'
        ]
   
    @run_after('init')
    def set_environment(self):
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
        }

    @run_before('run')
    def add_executable(self):
        self.executable = os.path.join(self.makeneko.stagedir,
                                       'neko')
        case_file = os.path.join(self.stagedir, 
                                 str(self.size),
                                 f'{self.case}.case')
        self.executable_opts.append(case_file)

    @run_before('run')
    def set_num_tasks(self):
        self.num_tasks_per_node = self.num_gpus_per_node
        self.num_tasks = self.num_nodes*self.num_tasks_per_node

    @run_before('run')
    def set_gpu_binding(self):
        self.job.launcher.options = [
            '--cpus-per-task=7',
            '--gpu-bind=map:4,5,2,3,6,7,0,1',
            '--gres-flags=allow-task-sharing'
        ]

    @sanity_function
    def normal_end(self):
        return sn.assert_found('Normal end.', self.stdout)

    @run_before('performance')
    def set_time_perf(self):
        timesteps = sn.extractall(r'Total elapsed time \(s\):\s\s\s+(\S+)', self.stdout, 1, float)

        pf = sn.make_performance_function(lambda: timesteps[-1], 's')
        self.perf_variables['total_runtime'] = pf

@rfm.simple_test
class lumi_neko_bench(NekoTGVBase):
    allref = {
        32768: {
            1: {
                'lumi:gpu': {
                    'total_runtime': (143, -0.50, 0.05, 's'),
                }
            },
            2: {
                'lumi:gpu': {
                    'total_runtime': (91, -0.50, 0.05, 's'),
                }
            },
            4: {
                'lumi:gpu': {
                    'total_runtime': (66, -0.50, 0.05, 's'),
                }
            }
        },
       262144: {
            8: {
                'lumi:gpu': {
                    'total_runtime': (396, -0.50, 0.05, 's'),
                }
            },
            16: {
                'lumi:gpu': {
                    'total_runtime': (272, -0.50, 0.05, 's'),
                }
            },
       } 
    }

    @run_after('init')
    def select_tests(self):
        try:
             found = self.allref[self.size][self.num_nodes]
        except KeyError:
            self.skip(f'Test for check of size {self.size} with {self.num_nodes} nodes skipped')

    @run_before('performance')
    def set_reference(self):
        self.reference = self.allref[self.size][self.num_nodes]

    @run_after('performance')
    def lower_the_better(self):
        perf_var = 'total_runtime'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][1]-self.perfvalues[key_str][0])/self.perfvalues[key_str][1])
