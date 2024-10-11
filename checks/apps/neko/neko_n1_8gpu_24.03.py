import reframe as rfm
import reframe.utility.sanity as sn
import os
import csv
import string

def get_gpu_device(partition):
    for device in partition.devices:
        if device.type == 'gpu':
            return device

class NekoError(Exception):
    pass


# Use this for children of NekoTestBase that don't need makeneko
class DummyBuildSystem(rfm.core.buildsystems.BuildSystem):
    def emit_build_commands(self, environ):
        return []

class MakeNeko(rfm.core.buildsystems.BuildSystem):
    srcfile = variable(str, type(None), value=None)

    def __init__(self):
        self.makeneko = os.path.join('/project/project_462000008/jigong2/neko/0.8.1/', 'bin', 'makeneko')

    def emit_build_commands(self, environ):
        if not self.srcfile:
            raise NekoError('Source file required')

        return [f'{self.makeneko} "{self.srcfile}"']

class NekoTestBase(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['PrgEnv-cray']

    modules = ['LUMI/24.03', 'partition/G', 'rocm/6.0.3', 'craype-accel-amd-gfx90a', 'EasyBuild-user', 'json-fortran/8.3.0-cpeCray-24.03']
    scheme = parameter(os.getenv('NEKO_SCHEME', 'pnpn').split(','))
    case = variable(str)

    extra_resources = {'gpu': {'num_gpus_per_node': '8'}}

    mesh_file = variable(str, value='')
    dt = variable(str, value='')
    T_end = variable(str, value='')

    abstol_vel = {'sp': '1d-5', 'dp': '1d-9'}
    abstol_prs = {'sp': '1d-5', 'dp': '1d-9'}

    # Set dofs to enable workrate perf var
    dofs = variable(int, value=0)
    first_workrate_timestep = variable(int, value=0)

    @run_before('compile')
    def copy_mesh_file(self):
        if self.mesh_file == '':
            return

        src = os.path.join(self.prefix, '..', self.mesh_file)
        dst = os.path.join(self.stagedir, self.mesh_file)
        self.postbuild_cmds += [
                f'mkdir -p {os.path.dirname(self.mesh_file)}',
                f'cp "{src}" "{dst}"'
        ]

    @run_before('run')
    def make_case_file(self):
        case_file = os.path.join(self.stagedir, self.case)
        case_template = case_file + '.template'

        self.executable_opts.append(self.case)

        if os.path.exists(case_file):
            pass
        elif os.path.exists(case_template):
            with open(case_template) as tf:
                ts = tf.read()
            template = string.Template(ts)

            keys = {
                'abstol_vel': self.abstol_vel['dp'],
                'abstol_prs': self.abstol_prs['dp'],
                'fluid_scheme': self.scheme,
                'mesh_file': self.mesh_file,
                'dt': self.dt,
                'T_end': self.T_end,
            }

            ss = template.substitute(keys)
            with open(case_file, 'w') as cf:
                cf.write(ss)
        else:
            raise NekoError(f'Cannot find {case_file} or {case_template}')

    @run_before('run')
    def set_num_tasks(self):
        gpu_device = get_gpu_device(self.current_partition)
        if gpu_device is None:
            raise NekoError("Device of type gpu not defined for partition!")
        self.num_tasks = gpu_device.num_devices

    @run_before('run')
    def select_device(self):
        try:
            select_device = self.current_partition.extras['select_device']
            self.executable_opts.insert(0, self.executable)
            self.executable = select_device
        except KeyError:
            pass

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind="map_cpu:49,57,17,25,1,9,33,41"']


    @sanity_function
    def normal_end(self):
        return sn.assert_found('Normal end.', self.stdout)

    @run_before('performance')
    def set_time_perf(self):
        timesteps = sn.extractall(r'Elapsed time \(s\):\s+(\S+)', self.stdout, 1, float)

        pf = sn.make_performance_function(lambda: timesteps[-1], 's')
        self.perf_variables['total_runtime'] = pf

        if self.dofs != 0:
            pes = self.num_tasks

            def workrate():
                end = sn.count(timesteps) - 1
                time = timesteps[end] - timesteps[self.first_workrate_timestep]
                dofs = 8**3 * 32**3
                iters = end - self.first_workrate_timestep
                return 1e-3 * dofs * iters / time / pes

            pf = sn.make_performance_function(workrate, 'Mdofs/s/pe')
            self.perf_variables['workrate'] = pf

class GetTgvDns(rfm.RunOnlyRegressionTest):
    descr = 'Download TGV DNS data'
    executable = './get-tgv-dns.sh'
    local = True

    @run_after('run')
    def load_enstrophy(self):
        self.enstrophy = {}
        path = os.path.join(self.stagedir, 'spectral_Re1600_512.gdiag')
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if row[0][0] == '#':
                    continue
                # time: value
                self.enstrophy[float(row[0])] = float(row[3])

    @sanity_function
    def check_data_count(self):
        return sn.assert_eq(sn.count(sn.defer(self.enstrophy)), 2000)

class TgvBase(NekoTestBase):
    descr = 'Run TGV and compare with DNS data'
    executable = './neko'
    case = 'tgv.case'
    tgv_dns = fixture(GetTgvDns, scope='session')

    @run_after('setup')
    def set_build(self):
        self.build_system = MakeNeko()
        self.sourcepath = 'tgv.f90'

    @sn.deferrable
    def max_error(self, time_ens):
        errs = []
        for time, ens in time_ens:
            # Round time to 3 decimals to find corresponding DNS sample
            time = round(time, 3)
            if time == 20.0:
                # DNS data does not include the last timestep
                continue
            try:
                dns = self.tgv_dns.enstrophy[time]
            except KeyError:
                raise NekoError(f'DNS enstrophy not sampled at {time}')
            errs.append(100 * abs(1 - ens/dns))
        return max(errs)

    @performance_function('%')
    def enstrophy_error(self):
        time_ens = sn.extractall(r'Time: (\S+).*Enstrophy: (\S+)', self.stdout, (1, 2), (float, float))
        return self.max_error(time_ens)


@rfm.simple_test
class Tgv32(TgvBase):
    dofs = 8**3 * 32**3
    # Where flow has become turbulent
    first_workrate_timestep = 12000

    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': {
                'total_runtime': (140, -0.50, 0.05, 's'),
                'enstrophy_error': (9.018, -0.01, 0.01, '%'),
            }
        }

