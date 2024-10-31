import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ELPA_Cholesky_GPU(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeCray']
    modules = ['ELPA', 'rocm']
    build_system = 'SingleSource'
    # This is stripped version of https://gitlab.mpcdf.mpg.de/elpa/elpa/-/blob/master/test/Fortran/test.F90
    # manually processed for cholesky solver only
    sourcepath = 'elpa_fortran_validate_real_double_cholesky_1stage_gpu_random.F90'
    executable = 'elpa_test'
    maintainers = ['mszpindler']
    num_gpus_per_node = 1
    tasks = 1

    tags = {'production', 'contrib'}

    reference = {
        'lumi:gpu': {'gpublas_timing': (0.6, -0.5, 0.5, 's')},
    }

    @run_before('compile')
    def set_compile_flags(self):
#ftn elpa_fortran_validate_real_double_cholesky_1stage_gpu_random.F90 ../elpa-new_release_2024.05.001/src/helpers/libelpa_private_la-mod_precision.o -o test -I../elpa-new_release_2024.05.001/modules  -L../elpa-new_release_2024.05.001/.libs -lelpa
        #self.build_system.cxxflags = ['-x hip']
        self.build_system.fflags = ['-I${EBROOTELPA}/include/elpa-${EBVERSIONELPA}/modules']
        self.build_system.ldflags = ['-L${EBROOTELPA}/lib -lelpa']

    @run_before('run')
    def set_env(self):
        self.env_vars = {'ELPA_DEFAULT_real_kernel': 'ELPA_2STAGE_REAL_AMD_GPU'}

    @sanity_function
    def validate_solution(self):
        num_devices = sn.count(sn.findall(r'^device#', self.stdout))
        return sn.assert_found(r'\s+\S+\s+elpa_cholesky_real_double_gpu', self.stdout)

    @performance_function('s')
    def gpublas_timing(self):
        return sn.extractsingle(r'\s+\S+\s+gpublas\s+\S+\s+(?P<time>\S+)',
                                   self.stdout, 'time', float)

