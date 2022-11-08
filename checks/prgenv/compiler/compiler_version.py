import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CCompilerVersion(rfm.RegressionTest):
    descr = 'Checks for C compiler version executed by cc wrapper'
    valid_systems = ['lumi:small']
    valid_prog_environs = ['*']
    build_system = 'SingleSource'
    sourcepath = 'cc_version.c'

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}

    @sanity_function
    def validate_solution(self):
        return sn.assert_gt(sn.len(sn.extractsingle(r'.*',
                            self.stdout, tag=0)), 0)

@rfm.simple_test
class FCompilerVersion(rfm.RegressionTest):
    descr = 'Checks for Fortran compiler version executed by ftn wrapper'
    valid_systems = ['lumi:small']
    valid_prog_environs = ['*']
    build_system = 'SingleSource'
    sourcepath = 'ftn_version.F90'

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}


    @sanity_function
    def validate_solution(self):
        return sn.assert_gt(sn.len(sn.extractsingle(r'.*',
                            self.stdout, tag=0)), 0)

@rfm.simple_test
class HIPCompilerVersion(rfm.RegressionTest):
    descr = 'Checks for HIP compiler version executed by hipcc command'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin-hip']
    build_system = 'SingleSource'
    sourcepath = 'cc_version.c'

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}

    @sanity_function
    def validate_solution(self):
        return sn.assert_gt(sn.len(sn.extractsingle(r'.*',
                            self.stdout, tag=0)), 0)
