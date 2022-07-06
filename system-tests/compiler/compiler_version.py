import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CCompilerVersion(rfm.RegressionTest):
    descr = 'Checks for C compiler version executed by cc wrapper'
    valid_systems = ['lumi:small', 'lumi:eap']
    valid_prog_environs = ['cpeCray', 'cpeGNU', 'builtin-hip']
    build_system = 'SingleSource'
    sourcepath = 'cc_version.c'
    #build_locally = False

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}

    #@run_before('compile')
    #def setflags(self):

    @sanity_function
    def validate_solution(self):
        return sn.assert_gt(sn.len(sn.extractsingle(r'.*',
                            self.stdout, tag=0)), 0)

@rfm.simple_test
class FCompilerVersion(rfm.RegressionTest):
    descr = 'Checks for Fortran compiler version executed by ftn wrapper'
    valid_systems = ['lumi:small']
    valid_prog_environs = ['cpeCray', 'cpeGNU']
    build_system = 'SingleSource'
    sourcepath = 'ftn_version.F90'

    maintainers = ['mszpindler']
    tags = {'production', 'craype'}


    @sanity_function
    def validate_solution(self):
        return sn.assert_gt(sn.len(sn.extractsingle(r'.*',
                            self.stdout, tag=0)), 0)
