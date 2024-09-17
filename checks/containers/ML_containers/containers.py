import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    descr = 'Pytorch distributed in container'
    valid_systems = ['lumi:gpu']
    container_platform = 'Singularity'
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    exclusive_access = True

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'Nodes':'2',
            'NCCL_NET_GDR_LEVEL':'3',
            'SINGULARITY_BIND':'/var/spool/slurmd:/var/spool/slurmd,'
                                '/opt/cray:/opt/cray,'
                                '/usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1,'
                                '/usr/lib64/libjansson.so.4:/usr/lib64/libjansson.so.4,'
                                f'{self.current_system.resourcesdir}:/rfm_resourcesdir'
        }
        self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

@rfm.simple_test
class test_pytorch_container(singularity_container_image):
    valid_prog_environs = ['builtin']
    cont_image = parameter([
        'rocm-5.7.3-python-3.12-pytorch-v2.2.2-dockerhash-5c27c559a371',
        'rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-c2cfdd3e6ad8',
        'rocm-6.1.3-python-3.12-pytorch-v2.4.0-dockerhash-4a501aa4c1ea',
        'rocm-6.2.0-python-3.10-pytorch-v2.3.0-dockerhash-e84685c13eba',
    ])
    reference = {
            'lumi:gpu': {
                'training_time' : (13.00, None, 0.1, 's'),
                }
            }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'GPU training time', self.stderr)

    @performance_function('s')
    def training_time(self):
        return sn.extractsingle(
            r"GPU training time= \d+:\d+:(\d+\.\d+)$",
            self.stderr, 1, float
        )
    
    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/pytorch'
        self.container_platform.image = os.path.join(
                self.current_system.resourcesdir,
                'containers',
                f'lumi-pytorch-{self.cont_image}.sif',
                )
        self.container_platform.command = 'bash conda-python-distributed.sh -u mnist/mnist_DDP.py --gpu --modelpath model'

@rfm.simple_test
class test_tensorflow_container(singularity_container_image):
    valid_prog_environs = ['builtin']
    reference = {
            'lumi:gpu': {
                'images_sec' : (450.00, -0.1, None, 's'),
                }
            }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Img/sec', self.stdout)

    @performance_function('s')
    def images_sec(self):
        return sn.extractsingle(
            r'(\d+\.\d+)\s+\+-\d+\.\d+',
            self.stdout, 1, float
        )

    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/tensorflow'
        self.container_platform.image = os.path.join(
                self.current_system.resourcesdir,
                'containers',
                'lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1-dockerhash-56a00af8ac92.sif'
                )
        self.container_platform.command = 'bash run-tensorflow.sh'


@rfm.simple_test
class test_container_rccl(singularity_container_image):
    valid_prog_environs = ['builtin']
    reference = {
        'lumi:gpu': {
            'busbw': (55.76, -0.05, None, 'GB/s'),
            'algbw': (29.21, -0.05, None, 'GB/s'),
        }
    }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Avg bus bandwidth', self.stdout)

    @performance_function('GB/s')
    def busbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<busbw>\S+)\s+\S+$',
            self.stdout, 'busbw', float
        )

    @performance_function('GB/s')
    def algbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<algbw>\S+)\s+\S+\s+\S+$',
            self.stdout, 'algbw', float
        )

    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/rccl'
        self.container_platform.image = os.path.join(
                self.current_system.resourcesdir,
                'containers',
                'lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0-dockerhash-e84685c13eba.sif'
                )
        self.container_platform.command = 'bash run.sh'

