import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class tensorflow_distr_cnn(rfm.RunOnlyRegressionTest):
    # The container used here doesn't include all the packages needed to run
    # this test. A requirements.txt file can be found on the src directory
    descr = 'Check the training throughput of a cnn with tf.distributed'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['singularity-bindings']
    sourcesdir = 'src'
    num_tasks = 4
    num_gpus_per_node = 8
    num_tasks_per_node = 1
    throughput_per_gpu = 325.5

    @run_before('run')
    def set_env_variables(self):
        # There are two issues with `tensorflow.distributed` affecting
        # this test: 1. TensorFlow tries to use `nvidia-smi` to detect
        # the number of  gpus and 2. it get's the name of the compute
        # nodes wrong, which makes the communication fail. Here we adapt
        # the file `slurm_cluster_resolver_lumi.py` from TensorFlow to Lumi.
        # The modification requires the user to define the environment
        # variable `LUMI_VISIBLE_DEVICES` as done below.
        tf_slurm_cluster_resulver_file = os.path.join(
            self.current_system.resourcesdir, 'tensorflow',
            'slurm_cluster_resolver_lumi.py')
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'SINGULARITY_BIND': (
                f'"{tf_slurm_cluster_resulver_file}:'
                '/usr/local/lib/python3.9/dist-packages/tensorflow/python'
                '/distribute/cluster_resolver/slurm_cluster_resolver.py,'
                '$SINGULARITY_BIND"'
            ),
            'LUMI_VISIBLE_DEVICES': (
                '$(seq --separator="," 0 $(($SLURM_GPUS_PER_NODE - 1)))'
            )
        }

    @run_after('init')
    def set_references(self):
        throughput_total = self.throughput_per_gpu * self.num_tasks
        self.reference = {
            'lumi:gpu': {
                'samples_per_sec_per_gpu': (self.throughput_per_gpu,
                                            -0.1, None, 'samples/sec'),
                'samples_per_sec_total': (throughput_total,
                                          -0.1, None, 'samples/sec')
            }
        }

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'tensorflow',
            'tensorflow_rocm5.2.0-tf2.9-dev.sif'
        )
        self.container_platform.command = (
            "bash -c '"
            "cd /rfm_workdir; "
            "python tf2_distr_synthetic_benchmark.py --batch-size=256'"
        )

    @sanity_function
    def assert_found_nccl_launch(self):
        return sn.all([
            sn.assert_found(r'NCCL INFO Launch mode Parallel/CGMD',
                            self.stdout),
            sn.assert_found(r'img/sec on \d+ GPU\(s\): \S+ \+',
                            self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec_per_gpu(self):
        return sn.extractsingle(
            r'Img/sec per GPU: (?P<samples_per_sec_per_gpu>\S+) \+',
            self.stdout, 'samples_per_sec_per_gpu', float
        )

    @performance_function('samples/sec')
    def samples_per_sec_total(self):
        return sn.extractsingle(
            r'img/sec on \d+ GPU\(s\): (?P<samples_per_sec_total>\S+) \+',
            self.stdout, 'samples_per_sec_total', float
        )


# @rfm.simple_test
# class tensorflow_keras_distr_cnn(tensorflow_distr_cnn):
#     throughput_per_gpu = 441.4
#
#     @run_before('run')
#     def set_container_variables(self):
#         super().set_container_variables()
#         self.container_platform.command = (
#             "bash -c '"
#             "cd /rfm_workdir; "
#             "python tf2_keras_distr_synthetic_benchmark.py --batch-size=256'"
#         )
