import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class deepspeed_bert_fetch_data_tokenizer(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the the dataset and tokenizer'''
    local = True
    modules = ['DeepSpeed']
    executable = 'python bert_squad_deepspeed_train.py --download-only'

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class deepspeed_bert_qa_train_base(rfm.RunOnlyRegressionTest):
    descr = ('Check the training throughput of a BERT with Squad for the QA '
             'task with DeepSpeed')
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    bert_cache = fixture(deepspeed_bert_fetch_data_tokenizer, scope='session')
    sourcesdir = 'src'
    executable = 'python bert_squad_deepspeed_train.py'
    executable_opts = ['--deepspeed_config ds_config.json',
                       '--num-epochs 5']
    num_tasks = 32
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    reference = {
        'lumi:gpu': {
            'samples_per_sec': (5579, -0.05, None, 'samples/sec')}
    }
    variables = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
        'NCCL_NET_GDR_LEVEL': '3',
        'TORCH_EXTENSIONS_DIR': './torch_extensions',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1'
    }

    @sanity_function
    def assert_world_rank(self):
        world_ranks = sn.extractall(r'world_rank=(?P<world_rank>\S+),',
                                    self.stdout, 'world_rank', float)
        return sn.all([
            sn.assert_eq(sorted(world_ranks), list(range(self.num_tasks))),
            sn.assert_found(r'Finished Training', self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec(self):
        return sn.avg(sn.extractall(r'SamplesPerSec=(?P<samples_per_sec>\S+),',
                                    self.stdout, 'samples_per_sec', float))


@rfm.simple_test
class deepspeed_bert_qa_train(deepspeed_bert_qa_train_base):
    # Besides `deepspeed`, the packages`transformers`, `datasets` and `rich`
    # are needed to run this test
    modules = ['DeepSpeed']

    @run_before('run')
    def prepare_job(self):
        bert_cache_dir = os.path.join(self.bert_cache.stagedir, 'cache')
        self.executable_opts.append(f'--bert-cache-dir {bert_cache_dir}')


@rfm.simple_test
class deepspeed_bert_qa_train_singularity(deepspeed_bert_qa_train_base):
    # Using a container from https://hub.docker.com/r/rocm/deepspeed
    # The container used here doesn't include all the packages needed to run
    # this test:
    # MPICC=mpicc pip install --user mpi4py
    # pip install --user datasets transformers tokenizers
    modules = ['OpenMPI']

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed',
            'deepspeed_rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1_deepspeed.sif'  # noqa E501
        )
        # FIXME: copy of `comm.py` that uses `hostname -i`
        # instead of `hostname -I`
        comm_py = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed', 'comm.py'
        )
        # FIXME: The current tag misses the `quantizer_hip.h` ops file
        quantizer_hip = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed', 'quantizer_hip.h'
        )
        deepspeed_dir = '/opt/conda/lib/python3.7/site-packages/deepspeed'
        self.container_platform.mount_points = [
            ('$SCRATCH', '$SCRATCH'),
            (self.bert_cache.stagedir, '/bert_cache_dir'),
            # FIXME: Mounting missing deepspeed files or files needing changes
            (comm_py, os.path.join(deepspeed_dir, 'comm/comm.py')),
            (quantizer_hip, os.path.join(deepspeed_dir, 'ops/csrc/includes/quantizer_hip.h'))  # noqa E501
        ]
        self.container_platform.command = ' '.join([
            self.executable,
            *self.executable_opts,
            '--bert-cache-dir /bert_cache_dir/cache'
        ])

    @run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to `mpirun` since the software
        # in the container is based on OpenMPI and it would fail with `srun`
        self.job.launcher = getlauncher('mpirun')()
