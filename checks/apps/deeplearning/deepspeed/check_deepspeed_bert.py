import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class deepspeed_bert_qa_train_base(rfm.RunOnlyRegressionTest):
    descr = ('Check the training throughput of a BERT with Squad for the QA '
             'task with DeepSpeed')
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
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
        'TORCH_EXTENSIONS_DIR': '$SCRATCH/torch_extensions'
    }

    @run_after('init')
    def prepare_build(self):
        resourcesdir = os.path.join(self.current_system.resourcesdir,
                                    'deepspeed', 'bert-base-uncased', 'cache')
        self.prerun_cmds.append(f'cp -r {resourcesdir} .')

    @sanity_function
    def assert_world_rank(self):
        world_ranks = sn.extractall(r'world_rank=(?P<world_rank>\S+),',
                                    self.stdout, 'world_rank', float)
        return sn.all([
            sn.assert_eq(sorted(world_ranks), list(range(self.num_tasks))),
            sn.assert_found(r'Training complete', self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec(self):
        return sn.avg(sn.extractall(r'SamplesPerSec=(?P<samples_per_sec>\S+),',
                                    self.stdout, 'samples_per_sec', float))


@rfm.simple_test
class deepspeed_bert_qa_train(deepspeed_bert_qa_train_base):
    modules = ['PyTorch']
    prerun_cmds = ['module unload aws-ofi-rccl',
                   'export TRANSFORMERS_OFFLINE=1',
                   '. $SCRATCH/deeepspeed-env/bin/activate',
                   'export HF_DATASETS_OFFLINE=1']
    executable = 'python bert_squad_deepspeed_train.py --deepspeed_config ds_config.json --num-epochs 5'


# @rfm.simple_test
class deepspeed_bert_qa_train_singularity(deepspeed_bert_qa_train_base):
    # The container used here doesn't include all the packages needed to run
    # this test:
    # MPICC=mpicc pip install --user mpi4py
    # pip install --user datasets transformers tokenizers
    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed',
            'deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif'
        )
        self.container_platform.mount_points = [('$SCRATCH', '$SCRATCH')]
        self.container_platform.command = (
            "bash -c '"
            "cd /rfm_workdir; "
            "export TRANSFORMERS_OFFLINE=1; "
            "export HF_DATASETS_OFFLINE=1; "
            "python bert_squad_deepspeed_train.py "
            "--deepspeed_config ds_config.json --num-epochs 5'"
        )

    @run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to `mpirun` since the software
        # in the container is based on OpenMPI and it would fail with `srun`
        self.job.launcher = getlauncher('mpirun')()
