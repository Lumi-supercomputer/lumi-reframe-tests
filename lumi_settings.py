# Reframe settings for LUMI 


site_configuration = {
    'systems': [
        {
            'name': 'lumi',
            'descr': 'lumi Supercomputer',
            'hostnames': ['uan01','uan02', 'uan03', 'uan04'],
            'modules_system': 'lmod',
            'resourcesdir': '/project/project_462000008/jigong2/resources/reframe/',
            'prefix': '/scratch/project_462000008/jigong2/reframe/',
            'partitions': [
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'environs': ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-aocc'],
                    'max_jobs': 32,
                    'time_limit': '30m',
                    'access': [
                          f'-p standard',
                          f'-A project_462000008'
                    ],
                    'processor': {
                        'num_cpus': 256,
                        'num_cpus_per_core': 2,
                        'num_cpus_per_socket': 128,
                        'num_sockets': 2
                    }
                },
                {
                    'name': 'gpu',
                    'descr': 'GPU nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'time_limit': '60m',
                    'access': [
                          f'-p dev-g',
                          f'-A project_462000008',
                          f'--exclusive'
                    ],
                    'environs': ['PrgEnv-gnu','PrgEnv-cray'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'amd',
                            'num_devices': 8
                        }
                    ],
                    'extras': {
                        'select_device': './rocm_select_gpu_device'
                    },
                    'container_platforms': [
                    {
                        'type': 'Singularity',
                        'modules': ['singularity/3.10.4-cpeGNU-22.06']
                    }
                    ],
                    'env_vars': [
                        ['MPICH_GPU_SUPPORT_ENABLED', '1']
                    ],
                    'resources': [
                       {
                        'name': 'gpu',
                        'options': ['--gpus-per-node={num_gpus_per_node}']
                        }
                    ]
                }

            ]
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['lumi']
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['lumi']
        },
        {
            'name': 'PrgEnv-aocc',
            'modules': ['PrgEnv-aocc'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['lumi']
        }
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_job_completion_time)s|reframe %(version)s|'
                        '%(check_info)s|jobid=%(check_jobid)s|'
                        '%(check_perf_var)s=%(check_perf_value)s|'
                        'ref=%(check_perf_ref)s '
                        '(l=%(check_perf_lower_thres)s, '
                        'u=%(check_perf_upper_thres)s)|'
                        '%(check_perf_unit)s'
                    ),
                    'append': True
                }
            ]
        }
    ],
}
