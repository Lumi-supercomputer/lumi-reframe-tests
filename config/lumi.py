project = 'project_462000008'

site_configuration = {
    'systems': [
        {
            'name': 'lumi',
            'descr': 'LUMI Cray EX Supercomputer',
            'hostnames': ['ln\d+-nmn', 'uan\d+-nmn.local', '\S+'],
            'modules_system': 'lmod',
            'modules': ['LUMI'],
            'resourcesdir': '/projappl/%s/reframe_resources/' % project,
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'modules': ['partition/L'],
                    'launcher': 'local'
                },
                {
                    'name': 'small',
                    'descr': 'Multicore nodes (AMD EPYC 7763, 256|512|1024GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAOCC',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['partition/C'],
                    'access': ['--partition small',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                    },
                    {
                    'name': 'standard',
                    'descr': 'Multicore nodes (AMD EPYC 7763, 256GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAOCC',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['partition/C'],
                    'access': ['--partition standard',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'gpu',
                    'descr': 'Multicore nodes (AMD EPYC 7A53 64-Core, 512|GB/cn), GPU (AMD Instinct MI250X 8/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-amd',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'ROCm',
                    ],
                    'max_jobs': 10,
                    'modules': ['partition/G'],
                    'access': ['--partition small-g',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gpus-per-node={num_gpus_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
            ]
        },
    ],
    'environments': [
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['lumi:small', 'lumi:standard'],
            'modules': ['PrgEnv-amd']
        },
        {
            'name': 'PrgEnv-amd',
            'target_systems': ['lumi:gpu'],
            'modules': ['PrgEnv-amd']
        },
        {
            'name': 'PrgEnv-cray',
            'target_systems': ['lumi'],
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['lumi'],
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'cpeAMD',
            'target_systems': ['lumi:gpu'],
            'modules': ['cpeAMD']
        },
        {
            'name': 'cpeAOCC',
            'target_systems': ['lumi:small', 'lumi:standard'],
            'modules': ['cpeAOCC']
        },
        {
            'name': 'cpeCray',
            'target_systems': ['lumi'],
            'modules': ['cpeCray']
        },
        {
            'name': 'cpeGNU',
            'target_systems': ['lumi'],
            'modules': ['cpeGNU']
        },
        {
            'name': 'ROCm',
            'cc': 'hipcc',
            'cxx': 'hipcc',
            'ftn': '',
            #'cflags': 
            #'ldflags': 
            'cppflags': ['-D__HIP_PLATFORM_AMD__'],
            'modules': ['rocm'],
            'target_systems': ['lumi:gpu']
        }
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '%(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|num_tasks=%(check_num_tasks)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',   # noqa: E501
                    'datefmt': '%FT%T%:z',
                    'append': True
                },
            ]
        }
    ],
    'modes': [
        {
            'name': 'maintenance',
            'options': [
                '--exec-policy=async',
                '--strict',
                '--output=/project/%s/$USER/regression/maintenance' % project,
                '--perflogdir=/project/%s/$USER/regression/maintenance/logs' % project,
                '--stage=/scratch/%s/regression/maintenance/stage' % project,
                '--report-file=/project/%s/$USER/regression/maintenance/reports/maint_report_{sessionid}.json' % project,
                '-Jreservation=maintenance',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--exec-policy=async',
                '--strict',
                '--output=/project/%s/$USER/regression/production' % project,
                '--perflogdir=/project/%s/$USER/regression/production/logs' % project,
                '--stage=/scratch/%s/regression/production/stage' % project,
                '--report-file=/project/%s/$USER/regression/production/reports/prod_report_{sessionid}.json' % project,
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ]
        }
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True,
            'remote_detect': False
        }
    ]
}
