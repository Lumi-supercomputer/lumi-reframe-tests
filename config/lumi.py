project = 'project_462000008'

site_configuration = {
    'systems': [
        {
            'name': 'lumi',
            'descr': 'LUMI Cray EX Supercomputer',
            'hostnames': ['ln\d+-nmn', 'uan\d+-nmn.local', '\S+'],
            'modules_system': 'lmod',
            'resourcesdir': '/projappl/%s/reframe_resources/' % project,
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'modules': ['LUMI', 'partition/L'],
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
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['LUMI', 'partition/C'],
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
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['LUMI', 'partition/C'],
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
                    'name': 'eap',
                    'descr': 'Multicore nodes (AMD EPYC 7662, 256|512|1024GB/cn), GPU (AMD Instinct MI100 4/cn)',
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
                        'builtin-hip',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                    ],
                    'max_jobs': 10,
                    'modules': ['LUMI/21.12', 'partition/EAP', 'rocm'],
                    'access': ['--partition eap',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gres=gpu:mi100:{num_gpus_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                    },
                    {
                    'name': 'gpu',
                    'descr': 'Multicore nodes (AMD EPYC 7A53 64-Core, 512|GB/cn), GPU (AMD Instinct MI250 8/cn)',
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
                        'builtin-hip',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                    ],
                    'max_jobs': 10,
                    'modules': ['LUMI/21.12', 'partition/G'],
                    'access': ['--partition gpu',
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
        {
            'name': 'generic',
            'descr': 'Generic fallback system',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'environs': ['builtin'],
                    'descr': 'Login nodes',
                    'launcher': 'local'
                }
            ],
            'hostnames': ['.*']
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['lumi'],
            'modules': ['cpeAMD']
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
             'name': 'PrgEnv-intel',
             'modules': ['PrgEnv-intel']
         },
        {
            'name': 'cpeAMD',
            'target_systems': ['lumi'],
            'modules': ['cpeAMD']
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
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn'
        },
        {
            'name': 'builtin-gcc',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'builtin-hip',
            'cc': 'hipcc',
            'cxx': 'hipcc',
            'ftn': '',
            'cflags': ['-I$MPICH_DIR/include'],
            'ldflags': ['-L$MPICH_DIR/lib', '-lmpi', '-L$CRAY_MPICH_ROOTDIR/gtl/lib/', '-lmpi_gtl_hsa'],
            'cppflags': ['-D__HIP_PLATFORM_AMD__'],
            #'modules': ['rocm'],
            'target_systems': ['lumi']
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
                '-Jnodelist=nodelist',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--unload-module=reframe',
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
