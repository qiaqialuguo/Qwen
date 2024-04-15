from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        default='/opt/large-model/qwen/qwen1/Qwen-1_8B-Chat',
        help='Checkpoint name or path, default to %(default)r',
    )
    parser.add_argument('--server-port',
                        type=int,
                        default=10019,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='0.0.0.0',
        help=
        'Demo server name. Default: 0.0.0.0',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args
