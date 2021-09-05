from argparse import ArgumentParser

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-e', '--epoch', type=int, default=150,
                           help='Specify number of epoch')
    argparser.add_argument('--channel', type=int, default=3)
    argparser.add_argument('--batch_size', type=int, default=3)
    argparser.add_argument('--num_cls', type=int, default=1)
    argparser.add_argument('--weight_dir', type=str, default='weights')
    return argparser.parse_args()
