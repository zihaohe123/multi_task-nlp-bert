import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--hidden_size', type=int, default=300)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--grad_max_norm', type=float, default=0.)  #

    parser.add_argument('--dropout_emb', type=float, default=0.3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test', action='store_true', default=False, help='Whether to just test the model')
    parser.add_argument('--multi_task', action='store_true', default=False, help='Whether to use multi-task learning')

    parser.add_argument('--gpu', type=str, default='', help='which gpus to use')
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)
    return args