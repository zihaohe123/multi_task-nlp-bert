import os
import time


def prepar_data():
    if not os.path.exists('data'):
        os.mkdir('data')

    # SNLI
    if not (os.path.exists('data/SNLI/train.txt')
            and os.path.exists('data/SNLI/dev.txt')
            and os.path.exists('data/SNLI/test.txt')):
        if not os.path.exists('data/snli_1.0.zip'):
            print('Downloading SNLI....')
            os.system('wget -P data/ https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
        print('Unzipping SNLI....')
        os.system('unzip -d data/ data/snli_1.0.zip')
        os.system('mv -f data/snli_1.0/snli_1.0_train.txt data/snli_1.0/train.txt')
        os.system('mv -f data/snli_1.0/snli_1.0_dev.txt data/snli_1.0/dev.txt')
        os.system('mv -f data/snli_1.0/snli_1.0_test.txt data/snli_1.0/test.txt')
        os.system('mv -f data/snli_1.0 data/SNLI')
    else:
        print('Found SNLI')

    # SST-2
    if not (os.path.exists('data/SST-2/train.tsv')
            and os.path.exists('data/SST-2/dev.tsv')
            and os.path.exists('data/SST-2/test.tsv')):
        if not os.path.exists('data/SST-2.zip'):
            print('Not Found data/SST-2.zip')
            assert False
            # print('Downloading SST-2....')
            # os.system('wget -P data/ https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8')
        print('Unzipping SST-2....')
        os.system('unzip -d data/ data/SST-2.zip')
    else:
        print('Found SST-2')

    # STS-B
    if not (os.path.exists('data/STS-B/train.tsv')
            and os.path.exists('data/STS-B/dev.tsv')
            and os.path.exists('data/STS-B/test.tsv')):
        if not os.path.exists('data/STS-B.zip'):
            print('Not Found data/STS-B.zip')
            assert False
            # print('Downloading STS-B....')
            # os.system('wget -P data/ https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5')
        print('Unzipping STS-B....')
        os.system('unzip -d data/ data/STS-B.zip')
    else:
        print('Found STS-B')

    # QNLI
    if not (os.path.exists('data/QNLI/train.tsv')
            and os.path.exists('data/QNLI/dev.tsv')
            and os.path.exists('data/QNLI/test.tsv')):
        if not os.path.exists('data/QNLIv2.zip'):
            print('Not Found data/QNLI.zip')
            assert False
            # print('Downloading QNLI....')
            # os.system(
            #     'wget -P data/ https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601')
        print('Unzipping QNLI....')
        os.system('unzip -d data/ data/QNLIv2.zip')
    else:
        print('Found QNLI')


def get_current_time():
    return str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))[:-2]


def calc_eplased_time_since(start_time):
    curret_time = time.time()
    seconds = int(curret_time - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    time_str = '{:0>2d}h{:0>2d}min{:0>2d}s'.format(hours, minutes, seconds)
    return time_str


def to_device(*data, device):
    new_data = []
    for each in data:
        each = each.to(device)
        new_data.append(each)
    return new_data


if __name__ == '__main__':
    prepar_data()