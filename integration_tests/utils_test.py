import os

from integration_tests.bchain.getdata_test import temp_dir

import trufflepig.utils as tfut


def test_clean_up(temp_dir):
    for irun in range(5):
        with open(os.path.join(temp_dir, 'test_{}.txt'.format(irun)), 'w') as fh:
            fh.write('Hello World!')

    filenames = sorted(os.listdir(temp_dir))
    assert len(filenames) == 5

    tfut.clean_up_directory(temp_dir, keep_last=2)

    filenames = sorted(os.listdir(temp_dir))
    assert len(filenames) == 2
    assert filenames[0].endswith('test_3.txt')
    assert filenames[1].endswith('test_4.txt')


