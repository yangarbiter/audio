import os
import unittest

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO
from torchaudio.datasets.utils import DiskCache, download_url, download_url


class TestDatasets(unittest.TestCase):
    path = "assets"

    def test_yesno(self):
        data = YESNO(self.path, return_dict=True)
        data[0]

    def test_vctk(self):
        data = VCTK(self.path, return_dict=True)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_commonvoice(self):
        path = os.path.join(self.path, "commonvoice")
        data = COMMONVOICE(path, "train.tsv", "tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        path = os.path.join(self.path, "commonvoice")
        data = COMMONVOICE(path, "train.tsv", "tatar")
        data = DiskCache(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_download_url(self):

        url = "http://www.patentsview.org/data/20171226/botanic.tsv.zip"
        hash_url = "94c642405619b20ecaf657b30e84bab787320649e751ed6ac629c0be613ded44"

        download_folder = "."
        download_url(url, download_folder)


if __name__ == "__main__":
    unittest.main()
