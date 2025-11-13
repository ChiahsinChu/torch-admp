# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import freud
import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList, dp_nblist, sort_pairs, vesin_nblist
from torch_admp.utils import to_numpy_array


class TestTorchNeighborList(unittest.TestCase):
    def setUp(self):
        # reference data
        rcut = 4.0
        l_box = 10.0
        box = np.diag([l_box, l_box, l_box])
        positions = np.random.rand(20, 3) * l_box

        fbox = freud.box.Box.from_matrix(box)
        aq = freud.locality.AABBQuery(fbox, positions)
        res = aq.query(positions, dict(r_max=rcut, exclude_ii=True))
        nblist = res.toNeighborList()
        nblist = np.vstack((nblist[:, 0], nblist[:, 1])).T
        nblist = nblist.astype(np.int32)
        msk = (nblist[:, 0] - nblist[:, 1]) < 0
        self.nblist_ref = nblist[msk]

        self.nblist = TorchNeighborList(rcut)
        self.positions = torch.tensor(positions)
        self.box = torch.tensor(box)

    def test_pairs(self):
        """
        Check that pairs are in the neighbor list.
        """
        pairs = self.nblist(self.positions, self.box)
        pairs = to_numpy_array(pairs)
        mask = pairs[:, 0] < pairs[:, 1]
        assert len(pairs[mask]) == len(self.nblist_ref)
        for p in pairs[mask]:
            mask = (self.nblist_ref[:, 0] == p[0]) & (self.nblist_ref[:, 1] == p[1])
            self.assertTrue(mask.any())


class TestNBList(unittest.TestCase):
    """Test nblist"""

    def setUp(self) -> None:
        # reference data
        l_box = 10.0
        box = np.diag([l_box, l_box, l_box])
        positions = np.random.rand(100, 3) * l_box

        self.positions = torch.tensor(positions)
        self.box = torch.tensor(box)

        # test: cutoff > l_box / 2!!!
        self.rcut = 6.0
        self.nnei = 150

    def test_consistent(self):
        pairs_1, ds_1, _buffer_scales = dp_nblist(
            self.positions, self.box, self.nnei, self.rcut
        )
        pairs_2, ds_2, _buffer_scales = vesin_nblist(
            self.positions, self.box, self.rcut
        )
        torch.testing.assert_close(sort_pairs(pairs_1), sort_pairs(pairs_2))
        torch.testing.assert_close(torch.sort(ds_1)[0], torch.sort(ds_2)[0])


if __name__ == "__main__":
    unittest.main()
