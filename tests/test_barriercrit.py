#
# test_barriercrit.py
#

import torch
from ceem.opt_criteria import GroupCriterion

from delsmm.barriercrit import MxNormBarrierCriterion
from delsmm.lagcrit import DELCriterion
from delsmm.smm import StructuredMechanicalModel


def test():

    qdim = 2
    B = 2
    T = 10

    t = torch.arange(T).unsqueeze(0).repeat(B, 1).float()
    q = torch.randn(B, T, qdim)

    sys = StructuredMechanicalModel(qdim=qdim, dt=0.1)
    lb = MxNormBarrierCriterion.mmxnorm(
        sys, q).detach() / 10.0  # interior point init

    barriercrit = MxNormBarrierCriterion(lb)
    delcrit = DELCriterion(t)

    crit = GroupCriterion([delcrit, barriercrit])

    print(crit(sys, q))


if __name__ == "__main__":
    test()
