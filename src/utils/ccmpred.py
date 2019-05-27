from src.utils import DATA_HOME

import os
import sys
import h5py
import torch
import numpy as np
from src.utils.torch_utils import device
from src.utils.sequence import FASTA

PERM = 'r'


CCM_HOME = "/home/yotamfr/development/ccmpred"
CCMS = h5py.File('%s/h5/ccms.h5' % DATA_HOME, PERM)

"""
$CCM_HOME/scripts/convert_alignment.py data/msas/9xim_D.a3m fasta data/ccm/9xim_D.aln
$CCM_HOME/bin/ccmpred data/ccm/9xim_D.aln data/ccm/9xim_D.mat
"""


def read_mat(mat_fname):
    with open(mat_fname, 'r') as f:
        return np.asarray([[float(f) for f in line.strip().split('\t')] for line in f])


def compute_memory_consumption_in_bytes(N, L):
    return 4 * (4 * (L * L * 21 * 21 + L * 20) + 23 * N * L + N + L * L) + 2 * N * L + 1024


def run_ccmppred(pdb_id, compute=True, ccm_home=CCM_HOME, data_home=DATA_HOME, max_cuda_mem_gb=8.0, num_threads=16):
    a3m_fname = "%s/msas/%s.a3m" % (data_home, pdb_id)
    aln_fname = "%s/ccm/%s.aln" % (data_home, pdb_id)
    mat_fname = "%s/ccm/%s.mat" % (data_home, pdb_id)
    if os.path.exists(mat_fname):
        return mat_fname
    if not compute:
        raise FileNotFoundError("file '%s' not found" % mat_fname)
    if not os.path.exists(a3m_fname):
        raise FileNotFoundError("file '%s' not found" % a3m_fname)
    if not os.path.exists(aln_fname):
        cline = '%s %s/scripts/convert_alignment.py %s fasta %s' % (sys.executable, ccm_home, a3m_fname, aln_fname)
        assert os.WEXITSTATUS(os.system(cline)) == 0
    _, fas = FASTA(a3m_fname)
    L, N = len(fas[pdb_id].seq), len(fas)
    mem = compute_memory_consumption_in_bytes(L, N) / 1e9
    if mem > max_cuda_mem_gb:
        cline = '%s/bin/ccmpred -t %d %s %s > /dev/null' % (ccm_home, num_threads, aln_fname, mat_fname)
        assert os.WEXITSTATUS(os.system(cline)) == 0
    else:
        cline = '%s/bin/ccmpred -d 0 %s %s > /dev/null' % (ccm_home, aln_fname, mat_fname)
        assert os.WEXITSTATUS(os.system(cline)) == 0
    return mat_fname


def get_ccmpred(pdb_id, compute, to_tensor=True):
    if pdb_id in CCMS:
        mat = CCMS[pdb_id][:]
    else:
        mat = read_mat(run_ccmppred(pdb_id, compute=compute))
        if PERM == 'a':
            CCMS.create_dataset(pdb_id, data=mat)
    if to_tensor:
        mat = torch.tensor(mat, dtype=torch.float, device=device)
    return mat
