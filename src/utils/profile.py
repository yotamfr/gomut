import h5py
import os.path as osp
from utils.data import DATA_HOME
from utils.constants import amino_acids
from utils.torch_utils import *
from utils.sequence import FASTA

MSAS_PATH = osp.join(DATA_HOME, 'msas')

PERM = 'r'
PROFS = h5py.File(osp.join(DATA_HOME, 'h5', 'profiles.h5'), PERM)


def read_profile_from_msa(pdb_id, msa_path=MSAS_PATH):
    _, fas = FASTA(osp.join(msa_path, '%s.a3m' % pdb_id))
    length, num_seqs = len(fas[pdb_id].seq), len(fas)
    msa = np.asarray([np.asarray(list(seq.seq))[:length] for seq in fas.values()])
    msa = msa.astype('<U1')
    profile = np.zeros((20, length))
    for i, aa in enumerate(amino_acids):
        profile[i, :] = np.sum((msa == aa).astype(np.float), 0)
    profile = np.divide(profile, num_seqs)
    return profile


def get_profile(pdb_id):
    if pdb_id in PROFS:
        profile_t = torch.tensor(PROFS[pdb_id][:], dtype=torch.float, device=device)
    else:
        profile = read_profile_from_msa(pdb_id)
        profile_t = torch.tensor(profile, dtype=torch.float, device=device)
        if PERM == 'a':
            PROFS.create_dataset(pdb_id, data=profile)
    return profile_t.transpose(0, 1)
