import h5py
import os.path as osp

from utils.sequence import *
from utils.constants import *
from utils.torch_utils import *
MSAS_PATH = osp.join('data', 'msas')

PERM = 'r'
PROFS = h5py.File('data/h5/profiles.h5', PERM)


def get_profile(pdb_id, msa_path=MSAS_PATH):
    if pdb_id in PROFS:
        profile = torch.tensor(PROFS[pdb_id][:], dtype=torch.float, device=device)
    else:
        _, fas = FASTA(osp.join(msa_path, '%s.a3m' % pdb_id))
        length, num_seqs = len(fas[pdb_id].seq), len(fas)
        msa = np.asarray([np.asarray(list(seq.seq))[:length] for seq in fas.values()])
        msa = msa.astype('<U1')
        profile = torch.zeros(20, length, device=device, dtype=torch.float)
        for i, aa in enumerate(amino_acids):
            profile[i, :] = torch.tensor((msa == aa).astype(np.float), device=device, dtype=torch.float).sum(0)
        profile.div_(num_seqs)
        PROFS.create_dataset(pdb_id, data=profile.data.cpu().numpy().astype(np.float))
    return profile.transpose(0, 1)
