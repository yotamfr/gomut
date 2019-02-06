import h5py
import random
from prody import *
from torch_utils import *
import pandas as pd
from tqdm import tqdm
from ast import literal_eval as make_tuple
from collections import OrderedDict
from build_datasets import FASTA
from constants import *

np.seterr('raise')
confProDy(verbosity='none')
random.seed(101)

REPO_PATH = os.path.join('data', 'pdbs_gz')
PAIRS_PATH = os.path.join('data', 'pairs.csv')
FPAIRS_PATH = os.path.join('data', 'failed_pairs.csv')

PDB_FPAIRS = pd.read_csv(FPAIRS_PATH).values.tolist()
PDB_FPAIRS_SET = set([(p[0], p[1]) for p in PDB_FPAIRS])
PDB_PAIRS = pd.read_csv(PAIRS_PATH).sort_values(by=['length'])
PDB_PAIRS = PDB_PAIRS[PDB_PAIRS.num_mutated == 1].values
PDB_PAIRS = PDB_PAIRS[~np.asarray([tuple(p) in PDB_FPAIRS_SET for p in PDB_PAIRS[:, :2]])]

PRECOMP = True
IDX = h5py.File('data/h5/idx.h5', 'a') if PRECOMP else None
COORDS = h5py.File('data/h5/coords.h5', 'a') if PRECOMP else None
RESNAMES = h5py.File('data/h5/resnames.h5', 'a') if PRECOMP else None
ATOMNAMES = h5py.File('data/h5/atomnames.h5', 'a') if PRECOMP else None

_, SEQS = FASTA('data/pdb_seqres.txt')

MAX_BATCH_SIZE = 32


class Atom(object):

    def __init__(self, name, coords):
        self.coords = coords
        self.name = name

    def getName(self):
        return self.name

    def getCoords(self):
        return self.coords


class Residue(object):

    def __init__(self, resname, atoms=[]):
        self.atoms = OrderedDict([(a.name, a) for a in atoms])
        self.resname = resname

    def getCoords(self):
        return [a.coords for a in self.atoms.values()]

    def select(self, selection_string='sc'):
        return Residue(self.resname, [a for a in self.atoms.values() if a.name not in BACKBONE])

    def getResname(self):
        return self.resname

    def __getitem__(self, item):
        return self.atoms.get(item, None)


def toX(residues):
    return [get_center(res) for res in residues]


def filter_residues(residues):
    return [res for res in residues if res.getResname() != 'HOH']


def get_center(res):
    if res.getResname() == 'GLY':
        if res['CA']:
            return res['CA'].getCoords()
        else:
            return np.mean(res.getCoords(), axis=0)
    elif res['CB']:
        return res['CB'].getCoords()
    elif res.select('sc') and len(res.select('sc').getCoords()) != 0:
        return np.mean(res.select('sc').getCoords(), axis=0)
    elif res['CA']:
        return res['CA'].getCoords()
    else:
        return np.mean(res.getCoords(), axis=0)


def get_contact_map(X, thr=8.0):
    pmat = pairwise_distances(torch.tensor(X, dtype=torch.float, device=device))
    return (pmat < thr) & (pmat > 0.0)


def get_distance_matrix(X):
    return pairwise_distances(torch.tensor(X, dtype=torch.float, device=device))


def convert_to_indices_sequence(seq):
    return torch.tensor([amino_acids.index(aa) for aa in seq], dtype=torch.long, device=device)


def get_contact_map_cpu(X, thr=8.0):
    pmat = pairwise_distances_cpu(np.asarray(X, dtype=np.float))
    return ((pmat < thr) & (pmat > 0.0)).astype(np.float)


def get_distance_matrix_cpu(X):
    return pairwise_distances_cpu(np.asarray(X, dtype=np.float))


def convert_to_indices_sequence_cpu(seq):
    return np.asarray([amino_acids.index(aa) for aa in seq], dtype=np.long)


def prepare_numpy_batch(data):
    x1, x2, y1, y2, ix, *_ = zip(*data)
    x1 = torch.tensor(np.stack(x1, 0), dtype=torch.float, device=device)
    x2 = torch.tensor(np.stack(x2, 0), dtype=torch.float, device=device)
    y1 = torch.tensor(np.stack(y1, 0), dtype=torch.float, device=device)
    y2 = torch.tensor(np.stack(y2, 0), dtype=torch.float, device=device)
    ix = torch.tensor(ix, dtype=torch.long, device=device)
    return x1, x2, y1, y2, ix


def prepare_torch_batch(data):
    x1, x2, y1, y2, ix, *_ = zip(*data)
    x1 = torch.stack(x1, 0)
    x2 = torch.stack(x2, 0)
    y1 = torch.stack(y1, 0)
    y2 = torch.stack(y2, 0)
    ix = torch.tensor(ix, dtype=torch.long, device=device)
    return x1, x2, y1, y2, ix


def batch_generator(loader, prepare):
    batch = []
    curr_length = None
    for inp in loader:
        x, *_ = inp
        if curr_length is None:
            curr_length = len(x)
        if (len(x) != curr_length) or (len(batch) == MAX_BATCH_SIZE):
            if batch:
                yield prepare(batch)
            curr_length = len(x)
            batch = [inp]
        else:
            batch.append(inp)
    if batch:
        yield prepare(batch)


def tostr(pdb1, chain_id1):
    return "%s_%s" % (pdb1, chain_id1)


def store_residues(st1, pdb1, chain_id1):
    residues = list(st1.select('protein')[chain_id1])
    data = [[i, j, r.getResname(), a.getName(), a.getCoords()]
            for i, r in enumerate(residues) for j, a in enumerate(r)]
    idx, _, resnames, atomnames, coords = zip(*data)
    IDX.create_dataset(tostr(pdb1, chain_id1), data=np.array(idx).astype(np.long))
    COORDS.create_dataset(tostr(pdb1, chain_id1), data=np.array(coords).astype(np.float))
    RESNAMES.create_dataset(tostr(pdb1, chain_id1), data=np.array(resnames).astype('|S9'))
    ATOMNAMES.create_dataset(tostr(pdb1, chain_id1), data=np.array(atomnames).astype('|S9'))
    return residues


def load_residues(pdb1, chain_id1):
    idx = IDX[tostr(pdb1, chain_id1)][:]
    coords = COORDS[tostr(pdb1, chain_id1)][:]
    resnames = RESNAMES[tostr(pdb1, chain_id1)][:].astype('U13')
    atomnames = ATOMNAMES[tostr(pdb1, chain_id1)][:].astype('U13')
    residues = {i: Residue(name) for i, name in zip(idx, resnames)}
    for i, name, xyz in zip(idx, atomnames, coords):
        residues[i].atoms[name] = Atom(name, xyz)
    return [residues[i] for i in set(idx)]


def load_or_parse_residues(pdb1, chain_id1):
    if PRECOMP and tostr(pdb1, chain_id1) in COORDS:
        residues = load_residues(pdb1, chain_id1)
        return residues
    src_path = os.path.join(REPO_PATH, '%s.pdb.gz' % pdb1)
    if not os.path.exists(src_path):
        return None
    st1, h1 = parsePDB(src_path, header=True, chain=chain_id1)
    if not (h1['experiment'] == 'X-RAY DIFFRACTION'):
        return None
    if st1 is None:
        return None
    residues = store_residues(st1, pdb1, chain_id1)
    return residues


def find_maximal_matching_shift(seq1, seq2):
    assert len(seq1) == len(seq2)
    shift = 0
    diff = np.where(seq1 != seq2)[0]
    for s in range(1, len(seq1)//2):
        d = np.where(seq1[s:] != seq2[:-s])[0]
        if len(d) == 0:
            break
        if len(d) < len(diff):
            shift = s
            diff = d
    return shift, diff


def handle_failure(pdb1, pdb2, reason):
    if (pdb1, pdb2) not in PDB_FPAIRS_SET:
        PDB_FPAIRS.append([pdb1, pdb2, reason])
        PDB_FPAIRS_SET.add((pdb1, pdb2))


def pairs_loader(list_of_pairs, n_iter, shuffle=False):

    if shuffle:
        random.shuffle(list_of_pairs)

    i_iter = 0
    N = len(list_of_pairs)
    i_pdb = np.random.randint(0, len(list_of_pairs))

    while i_iter < n_iter:
        pdb_id1, pdb_id2, mutated, _, length = list_of_pairs[i_pdb]
        mutated = make_tuple(mutated)
        pdb1, chain_id1 = pdb_id1.split('_')
        pdb2, chain_id2 = pdb_id2.split('_')
        i_pdb = (i_pdb + 1) % N

        residues1 = load_or_parse_residues(pdb1, chain_id1)
        if residues1 is None:
            handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb1)
            continue
        residues2 = load_or_parse_residues(pdb2, chain_id2)
        if residues2 is None:
            handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb2)
            continue

        X1, X2, idx = zip(*[(x1, x2, i) for i, (x1, x2) in enumerate(zip(toX(residues1), toX(residues2)))
                            if (isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray))])

        if not set(mutated).issubset(set(idx)):
            handle_failure(pdb_id1, pdb_id2, 'residue(s): %s are missing' % mutated)
            continue

        try:
            seq1 = np.asarray([AA_dict[res.getResname()] for res in residues1])[np.asarray(idx)]
            seq2 = np.asarray([AA_dict[res.getResname()] for res in residues2])[np.asarray(idx)]
        except KeyError as e:
            handle_failure(pdb_id1, pdb_id2, 'KeyError: %s' % str(e))
            continue

        diff = np.where(seq1 != seq2)[0]
        if len(diff) != len(mutated):
            shift1, diff1 = find_maximal_matching_shift(seq1, seq2)
            shift2, diff2 = find_maximal_matching_shift(seq2, seq1)
            if len(diff1) == len(mutated):
                seq1, X1 = seq1[shift1:], X1[shift1:]
                seq2, X2 = seq2[:-shift1], X2[:-shift1]
                diff = diff1
            elif len(diff2) == len(mutated):
                seq2, X2 = seq1[shift2:], X2[shift2:]
                seq1, X1 = seq1[:-shift2], X1[:-shift2]
                diff = diff2
            else:
                handle_failure(pdb_id1, pdb_id2, 'num_mutated: %d (expected %d)' % (len(diff), len(mutated)))
                continue

        if SEQS[pdb_id1][mutated[0]] != seq1[diff[0]]:
            continue    #   TODO: decide what to do
        if SEQS[pdb_id2][mutated[0]] != seq2[diff[0]]:
            continue    #   TODO: decide what to do

        # if device.type == 'cpu':
        #     pmat1 = get_distance_matrix_cpu(X1)
        #     pmat2 = get_distance_matrix_cpu(X2)
        #     iseq1 = convert_to_indices_sequence_cpu(seq1)
        #     iseq2 = convert_to_indices_sequence_cpu(seq2)
        # else:
        pmat1 = get_distance_matrix(X1)
        pmat2 = get_distance_matrix(X2)
        iseq1 = convert_to_indices_sequence(seq1)
        iseq2 = convert_to_indices_sequence(seq2)

        assert iseq1.shape == iseq2.shape
        assert pmat1.shape == pmat2.shape
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq2.shape[0] == pmat2.shape[0]

        yield iseq1, iseq2, pmat1, pmat2, diff, pdb_id1, pdb_id2, seq1, seq2
        i_iter += 1

    pd.DataFrame(PDB_FPAIRS, columns=['pdb1', 'pdb2', 'reason']).to_csv(FPAIRS_PATH, index=False)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    dist = torch.clamp(dist, 0.0, np.inf)
    return torch.sqrt(dist)


def pairwise_distances_cpu(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = np.sum(x ** 2, axis=1).ravel()
    if y is not None:
        y_t = y.T
        y_norm = np.sum(y ** 2, axis=1).ravel()
    else:
        y_t = x.T
        y_norm = x_norm.ravel()

    dist = x_norm + y_norm - 2.0 * np.dot(x, y_t)

    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - np.diag(np.diag(dist))
    dist = np.clip(dist, 0.0, np.inf)
    return np.sqrt(dist)


if __name__ == "__main__":
    pbar = tqdm(total=10000, desc='pairs loaded')
    for s1, s2, m1, m2, idx in batch_generator(pairs_loader(PDB_PAIRS, 10000), prepare_torch_batch):
        assert m1.shape == m2.shape
        pbar.update(len(idx))
    pbar.close()
