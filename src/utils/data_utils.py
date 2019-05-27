from itertools import combinations
from collections import defaultdict
from scipy.stats import pearsonr
from tqdm import tqdm

from prody import parsePDB, fetchPDB, confProDy
from collections import OrderedDict
from difflib import SequenceMatcher
from src.utils import PAD_SS, to_one_letter, BACKBONE
from src.utils.profile import *
from src.utils.data import *
from src.utils.stride import *

LOAD_CCM = False

if LOAD_CCM: from src.utils.ccmpred import *

warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr('raise')
confProDy(verbosity='none')
random.seed(101)

PERM = 'r'
IDX = h5py.File(osp.join(DATA_HOME, 'h5', 'idx.h5'), PERM)
BETAS = h5py.File(osp.join(DATA_HOME, 'h5', 'betas.h5'), PERM)
COORDS = h5py.File(osp.join(DATA_HOME, 'h5', 'coords.h5'), PERM)
RESNAMES = h5py.File(osp.join(DATA_HOME, 'h5', 'resnames.h5'), PERM)
ATOMNAMES = h5py.File(osp.join(DATA_HOME, 'h5', 'atomnames.h5'), PERM)

_, SEQs = FASTA(osp.join(DATA_HOME, 'etc', 'pdb_seqres.txt'))

MAX_ALLOWED_SHIFT = 10
MAX_BATCH_SIZE = 1
MAX_LENGTH = 512
MIN_LENGTH = 32


class Atom(object):

    def __init__(self, name, coords, beta):
        self.name = name.strip()
        self.coords = coords
        self.beta = beta

    def getName(self):
        return self.name

    def getCoords(self):
        return self.coords

    def getBeta(self):
        return self.beta


class Residue(object):

    def __init__(self, resname, atoms=[]):
        self.atoms = OrderedDict([(a.name, a) for a in atoms])
        self.resname = resname.strip()

    def getCoords(self):
        return [a.coords for a in self.atoms.values()]

    def select(self, selection_string='sc'):
        return Residue(self.resname, [a for a in self.atoms.values() if a.name not in BACKBONE])

    def getResname(self):
        return self.resname

    def getBetas(self):
        return [a.beta for a in self.atoms.values()]

    def __getitem__(self, item):
        return self.atoms.get(item, None)


def to_betas(residues):
    return torch.tensor([np.mean(res.getBetas()) for res in residues], dtype=torch.float32, device=device)


def filter_residues(residues):
    return [res for res in residues if res.getResname() != 'HOH']


def get_center(res):
    if res['CA']:
        return res['CA'].getCoords()
    else:
        return None


def get_center2(res):
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


def toX(residues, get_center=get_center):
    return [get_center(res) for res in residues]


def get_contact_map(pmat, thr=8.0):
    return pmat < thr


def get_distance_matrix(X):
    return pairwise_distances(torch.tensor(X, dtype=torch.float, device=device)).sqrt()


def aa2onehot(aa):
    oh = [0.] * len(amino_acids)
    try:
        oh[amino_acids.index(aa)] = 1.0
    except ValueError as e:
        if aa == 'B':
            oh[amino_acids.index('N')] = 0.5
            oh[amino_acids.index('D')] = 0.5
        elif aa == 'Z':
            oh[amino_acids.index('Q')] = 0.5
            oh[amino_acids.index('E')] = 0.5
        else:
            raise e
    return oh


def convert_to_onehot_sequence(seq):
    return torch.tensor([aa2onehot(aa) for aa in seq], dtype=torch.float, device=device)


def prepare_xu_batch(data):
    seq, prof, cmat, dmat, pdb, *_ = zip(*data)
    ohot = torch.stack([convert_to_onehot_sequence(s) for s in seq], 0)
    prof = torch.stack([torch.tensor(p, device=device, dtype=torch.float) for p in prof], 0)
    dmat = torch.stack([torch.tensor(d, device=device, dtype=torch.float) for d in dmat], 0)
    cmat = torch.stack([torch.tensor(c, device=device, dtype=torch.float) for c in cmat], 0)
    return ohot, prof, cmat, dmat, pdb


def prepare_pdb_batch(data):
    iseq1, beta1, prof1, pmat1, pdb1, *_ = zip(*data)
    iseq1 = torch.stack(iseq1, 0)
    beta1 = torch.stack(beta1, 0)
    prof1 = torch.stack(prof1, 0)
    pmat1 = torch.stack(pmat1, 0)
    return iseq1, beta1, prof1, pmat1, pdb1


def prepare_pdb_batch_w_dssp(data):
    iseq1, beta1, prof1, pmat1, dssp1, pdb1, *_ = zip(*data)
    iseq1 = torch.stack(iseq1, 0)
    beta1 = torch.stack(beta1, 0)
    prof1 = torch.stack(prof1, 0)
    pmat1 = torch.stack(pmat1, 0)
    dssp1 = torch.stack(dssp1, 0)
    return iseq1, beta1, prof1, pmat1, dssp1, pdb1


def prepare_pairs_batch(data):
    iseq1, iseq2, beta1, beta2, prof1, prof2, dmat1, dmat2, mut_ix, pdb1, pdb2, *_ = zip(*data)
    ohot1 = torch.stack([convert_to_onehot_sequence(s) for s in iseq1], 0)
    ohot2 = torch.stack([convert_to_onehot_sequence(s) for s in iseq2], 0)
    beta1 = torch.stack([torch.tensor(b, device=device, dtype=torch.float) for b in beta1], 0)
    beta2 = torch.stack([torch.tensor(b, device=device, dtype=torch.float) for b in beta2], 0)
    prof1 = torch.stack([torch.tensor(p, device=device, dtype=torch.float) for p in prof1], 0)
    prof2 = torch.stack([torch.tensor(p, device=device, dtype=torch.float) for p in prof2], 0)
    dmat1 = torch.stack([torch.tensor(d, device=device, dtype=torch.float) for d in dmat1], 0)
    dmat2 = torch.stack([torch.tensor(d, device=device, dtype=torch.float) for d in dmat2], 0)
    mut_ix = torch.stack(mut_ix, 0)
    return ohot1, ohot2, beta1, beta2, prof1, prof2, dmat1, dmat2, mut_ix, pdb1, pdb2


def batch_generator(loader, prepare, batch_size=MAX_BATCH_SIZE):
    batch = []
    curr_length = None
    for inp in loader:
        meta, data = inp
        x = data[0]
        if (len(x) != curr_length) or (len(batch) == batch_size):
            if batch:
                yield prepare(batch)
            batch = [data + meta]
        else:
            batch.append(data + meta)
        curr_length = len(x)
    if batch:
        yield prepare(batch)


def tostr(pdb1, chain_id1):
    return "%s_%s" % (pdb1, chain_id1)


def store_residues(st1, pdb1, chain_id1):
    residues = list(st1.select('protein')[chain_id1])
    data = [[i, j, r.getResname(), a.getName(), a.getCoords(), a.getBeta()]
            for i, r in enumerate(residues) for j, a in enumerate(r)]
    idx, _, resnames, atomnames, coords, betas = zip(*data)
    key = tostr(pdb1, chain_id1)
    if PERM != 'a': return residues
    if key not in IDX: IDX.create_dataset(key, data=np.array(idx).astype(np.long))
    if key not in BETAS: BETAS.create_dataset(key, data=np.array(betas).astype(np.float))
    if key not in COORDS: COORDS.create_dataset(key, data=np.array(coords).astype(np.float))
    if key not in RESNAMES: RESNAMES.create_dataset(key, data=np.array(resnames).astype('|S9'))
    if key not in ATOMNAMES: ATOMNAMES.create_dataset(key, data=np.array(atomnames).astype('|S9'))
    return residues


def load_residues(pdb1, chain_id1):
    idx = IDX[tostr(pdb1, chain_id1)][:]
    betas = BETAS[tostr(pdb1, chain_id1)][:]
    coords = COORDS[tostr(pdb1, chain_id1)][:]
    resnames = RESNAMES[tostr(pdb1, chain_id1)][:].astype('U13')
    atomnames = ATOMNAMES[tostr(pdb1, chain_id1)][:].astype('U13')
    residues = {i: Residue(name) for i, name in zip(idx, resnames)}
    for i, name, xyz, beta in zip(idx, atomnames, coords, betas):
        residues[i].atoms[name] = Atom(name, xyz, beta)
    return [residues[i] for i in set(idx)]


def load_or_parse_residues(pdb1, chain_id1, repo_path=REPO_PATH,
                           allowed_solving_methods=['SOLUTION NMR', 'X-RAY DIFFRACTION']):
    if tostr(pdb1, chain_id1) in COORDS:
        residues = load_residues(pdb1, chain_id1)
        return residues
    src_path = os.path.join(repo_path, '%s.pdb.gz' % pdb1)
    if not os.path.exists(src_path):
        fetchPDB(pdb1, folder=os.path.dirname(src_path))
    if not os.path.exists(src_path):
        return None
    st1, h1 = parsePDB(src_path, header=True, chain=chain_id1)
    if (st1 is None) or (h1 is None):
        return None
    if h1['experiment'] not in allowed_solving_methods:
        return None
    residues = store_residues(st1, pdb1, chain_id1)
    return residues


def load_or_parse_residue_pair(pdb_id1, pdb_id2):
    try:
        pdb1, chain_id1 = pdb_id1[:4], pdb_id1[4:]
        residues1 = load_or_parse_residues(pdb1, chain_id1)
        pdb2, chain_id2 = pdb_id2[:4], pdb_id2[4:]
        residues2 = load_or_parse_residues(pdb2, chain_id2)
        assert (residues1 is not None) and (residues2 is not None)
    except (AssertionError, OSError) as e:
        return None
    return residues1, residues2


def find_maximal_matching_shift(seq1, seq2, k=1, max_shift=MAX_ALLOWED_SHIFT):
    def binary_array(sq1, sq2):
        return [int(c1 != c2) for c1, c2 in zip(sq1, sq2)]

    def indices(bin_arr):
        return [i for i, c in enumerate(bin_arr) if c == 1]

    size = min(len(seq1), len(seq2))
    min_size = size - max_shift
    a, b = 0, 0
    str1 = seq1[a: a + size]
    str2 = seq2[b: b + size]
    bin_array = binary_array(str1, str2)
    while (size > min_size) and (sum(bin_array) != k):
        for a in range(0, len(seq1) - size):
            str1 = seq1[a: a + size]
            for b in range(0, len(seq2) - size):
                str2 = seq2[b: b + size]
                bin_array = binary_array(str1, str2)
                if sum(bin_array) == k:
                    return a, b, size, indices(bin_array)
        size -= 1
    return a, b, size, indices(bin_array)


def handle_failure(pdb1, reason, failed_list=FAILED_PDBs_LIST, failed_pdbs=FAILED_PDBs_SET):
    if pdb1 in failed_pdbs:
        return
    failed_list.append([pdb1, reason])
    failed_pdbs.add(pdb1)


def save_failures(failed_path=FAILED_PDBs_PATH, failed_list=FAILED_PDBs_LIST):
    pd.DataFrame(failed_list, columns=['pdb', 'reason']).to_csv(failed_path, index=False)


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
    return dist


def to_seq(residues):
    return ''.join([to_one_letter(res.getResname()) for res in residues])


def align(str1, str2):
    match = SequenceMatcher(None, str1, str2)
    m = match.find_longest_match(0, len(str1), 0, len(str2))
    return m


def align_helper(sequence, profile, residues, cmat=None):
    seqres = to_seq(residues)
    m = align(seqres, sequence)
    L = m.size
    seq = sequence[m.b:m.b + L]
    residues = residues[m.a:m.a + L]
    if cmat is not None:
        cmat = cmat[m.b:m.b + L, m.b:m.b + L]
    prof = profile[m.b:m.b + L, :]
    return seq, residues, cmat, prof


def is_okay(res):
    return (isinstance(get_center(res), np.ndarray)) and (res.getResname() in AA_dict)


def load_residues_and_profile_and_ccm(pdb1, chain_id1, sequence_dict=SEQs, load_ccm=LOAD_CCM):
    pdb_id1 = tostr(pdb1, chain_id1)
    residues1 = load_or_parse_residues(pdb1, chain_id1)
    if residues1 is None:
        raise IOError('pdb \'%s\' failed to parse' % pdb1)
    residues1 = list(filter(is_okay, residues1))
    try:
        prof1 = get_profile(pdb_id1)    # throws FileNotFoundError
    except FileNotFoundError:
        raise FileNotFoundError('could not find profile for: \'%s\'' % pdb_id1)
    m = align(to_seq(residues1), sequence_dict[pdb_id1].seq)
    residues1, prof1 = residues1[m.a: m.a + m.size], prof1[m.b: m.b + m.size, :]
    assert len(residues1) == len(prof1)
    if m.size <= 1:
        raise IOError('could not match seq to profile for: \'%s\'' % pdb_id1)
    if load_ccm:
        try:
            ccm1 = get_ccmpred(tostr(pdb1, chain_id1), compute=True, to_tensor=True)
        except AssertionError:
            raise MemoryError('Assert failed- not enough memory on GPU!')
        ccm1 = ccm1[m.b:m.b + m.size, m.b:m.b + m.size]
        return residues1, prof1, ccm1
    return residues1, prof1


def load_residues_profile_stride(pdb1, chain_id1):
    residues1, prof1, *_ = load_residues_and_profile_and_ccm(pdb1, chain_id1, load_ccm=LOAD_CCM)
    try:
        dssp1, seq1 = get_stride(pdb1, chain_id1)
        m1 = align(to_seq(residues1), seq1)
        dssp1 = ([PAD_SS] * m1.a) + dssp1[m1.b:m1.b + m1.size] + ([PAD_SS] * (len(residues1) - m1.a - m1.size))
    except ValueError:
        dssp1 = [PAD_SS] * len(residues1)
    assert len(dssp1) == len(residues1)
    assert len(prof1) == len(residues1)
    return residues1, prof1, torch.tensor(dssp1, dtype=torch.long, device=device)


def filter_pair_dataset(dataset_pairs, threshold):

    def subsample(arr, n=2):
        return arr[0:len(arr):n]

    def compute_correlation(d1, d2):
        a1, a2 = subsample(d1.ravel()), subsample(d2.ravel())
        return pearsonr(a1, a2)

    pair_id_to_list_of_pairs = defaultdict(list)
    for rec in tqdm(dataset_pairs, desc="pairs processed"):
        pair_id_to_list_of_pairs[rec[b'pair_id']].append(rec)

    dataset_filtered = []
    for pair_id, records in tqdm(pair_id_to_list_of_pairs.items(), desc="unique seq-pairs filtered"):
        deltas = [rec[b'dmat1'] - rec[b'dmat2'] for rec in records]
        if len(deltas) == 1:
            dataset_filtered.extend(records)
            continue
        cors = [compute_correlation(p1, p2) for (p1, p2) in combinations(deltas, 2)]
        avg_pearsonr = np.mean(cors)
        if avg_pearsonr < threshold:
            continue
        dataset_filtered.extend(records)

    return dataset_filtered
