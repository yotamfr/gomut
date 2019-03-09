from prody import *
from ast import literal_eval as make_tuple
from difflib import SequenceMatcher
from .profile import *
from .data import *
from .stride import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr('raise')
confProDy(verbosity='none')
random.seed(101)

PERM = 'r'
IDX = h5py.File('data/h5/idx.h5', PERM)
BETAS = h5py.File('data/h5/betas.h5', PERM)
COORDS = h5py.File('data/h5/coords.h5', PERM)
RESNAMES = h5py.File('data/h5/resnames.h5', PERM)
ATOMNAMES = h5py.File('data/h5/atomnames.h5', PERM)

_, SEQs = FASTA('data/pdb_seqres.txt')

MAX_ALLOWED_SHIFT = 10
MAX_BATCH_SIZE = 1
MAX_LENGTH = 525
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
    return pairwise_distances(torch.tensor(X, dtype=torch.float, device=device))


def convert_to_indices_sequence(seq):
    return torch.tensor([amino_acids.index(aa) for aa in seq], dtype=torch.long, device=device)


# def get_distance_matrix_cpu(X):
#     return pairwise_distances_cpu(np.asarray(X, dtype=np.float))


# def convert_to_indices_sequence_cpu(seq):
#     return np.asarray([amino_acids.index(aa) for aa in seq], dtype=np.long)


# def prepare_numpy_batch(data):
#     x1, x2, y1, y2, ix, *_ = zip(*data)
#     x1 = torch.tensor(np.stack(x1, 0), dtype=torch.float, device=device)
#     x2 = torch.tensor(np.stack(x2, 0), dtype=torch.float, device=device)
#     y1 = torch.tensor(np.stack(y1, 0), dtype=torch.float, device=device)
#     y2 = torch.tensor(np.stack(y2, 0), dtype=torch.float, device=device)
#     ix = torch.tensor(ix, dtype=torch.long, device=device)
#     return x1, x2, y1, y2, ix


def prepare_pdb_batch(data):
    iseq1, beta1, prof1, pmat1, dssp1, pdb1, *_ = zip(*data)
    iseq1 = torch.stack(iseq1, 0)
    beta1 = torch.stack(beta1, 0)
    prof1 = torch.stack(prof1, 0)
    pmat1 = torch.stack(pmat1, 0)
    dssp1 = torch.stack(dssp1, 0)
    return iseq1, beta1, prof1, pmat1, dssp1, pdb1


def prepare_pairs_batch(data):
    iseq1, iseq2, beta1, beta2, prof1, prof2, pmat1, pmat2, mutix, pdb1, pdb2, *_ = zip(*data)
    iseq1 = torch.stack(iseq1, 0)
    iseq2 = torch.stack(iseq2, 0)
    beta1 = torch.stack(beta1, 0)
    beta2 = torch.stack(beta2, 0)
    prof1 = torch.stack(prof1, 0)
    prof2 = torch.stack(prof2, 0)
    pmat1 = torch.stack(pmat1, 0)
    pmat2 = torch.stack(pmat2, 0)
    mutix = torch.stack(mutix, 0)
    return iseq1, iseq2, beta1, beta2, prof1, prof2, pmat1, pmat2, mutix, pdb1, pdb2


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


def load_or_parse_residues(pdb1, chain_id1, repo_path=REPO_PATH):
    if tostr(pdb1, chain_id1) in COORDS:
        residues = load_residues(pdb1, chain_id1)
        return residues
    src_path = os.path.join(repo_path, '%s.pdb.gz' % pdb1)
    if not os.path.exists(src_path):
        return None
    st1, h1 = parsePDB(src_path, header=True, chain=chain_id1)
    if not (h1['experiment'] == 'X-RAY DIFFRACTION'):
        return None
    if st1 is None:
        return None
    residues = store_residues(st1, pdb1, chain_id1)
    return residues


def find_maximal_matching_shift(seq1, seq2, k=1, max_shift=MAX_ALLOWED_SHIFT):
    size = min(len(seq1), len(seq2))
    min_size = size - max_shift
    a, b = 0, 0
    str1 = seq1[a: a + size]
    str2 = seq2[b: b + size]
    diff = np.where(str1 != str2)[0]
    while (size > min_size) and (len(diff) != k):
        for a in range(0, len(seq1) - size):
            for b in range(0, len(seq2) - size):
                str1 = seq1[a: a + size]
                str2 = seq2[b: b + size]
                diff = np.where(str1 != str2)[0]
        size -= 1
    return a, b, size, diff


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


# def pairwise_distances_cpu(x, y=None):
#     '''
#     Input: x is a Nxd matrix
#            y is an optional Mxd matirx
#     Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
#             if y is not given then use 'y=x'.
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#     '''
#     x_norm = np.sum(x ** 2, axis=1).ravel()
#     if y is not None:
#         y_t = y.T
#         y_norm = np.sum(y ** 2, axis=1).ravel()
#     else:
#         y_t = x.T
#         y_norm = x_norm.ravel()
#
#     dist = x_norm + y_norm - 2.0 * np.dot(x, y_t)
#
#     # Ensure diagonal is zero if x=y
#     if y is None:
#         dist = dist - np.diag(np.diag(dist))
#     dist = np.clip(dist, 0.0, np.inf)
#     return dist


def to_seq(residues):
    try:
        return ''.join([AA_dict[res.getResname()] for res in residues])
    except AttributeError as e:
        raise e


def align_sequence_and_profile(residues, pseq, prof):
    '''
    :param residues: seq (numpy array) as parsed from pdb atoms
    :param pseq: seq (string) as parsed from fasta file
    :param prof: profile (probability matrix)
    :return: prof1 & seq1 sharing the longest common subsequence
    '''
    str1, str2 = to_seq(residues), pseq
    match = SequenceMatcher(None, str1, str2)
    try:
        m = match.find_longest_match(0, len(str1), 0, len(str2))
    except TypeError as e:
        raise e
    if m.size != 0:
        residues = residues[m.a: m.a + m.size]
        prof = prof[m.b: m.b + m.size]
    return m, residues, prof


def is_okay(res):
    return (isinstance(get_center(res), np.ndarray)) and (res.getResname() in AA_dict)


def load_residues_and_profile(pdb1, chain_id1, sequence_dict=SEQs):
    pdb_id1 = tostr(pdb1, chain_id1)
    residues1 = load_or_parse_residues(pdb1, chain_id1)
    if residues1 is None:
        raise ValueError('pdb \'%s\' failed to parse' % pdb1)
    residues1 = list(filter(is_okay, residues1))
    try:
        prof1 = get_profile(pdb_id1)    # throws FileNotFoundError
    except FileNotFoundError:
        raise FileNotFoundError('could not find profile for: \'%s\'' % pdb_id1)
    match1, residues1, prof1 = align_sequence_and_profile(residues1, sequence_dict[pdb_id1].seq, prof1)
    if match1.size <= 1:
        raise ValueError('could not match seq to profile for: \'%s\'' % pdb_id1)
    assert len(residues1) == len(prof1)
    return residues1, prof1


def load_residues_profile_stride(pdb1, chain_id1):
    residues1, prof1 = load_residues_and_profile(pdb1, chain_id1)
    dssp1, seq1 = get_stride(pdb1, chain_id1)
    m1, _, _ = align_sequence_and_profile(residues1, seq1, dssp1)
    dssp1 = ([PAD_SS] * m1.a) + dssp1[m1.b:m1.b + m1.size] + ([PAD_SS] * (len(residues1) - m1.a - m1.size))
    assert len(dssp1) == len(residues1)
    assert len(prof1) == len(residues1)
    return residues1, prof1, torch.tensor(dssp1, dtype=torch.long, device=device)


class Loader(object):
    def __init__(self, list_of_items, n_iter):
        self.list_of_items = list_of_items
        self.n_iter = n_iter
        # self.i_pdb = np.random.randint(0, len(list_of_items))
        self.i_pdb = 0
        self.i_iter = 0
        self.N = len(list_of_items)

    def reset(self):
        self.i_iter = 0
        
    def next(self):
        raise NotImplementedError

    def __next__(self):
        if self.i_iter < self.n_iter:
            ret = None
            while ret is None:
                ret = self.next()
                self.i_pdb = (self.i_pdb + 1) % self.N
            self.i_iter += 1
            return ret
        else:
            save_failures()
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_iter


class PairsLoader(Loader):
    def __init__(self, list_of_items, n_iter):
        super(PairsLoader, self).__init__(list_of_items, n_iter)

    def next(self):
        pdb_id1, pdb_id2, mutated, _, length = self.list_of_items[self.i_pdb]

        if length > MAX_LENGTH:
            return None

        mutated = make_tuple(mutated)
        pdb1, chain_id1 = pdb_id1.split('_')
        pdb2, chain_id2 = pdb_id2.split('_')

        try:
            residues1, prof1 = load_residues_and_profile(pdb1, chain_id1)
        except (FileNotFoundError, ValueError) as e:
            print(tostr(pdb1, chain_id1))
            return handle_failure(pdb_id1, str(e))

        try:
            residues2, prof2 = load_residues_and_profile(pdb2, chain_id2)
        except (FileNotFoundError, ValueError) as e:
            print(tostr(pdb2, chain_id2))
            return handle_failure(pdb_id2, str(e))

        a, b, size, diff = find_maximal_matching_shift(to_seq(residues1), to_seq(residues2), k=len(mutated))
        if size < MIN_LENGTH:
            return None
        if len(diff) != len(mutated):
            return None
        residues1, prof1 = residues1[a: a + size], prof1[a: a + size, :]
        residues2, prof2 = residues2[b: b + size], prof2[b: b + size, :]

        pmat1 = get_distance_matrix(toX(residues1))
        pmat2 = get_distance_matrix(toX(residues2))
        seq1 = to_seq(residues1)
        seq2 = to_seq(residues2)
        iseq1 = convert_to_indices_sequence(seq1)
        iseq2 = convert_to_indices_sequence(seq2)
        betas1 = to_betas(residues1)
        betas2 = to_betas(residues2)
        mutix = torch.tensor(diff, device=device)

        assert iseq1.shape == iseq2.shape
        assert pmat1.shape == pmat2.shape
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq2.shape[0] == pmat2.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        assert iseq2.shape[0] == prof2.shape[0]

        data = iseq1, iseq2, betas1, betas2, prof1, prof2, pmat1.sqrt(), pmat2.sqrt(), mutix
        meta = pdb_id1, pdb_id2, seq1, seq2
        return meta, data


class SimplePdbLoader(Loader):
    def __init__(self, list_of_items, n_iter):
        super(SimplePdbLoader, self).__init__(list_of_items, n_iter)

    def next(self):
        pdb_id1, length = self.list_of_items[self.i_pdb]
        pdb1, chain_id1 = pdb_id1.split('_')
        try:
            residues1, prof1 = load_residues_and_profile(pdb1, chain_id1)
        except (FileNotFoundError, ValueError) as e:
            return handle_failure(pdb_id1, str(e))

        pmat1 = get_distance_matrix(toX(residues1, get_center=get_center2))

        seq1 = to_seq(residues1)
        iseq1 = convert_to_indices_sequence(seq1)
        betas1 = to_betas(residues1)

        if not (MIN_LENGTH <= len(residues1) <= MAX_LENGTH):
            return None
        if len(residues1) < length // 2:
            return None
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        data = iseq1, betas1, prof1, pmat1.sqrt()
        meta = pdb_id1, seq1
        return meta, data


class PdbLoader(Loader):
    def __init__(self, list_of_items, n_iter):
        super(PdbLoader, self).__init__(list_of_items, n_iter)

    def next(self):
        pdb_id1, length = self.list_of_items[self.i_pdb]
        pdb1, chain_id1 = pdb_id1.split('_')
        try:
            residues1, prof1, dssp1 = load_residues_profile_stride(pdb1, chain_id1)
        except (FileNotFoundError, ValueError) as e:
            return handle_failure(pdb_id1, str(e))

        pmat1 = get_distance_matrix(toX(residues1, get_center=get_center2))

        seq1 = to_seq(residues1)
        iseq1 = convert_to_indices_sequence(seq1)
        betas1 = to_betas(residues1)

        if not (MIN_LENGTH <= len(residues1) <= MAX_LENGTH):
            return None
        if len(residues1) < length // 2:
            return None
        assert iseq1.shape[0] == dssp1.shape[0]
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        data = iseq1, betas1, prof1, pmat1.sqrt(), dssp1
        meta = pdb_id1, seq1
        return meta, data


def pairs_loader(list_of_pairs, n_iter):
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
            continue
        residues2 = load_or_parse_residues(pdb2, chain_id2)
        if residues2 is None:
            continue

        X1, idx1 = zip(*[(x, i) for i, x in enumerate(toX(residues1)) if isinstance(x, np.ndarray)])
        X2, idx2 = zip(*[(x, i) for i, x in enumerate(toX(residues2)) if isinstance(x, np.ndarray)])

        if not (set(mutated).issubset(set(idx1)) and set(mutated).issubset(set(idx2))):
            continue

        try:
            seq1 = np.asarray([AA_dict[res.getResname()] for res in residues1])[np.asarray(idx1)]
            seq2 = np.asarray([AA_dict[res.getResname()] for res in residues2])[np.asarray(idx2)]
        except KeyError as e:
            continue

        a, b, size, diff = find_maximal_matching_shift(seq1, seq2, k=len(mutated))
        seq1, X1 = seq1[a: a + size], np.asarray(X1)[a: a + size, :]
        seq2, X2 = seq2[b: b + size], np.asarray(X2)[b: b + size, :]

        if len(diff) != len(mutated):
            continue

        if SEQs[pdb_id1][mutated[0]] != seq1[diff[0]]:
            continue    # TODO: decide what to do
        if SEQs[pdb_id2][mutated[0]] != seq2[diff[0]]:
            continue    # TODO: decide what to do

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
