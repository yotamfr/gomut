from prody import *
from ast import literal_eval as make_tuple
from difflib import SequenceMatcher
from .profile import *
from .data import *

np.seterr('raise')
confProDy(verbosity='none')
random.seed(101)

PERM = 'r'
IDX = h5py.File('data/h5/idx.h5', PERM)
COORDS = h5py.File('data/h5/coords.h5', PERM)
RESNAMES = h5py.File('data/h5/resnames.h5', PERM)
ATOMNAMES = h5py.File('data/h5/atomnames.h5', PERM)

_, SEQS = FASTA('data/pdb_seqres.txt')

MAX_BATCH_SIZE = 4
MAX_LENGTH = 500


class Atom(object):

    def __init__(self, name, coords):
        self.coords = coords
        self.name = name.strip()

    def getName(self):
        return self.name

    def getCoords(self):
        return self.coords


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


# def get_contact_map_cpu(X, thr=8.0):
#     pmat = pairwise_distances_cpu(np.asarray(X, dtype=np.float))
#     return ((pmat < thr) & (pmat > 0.0)).astype(np.float)
#
#
# def get_distance_matrix_cpu(X):
#     return pairwise_distances_cpu(np.asarray(X, dtype=np.float))
#
#
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


def prepare_torch_batch(data):
    iseq1, iseq2, prof1, prof2, pmat1, pmat2, mutix, pdb1, pdb2, *_ = zip(*data)
    iseq1 = torch.stack(iseq1, 0)
    iseq2 = torch.stack(iseq2, 0)
    prof1 = torch.stack(prof1, 0)
    prof2 = torch.stack(prof2, 0)
    pmat1 = torch.stack(pmat1, 0)
    pmat2 = torch.stack(pmat2, 0)
    mutix = torch.stack(mutix, 0)
    return iseq1, iseq2, prof1, prof2, pmat1, pmat2, mutix, pdb1, pdb2


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
    if PERM == 'a':
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
    if tostr(pdb1, chain_id1) in COORDS:
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


def find_maximal_matching_shift(seq1, seq2, k=1):
    size = min(len(seq1), len(seq2))
    min_size = size - 10
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


def handle_failure(pdb1, pdb2, reason):
    if (pdb1, pdb2) not in PDB_FPAIRS_SET:
        PDB_FPAIRS.append([pdb1, pdb2, reason])
        PDB_FPAIRS_SET.add((pdb1, pdb2))


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
            handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb1)
            continue
        residues2 = load_or_parse_residues(pdb2, chain_id2)
        if residues2 is None:
            handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb2)
            continue

        X1, idx1 = zip(*[(x, i) for i, x in enumerate(toX(residues1)) if isinstance(x, np.ndarray)])
        X2, idx2 = zip(*[(x, i) for i, x in enumerate(toX(residues2)) if isinstance(x, np.ndarray)])

        if not (set(mutated).issubset(set(idx1)) and set(mutated).issubset(set(idx2))):
            handle_failure(pdb_id1, pdb_id2, 'residue(s): %s are missing' % mutated)
            continue

        try:
            seq1 = np.asarray([AA_dict[res.getResname()] for res in residues1])[np.asarray(idx1)]
            seq2 = np.asarray([AA_dict[res.getResname()] for res in residues2])[np.asarray(idx2)]
        except KeyError as e:
            handle_failure(pdb_id1, pdb_id2, 'KeyError: %s' % str(e))
            continue

        a, b, size, diff = find_maximal_matching_shift(seq1, seq2, k=len(mutated))
        seq1, X1 = seq1[a: a + size], np.asarray(X1)[a: a + size, :]
        seq2, X2 = seq2[b: b + size], np.asarray(X2)[b: b + size, :]

        if len(diff) != len(mutated):
            handle_failure(pdb_id1, pdb_id2, 'num_mutated: %d (expected %d)' % (len(diff), len(mutated)))
            continue

        if SEQS[pdb_id1][mutated[0]] != seq1[diff[0]]:
            continue    # TODO: decide what to do
        if SEQS[pdb_id2][mutated[0]] != seq2[diff[0]]:
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


def align_sequence_and_profile(seq1, pseq1, prof1):
    '''
    :param seq1: seq (numpy array) as parsed from pdb atoms
    :param pseq1: seq (string) as parsed from fasta file
    :param prof1: profile (probability matrix)
    :return: prof1 & seq1 sharing the longest common subsequence
    '''
    str2 = pseq1.seq
    str1 = ''.join(seq1)
    match = SequenceMatcher(None, str1, str2)
    m = match.find_longest_match(0, len(str1), 0, len(str2))
    if m.size != 0:
        seq1 = seq1[m.a: m.a + m.size]
        prof1 = prof1[m.b: m.b + m.size, :]
    return m, seq1, prof1


class Loader(object):
    def __init__(self, list_of_pairs, n_iter):
        self.list_of_pairs = list_of_pairs
        self.n_iter = n_iter
        # self.i_pdb = np.random.randint(0, len(list_of_pairs))
        self.i_pdb = 0
        self.i_iter = 0
        self.N = len(list_of_pairs)

    def reset(self):
        self.i_iter = 0

    def next(self):
        pdb_id1, pdb_id2, mutated, _, length = self.list_of_pairs[self.i_pdb]
        if length > MAX_LENGTH:
            return None

        mutated = make_tuple(mutated)
        pdb1, chain_id1 = pdb_id1.split('_')
        pdb2, chain_id2 = pdb_id2.split('_')
        self.i_pdb = (self.i_pdb + 1) % self.N

        residues1 = load_or_parse_residues(pdb1, chain_id1)
        if residues1 is None:
            return handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb1)
        residues2 = load_or_parse_residues(pdb2, chain_id2)
        if residues2 is None:
            return handle_failure(pdb_id1, pdb_id2, 'pdb \'%s\' failed to parse' % pdb2)

        X1, idx1 = zip(*[(x, i) for i, x in enumerate(toX(residues1)) if isinstance(x, np.ndarray)])
        X2, idx2 = zip(*[(x, i) for i, x in enumerate(toX(residues2)) if isinstance(x, np.ndarray)])

        if not (set(mutated).issubset(set(idx1)) and set(mutated).issubset(set(idx2))):
            return handle_failure(pdb_id1, pdb_id2, 'residue(s): %s are missing' % mutated)

        try:
            seq1 = np.asarray([AA_dict[res.getResname()] for res in residues1])[np.asarray(idx1)]
            seq2 = np.asarray([AA_dict[res.getResname()] for res in residues2])[np.asarray(idx2)]
        except KeyError as e:
            return handle_failure(pdb_id1, pdb_id2, 'KeyError: %s' % str(e))

        try:
            prof1 = get_profile(pdb_id1)
        except FileNotFoundError:
            return handle_failure(pdb_id1, pdb_id2, 'could not find profile for: \'%s\'' % pdb_id1)
        try:
            prof2 = get_profile(pdb_id2)
        except FileNotFoundError:
            return handle_failure(pdb_id1, pdb_id2, 'could not find profile for: \'%s\'' % pdb_id2)

        pseq1, pseq2 = SEQS[pdb_id1], SEQS[pdb_id2]

        match1, seq1, prof1 = align_sequence_and_profile(seq1, pseq1, prof1)
        if match1.size <= 1:
            return handle_failure(pdb_id1, pdb_id2, 'could not match seq to profile for: \'%s\'' % pdb_id1)
        assert len(seq1) == len(prof1)

        match2, seq2, prof2 = align_sequence_and_profile(seq2, pseq2, prof2)
        if match2.size <= 1:
            return handle_failure(pdb_id1, pdb_id2, 'could not match seq to profile for: \'%s\'' % pdb_id2)
        assert len(seq2) == len(prof2)

        a, b, size, diff = find_maximal_matching_shift(seq1, seq2, k=len(mutated))
        if size == 0: return    # TODO: handle error...
        seq1, X1, prof1 = seq1[a: a + size], np.asarray(X1)[a: a + size, :], prof1[a: a + size, :]
        seq2, X2, prof2 = seq2[b: b + size], np.asarray(X2)[b: b + size, :], prof2[b: b + size, :]
        if len(diff) != len(mutated):
            return handle_failure(pdb_id1, pdb_id2, 'num_mutated: %d (expected %d)' % (len(diff), len(mutated)))

        pmat1 = get_distance_matrix(X1)
        pmat2 = get_distance_matrix(X2)
        iseq1 = convert_to_indices_sequence(seq1)
        iseq2 = convert_to_indices_sequence(seq2)
        mutix = torch.tensor(diff, device=device)

        assert iseq1.shape == iseq2.shape
        assert pmat1.shape == pmat2.shape
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq2.shape[0] == pmat2.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        assert iseq2.shape[0] == prof2.shape[0]

        return iseq1, iseq2, prof1, prof2, pmat1.sqrt(), pmat2.sqrt(), mutix, pdb_id1, pdb_id2, seq1, seq2

    def __next__(self):
        if self.i_iter < self.n_iter:
            ret = self.next()
            while ret is None:
                ret = self.next()
            self.i_iter += 1
            return ret
        else:
            pd.DataFrame(PDB_FPAIRS, columns=['pdb1', 'pdb2', 'reason']).to_csv(FPAIRS_PATH, index=False)
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_iter
