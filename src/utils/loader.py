import pickle
from ast import literal_eval as make_tuple

from utils.data_utils import *


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

        if length < MIN_LENGTH:
            return None

        mutated = make_tuple(mutated)
        pdb1, chain_id1 = pdb_id1.split('_')
        pdb2, chain_id2 = pdb_id2.split('_')

        try:
            residues1, prof1, *_ = load_residues_and_profile_and_ccm(pdb1, chain_id1)
        except (FileNotFoundError, IOError) as e:
            return handle_failure(pdb_id1, str(e))

        try:
            residues2, prof2, *_ = load_residues_and_profile_and_ccm(pdb2, chain_id2)
        except (FileNotFoundError, IOError) as e:
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
        iseq1 = convert_to_onehot_sequence(seq1)
        iseq2 = convert_to_onehot_sequence(seq2)
        betas1 = to_betas(residues1)
        betas2 = to_betas(residues2)
        mutix = torch.tensor(diff, device=device)

        assert iseq1.shape == iseq2.shape
        assert pmat1.shape == pmat2.shape
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq2.shape[0] == pmat2.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        assert iseq2.shape[0] == prof2.shape[0]

        data = iseq1, iseq2, betas1, betas2, prof1, prof2, pmat1, pmat2, mutix
        meta = pdb_id1, pdb_id2, seq1, seq2
        return meta, data


class SimplePdbLoader(Loader):
    def __init__(self, list_of_items, n_iter):
        super(SimplePdbLoader, self).__init__(list_of_items, n_iter)

    def next(self):
        pdb_id1, length = self.list_of_items[self.i_pdb]
        if length < MIN_LENGTH:
            return None
        pdb1, chain_id1 = pdb_id1.split('_')

        try:
            residues1, prof1, *_ = load_residues_and_profile_and_ccm(pdb1, chain_id1)
        except (FileNotFoundError, IOError) as e:
            return handle_failure(pdb_id1, str(e))

        if len(residues1) < MIN_LENGTH:
            return None
        if len(residues1) < length // 2:
            return None
        if len(residues1) > MAX_LENGTH:
            start = np.random.randint(0, max(0, len(residues1) - MAX_LENGTH // 2))
            end = min(len(residues1), start + MAX_LENGTH)
            residues1 = residues1[start:end]
            prof1 = prof1[start:end]

        pmat1 = get_distance_matrix(toX(residues1, get_center=get_center2))

        seq1 = to_seq(residues1)
        iseq1 = convert_to_onehot_sequence(seq1)
        betas1 = to_betas(residues1)

        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        data = iseq1, betas1, prof1, pmat1
        meta = pdb_id1, seq1
        return meta, data


class PdbLoader(Loader):
    def __init__(self, list_of_items, n_iter):
        super(PdbLoader, self).__init__(list_of_items, n_iter)

    def next(self):
        pdb_id1, length = self.list_of_items[self.i_pdb]
        if length < MIN_LENGTH:
            return None
        pdb1, chain_id1 = pdb_id1.split('_')

        try:
            residues1, prof1, dssp1 = load_residues_profile_stride(pdb1, chain_id1)
        except (FileNotFoundError, IOError) as e:
            return handle_failure(pdb_id1, str(e))

        if len(residues1) < MIN_LENGTH:
            return None
        if len(residues1) < length // 2:
            return None
        if len(residues1) > MAX_LENGTH:
            start = np.random.randint(0, max(0, len(residues1) - MAX_LENGTH // 2))
            end = min(len(residues1), start + MAX_LENGTH)
            residues1 = residues1[start:end]
            dssp1 = dssp1[start:end]
            prof1 = prof1[start:end]

        pmat1 = get_distance_matrix(toX(residues1, get_center=get_center2))

        seq1 = to_seq(residues1)
        iseq1 = convert_to_onehot_sequence(seq1)
        betas1 = to_betas(residues1)

        assert iseq1.shape[0] == dssp1.shape[0]
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq1.shape[0] == prof1.shape[0]
        data = iseq1, betas1, prof1, pmat1, dssp1
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
            continue  # TODO: decide what to do
        if SEQs[pdb_id2][mutated[0]] != seq2[diff[0]]:
            continue  # TODO: decide what to do

        pmat1 = get_distance_matrix(X1)
        pmat2 = get_distance_matrix(X2)
        iseq1 = convert_to_onehot_sequence(seq1)
        iseq2 = convert_to_onehot_sequence(seq2)

        assert iseq1.shape == iseq2.shape
        assert pmat1.shape == pmat2.shape
        assert iseq1.shape[0] == pmat1.shape[0]
        assert iseq2.shape[0] == pmat2.shape[0]

        yield iseq1, iseq2, pmat1, pmat2, diff, pdb_id1, pdb_id2, seq1, seq2
        i_iter += 1


class XuLoader(object):

    def __init__(self, path_to_pickle):
        with open(path_to_pickle, 'rb') as f:
            self._d = d = pickle.load(f, encoding='bytes')
        self._p = sorted(range(len(self)), key=lambda i: len(d[i][b'sequence']))
        self.n_epoch = 0
        self.i_iter = 0

    def reset(self):
        self.i_iter = 0
        self.n_epoch += 1

    def shuffle(self):
        self.reset()
        np.random.seed(self.n_epoch)
        self._p = np.random.permutation(len(self))

    def __len__(self):
        return len(self._d)

    def __next__(self):
        ret = None
        while (ret is None) and (self.i_iter < len(self)):
            ret = self.next()
            self.i_iter += 1
        if ret is None:
            save_failures()
            raise StopIteration
        return ret

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return self._d[self._p[i]]

    def next(self):
        rec = self[self.i_iter]
        name = rec[b'name'].decode('utf8')
        sequence = rec[b'sequence'].decode('utf8')
        pdb, chain_id = name[:4], name[4:]
        residues = load_or_parse_residues(pdb, chain_id)
        try:
            assert residues is not None
        except AssertionError as e:
            return None
        cmat = rec[b'ccmpredZ']
        profile = rec[b'PSFM']
        seqres = to_seq(residues)
        m = align(seqres, sequence)
        seq = sequence[m.b:m.b+m.size]
        coords = toX(residues[m.a:m.a+m.size])
        cmat = cmat[m.b:m.b+m.size, m.b:m.b+m.size]
        prof = profile[m.b:m.b+m.size, :]
        if len(seq) < MIN_LENGTH:
            return None
        if len(coords) > MAX_LENGTH:
            start = np.random.randint(0, max(0, len(coords) // 2))
            end = min(len(coords), start + len(coords) // 2)
            coords = coords[start:end]
            cmat = cmat[start:end, start:end]
            prof = prof[start:end, :]
        dmat = get_distance_matrix(coords)
        try:
            assert cmat.shape == dmat.shape
            assert len(prof) == len(dmat) == len(seq)
            assert dmat.shape[0] == dmat.shape[1]
        except AssertionError as e:
            return None
        data = [seq, prof, cmat, dmat]
        meta = [name]
        return meta, data
