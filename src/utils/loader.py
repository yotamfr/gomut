from src.utils.data_utils import *


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


class DatasetLoader(object):
    def __init__(self, dataset):
        self._d = d = dataset
        self._p = sorted(range(len(self)), key=lambda i: d[i][b'length'])
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

    def next(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return self._d[self._p[i]]


class XuLoader(DatasetLoader):

    def __init__(self, dataset):
        super(XuLoader, self).__init__(dataset)

    def next(self):
        rec = self[self.i_iter]
        name = rec[b'name'].decode('utf8')
        sequence = rec[b'seqres'].decode('utf8')
        prof = torch.tensor(rec[b'PSFM'], dtype=torch.float, device=device)
        dmat = torch.tensor(rec[b'dist_matrix'], dtype=torch.float, device=device)
        data = [prof, dmat]
        meta = [name, sequence]
        return meta, data


class PairsLoader(DatasetLoader):
    def __init__(self, dataset):
        super(PairsLoader, self).__init__(dataset)

    def next(self):
        rec = self[self.i_iter]

        dmat1 = torch.tensor(rec[b'dmat1'], dtype=torch.float, device=device)
        dmat2 = torch.tensor(rec[b'dmat2'], dtype=torch.float, device=device)
        seq1, seq2 = rec[b'seq1'], rec[b'seq2']
        prof1 = torch.tensor(rec[b'prof1'], dtype=torch.float, device=device)
        prof2 = torch.tensor(rec[b'prof2'], dtype=torch.float, device=device)
        pdb_id1, pdb_id2 = rec[b'pdb1'], rec[b'pdb2']
        mutix = torch.tensor(rec[b'mutix'], device=device)

        assert len(seq1) == len(seq2)
        assert dmat1.shape == dmat2.shape
        assert len(seq1) == dmat1.shape[0]
        assert len(seq2) == dmat2.shape[0]
        assert len(seq1) == prof1.shape[0]
        assert len(seq2) == prof2.shape[0]

        data = seq1, seq2,  prof1, prof2, dmat1, dmat2, mutix
        meta = pdb_id1, pdb_id2, seq1, seq2
        return meta, data
