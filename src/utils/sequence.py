from collections import OrderedDict


class Sequence(object):

    def __init__(self, uid, mol, length, seq=''):
        self.id = uid
        self.seq = seq
        self.mol = mol
        self.length = length

    def __len__(self):
        return len(self.seq)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.seq == other.seq

    def __getitem__(self, i):
        return self.seq[i]


def FASTA(filename):
    order = []
    sequences = OrderedDict([])
    for line in open(filename, 'r'):
        if line.startswith('>'):
            try:
                uid, mol, length = line[1:].rstrip('\n').split()[:3]
                mol = mol.split(':')[1]
                length = int(length.split(':')[1])
                order.append(uid)
                sequences[uid] = Sequence(uid, mol, length, seq='')
            except (ValueError, IndexError):
                uid = line[1:].rstrip('\n').split()[0]
                sequences[uid] = Sequence(uid, None, None, seq='')
        else:
            sequences[uid].seq += line.rstrip('\n').rstrip('*')
    return order, sequences
