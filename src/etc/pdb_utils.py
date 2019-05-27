import numpy as np
from functools import lru_cache
from functools import reduce
from utils import to_one_letter

AA_dict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "B",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLX": "Z",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AAA_dict = {val: key for key, val in AA_dict.items()}

ATOMS = ['HD22', 'HD23', 'HD21', 'O3', 'HG11', 'CH2', 'HG13', 'HZ1', 'HZ3', 'HZ2', 'HA2', 'HA3',
         'H', 'P', "C2'", 'HG12', 'OH', 'OG', 'HB1', 'HB3', 'HB2', 'HZ', 'OP1', 'OP2', 'OP3',
         'CZ2', 'CZ3', "C5'", 'HH', 'HB', 'HE22', 'HE21', 'HA', 'HG', 'HE', 'NE2', 'C3', 'C',
         'OXT', 'O', "C4'", 'CE3', 'CE2', 'CE1', 'HD3', 'HD2', 'HD1', 'H2', 'H3', 'H1', 'HH12',
         'HH11', 'HG21', 'HG23', 'HG22', 'OP4', 'OE2', 'OE1', 'CD1', 'CD2', 'HE1', 'HE2', 'HE3',
         'NE', 'HH22', 'NZ', 'HH21', 'ND1', 'C2', 'ND2', 'C6', 'C5', 'C4', 'OD1', 'OD2', 'CG',
         'CE', 'N', 'CZ', 'CG1', 'N1', 'CG2', 'F1', 'F2', 'SG', 'SD', 'HH2', 'OG1', 'NE1', 'HG2',
         'HG3', 'HG1', 'CB', 'CA', 'NH1', 'NH2', 'HD13', 'HD12', 'HD11', 'CD']


ATOM_TYPES = list(set([a[0] for a in ATOMS]))
ATOM_POSITIONS = list(set([a[1:] if len(a) > 1 else '' for a in ATOMS]))


UNKNOWN = 'UNK'

UNHANDLED = {'HETSYN', 'COMPND', 'SOURCE', 'KEYWDS', 'EXPDTA', 'AUTHOR', 'REVDAT', 'JRNL', 'SIGATM',
             'REMARK', 'DBREF', 'SEQADV', 'HET', 'HETNAM', 'FORMUL', 'HELIX', 'CAVEAT',
             'SHEET', 'SITE', 'LINK', 'CISPEP', 'SITE', 'CRYST1', 'MTRIX1', 'MTRIX2', 'MTRIX3',
             'ORIGX1', 'ORIGX2', 'ORIGX3', 'SSBOND', 'MODRES', 'SPRSDE', 'DBREF1', 'OBSLTE',
             'SCALE1', 'SCALE2', 'SCALE3', 'SEQRES', 'MASTER', 'END', 'NUMMDL', 'MDLTYP',
             'SPLIT', 'HYDBND', 'SIGUIJ', 'DBREF2', 'SLTBRG'}  # TODO: Handle Multiple Models


amino_acids = list('ARNDCQEGHILKMFPSTWYV')
CNOS = list('CNOS')


class Atom(object):

    def __init__(self, name, num, residue, x, y, z, occup, temp, chain_id):
        assert name != ''
        self.num = num
        self.name = name
        self.temp = temp
        self.occup = occup
        self.res = residue
        self._coord = (x, y, z)
        self.orig_chain = chain_id

    @property
    def backbone(self):
        return self.name in ['CA', 'C', 'N', 'O']

    @property
    def res_name(self):
        return self.res.name

    @property
    def res_num(self):
        return self.res.num

    @property
    def chain_id(self):
        return self.res.chain_id

    @property
    def type(self):
        return self.name[0]

    @property
    def pos(self):
        return self.name[1:] if len(self.name) > 1 else ''

    @property
    def coord(self):
        return self._coord

    @property
    def x(self):
        return self._coord[0]

    @property
    def y(self):
        return self._coord[1]

    @property
    def z(self):
        return self._coord[2]

    def __eq__(self, other):
        return (self.name == other.name) and (self.res == other.res)

    def __str__(self):
        atom_num = self.num
        atom_name = self.name
        res_num = self.res_num
        aaa = AAA_dict[self.res.name]
        chain_id = self.chain_id
        occup = self.occup
        temp = self.temp
        (x, y, z) = self._coord
        return '{typ: <6}{0: >5}  {1: <4}{2: <4}{3: <1}{4: >4}    ' \
               '{5:8.3f}{6:8.3f}{7:8.3f}{8:6.2f}{9:6.2f}          ' \
               ' {10: <4}'\
            .format(atom_num, atom_name, aaa, chain_id, res_num, x, y, z, occup, temp, self.type, typ='ATOM')


class Residue(object):

    def __init__(self, three_letter_name, num, index, chain, atoms=[]):
        self.atoms = atoms
        self.chain = chain
        self.index = index
        self.num = num   # Not necessarily numeric
        self._name = three_letter_name

    @property
    def name(self):
        return to_one_letter(self._name)

    @property
    def center(self):
        if self._name == 'GLY':
            return np.asarray(self.get_atom_by_name("CA").coord)
        return np.mean([a.coord for a in self.atoms if not a.backbone], axis=0)

    @lru_cache(maxsize=None)
    def get_atom_by_name(self, name):
        x = [a for a in self.atoms if a.name == name]
        if len(x) == 0:
            return None
        return x[0]

    @property
    def ca(self):
        return self.get_atom_by_name("CA")

    @property
    def c(self):
        return self.get_atom_by_name("C")

    @property
    def n(self):
        return self.get_atom_by_name("N")

    @property
    def chain_id(self):
        return self.chain.chain_id

    def __eq__(self, other):
        return (other.chain == self.chain) and (other.index == self.index)

    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def __getitem__(self, i):
        return self.atoms[i]

    def __hash__(self):
        return hash((self.chain, self.num, self._name))

    def __str__(self):
        return "<Residue %s:%s>" % (self._name, self.num)


class Chain(object):

    def __init__(self, pdb, chain_id, residues=[]):
        self.index = {res.num: i for i, res in enumerate(residues)}
        self.residues = residues
        self.chain_id = chain_id
        self.pdb = pdb

    def __iter__(self):
        for res in self.residues:
            yield res

    def __getitem__(self, i):
        return self.residues[i]

    def __str__(self):
        return "<Chain %s>" % self.id

    @property
    def atoms(self):
        return reduce(lambda atoms, res: atoms + res.atoms, self.residues, [])

    @property
    def seq(self):
        return ''.join([res.name for res in self.residues])

    @property
    def id(self):
        return "%s_%s" % (self.pdb, self.chain_id)

    def __eq__(self, other):
        return other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def __len__(self):
        return len(self.residues)


class PDB(object):

    def __init__(self, modelname, chains=[], chain_dict={}):
        self._id_to_ix = {}
        self._id = modelname
        self._chains = []
        self._atoms = []

        for c in chains:
            if len(chain_dict) > 0:
                if c.chain_id not in chain_dict:
                    continue
                cid = chain_dict[c.chain_id]
                c.chain_id = cid
            else:
                cid = c.chain_id
            self._id_to_ix[cid] = len(self._chains)
            self._chains.append(c)
            for res in c.residues:
                self._atoms.extend(res.atoms)

        for a in self._atoms:
            if len(chain_dict) > 0:
                a.res.chain = self[chain_dict[a.orig_chain]]
            else:
                a.res.chain = self[a.orig_chain]

        assert not np.any([a.res.chain is None for a in self._atoms])

    @property
    def pdb(self):
        return self._id

    @property
    def chains(self):
        return {c.chain_id: c for c in self._chains}

    @property
    def residues(self):
        return [res for c in self._chains for res in c.residues]

    @property
    def atoms(self):
        return self._atoms

    def __iter__(self):
        for chain in self._chains:
            yield chain

    def __hash__(self):
        return hash(str(self))

    def __getitem__(self, chain_id):
        return self._chains[self._id_to_ix[chain_id]]

    def __str__(self):
        return "<%s: %s>" % (self.pdb, ','.join([c.chain_id for c in self._chains]))

    def to_pdb(self, path):
        lines = []
        for chain in self._chains:
            for atom in chain.atoms:
                lines.append("%s\n" % atom)
            lines.append("TER\n")
        with open(path, "w+") as f:
            f.writelines(lines)


def _handle_line(line, atoms, residues, chains, pdb, chain_id='A', residue_num=0):

    if not line:
        return atoms, residues, chains, chain_id, residue_num
    typ = line.strip().split(' ')[0]

    if typ in UNHANDLED:
        pass
    elif typ == 'MODEL':
        pass
    elif typ == 'HEADER':
        pass
    elif typ == 'TITLE':
        pass

    elif typ == 'ATOM' or typ == 'HETATM':
        # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        l_line = line[6:11], line[12:16], line[17:20], line[21], line[22:27], \
                 line[30:38], line[38:46], line[46:54], \
                 line[54:60], line[60:66], line[72:76], line[76:78]
        l_line = [s.strip() for s in l_line]
        atom_num, atom_name, res_name, chain_id, res_num, x, y, z, occup, temp, ident, sym = l_line
        chain_id = chain_id if chain_id else 'A'            # TODO: why modeller outputs '' chain_id?
        atom_num, res_num = int(atom_num), res_num

        try:
            occup, temp = float(occup), float(temp)
        except ValueError:
            occup, temp = 1.00, 00.00
        try:
            to_one_letter(res_name)
        except KeyError:
            return atoms, residues, chains, chain_id, res_num   # TODO : Handle Not Amino Acid!

        atom = Atom(atom_name, atom_num, None, float(x), float(y), float(z), occup, temp, chain_id)

        if len(residues) == 0 or residue_num != res_num:
            res = Residue(res_name, res_num, len(residues), None, [atom])
            residues.append(res)
        else:
            res = residues[-1]
            res.atoms.append(atom)

        atom.res = res
        residue_num = res_num
        atoms.append(atom)

    elif typ == 'TER':
        assert chain_id
        chain = Chain(pdb, chain_id, residues)
        chains.append(chain)
        residues = []

    elif 'HETATM' in typ:
        pass
    elif 'CONECT' in typ:
        pass
    elif 'ANISOU' in typ:
        pass

    else:
        raise ValueError("Unidentified type: '%s', PDB: %s" % (typ, pdb))

    return atoms, residues, chains, chain_id, residue_num


def parse_pdb(pdb, fd, chain_dict={}):
    line = fd.readline()
    atoms, residues, chains, chain_id, res_num = _handle_line(line, [], [], [], pdb)
    models = []
    while line:
        while line and line.split()[0] != 'ENDMDL':
            atoms, residues, chains, chain_id, res_num = _handle_line(line, atoms, residues, chains, pdb, chain_id, res_num)
            line = fd.readline()
        st = PDB(pdb, [c for c in chains if len(c) > 0], chain_dict)
        models.append(st)
        line = fd.readline()
    try: return models[0]
    except IndexError: print(pdb)


def parse_pdb2(pdb, path):
    return parse_pdb(pdb, open(path, "r"))


def to_fasta(structs, out_file):
    lines = []
    for st in structs:
        lines.extend([">%s\n%s\n" % (c.id, c.seq) for c in st.chains.values()])
    with open(out_file, "w+") as f:
        f.writelines(lines)
