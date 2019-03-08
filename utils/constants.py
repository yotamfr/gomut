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

amino_acids = list('ARNDCQEGHILKMFPSTWYV')


BACKBONE = ['CA', 'C', 'N', 'O']


"""
   H	    Alpha helix
   G	    3-10 helix
   I	    PI-helix
   E	    Extended conformation
   B or	b   Isolated bridge
   T	    Turn
   C	    Coil (none of the above)
"""

SECONDARY_STRUCTURE = ['B', 'H', 'G', 'I', 'E', 'H', 'C', 'T']
PAD_SS = -1
