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


def to_one_letter(three_letter_name):
    try:
        return AA_dict[three_letter_name]
    except KeyError:
        if three_letter_name == 'PHD':
            return 'D'
        if three_letter_name == 'TPO':
            return 'T'
        if three_letter_name == 'SEP':
            return 'S'
        if three_letter_name == 'CSO':
            return 'C'
        if three_letter_name == 'PYL':
            return 'K'
        if three_letter_name == 'SEC':
            return 'C'
        if three_letter_name == 'F2F':
            return 'F'
        if three_letter_name == 'HIC':
            return 'H'
        if three_letter_name == 'CGU':
            return 'E'
        if three_letter_name == 'PTR':
            return 'Y'
        if three_letter_name == 'MSE':
            return 'M'
        if three_letter_name == '4HT':
            return 'W'
        if three_letter_name == 'DHI':
            return 'H'
        if three_letter_name == 'LLP':
            return 'K'
        if three_letter_name == 'NAG':
            return 'X'
        if three_letter_name == 'UNK':
            return 'X'
        raise KeyError("Unidentified res: \'%s\'" % three_letter_name)


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

SECONDARY_STRUCTURE = ['H', 'G', 'I', 'E', 'B', 'T', 'C']
PAD_SS = -1
