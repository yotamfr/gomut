import os
import random
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

random.seed(101)

REPO_PATH = os.path.join('data', 'pdbs_gz')
PAIRS_PATH = os.path.join('data', 'pairs.csv')
FAILED_PDBs_PATH = os.path.join('data', 'failed_pdbs.csv')
CULL_PDB_PATH = os.path.join('data', 'cullpdb_pc25_res2.5_R1.0_d190212_chains12714')

CULL_PDB_DF = pd.read_csv(CULL_PDB_PATH, sep=r"\s*", engine='python')
CULL_PDBs = [["%s_%s" % (s[0:4].lower(), s[4]), l] for s, l in CULL_PDB_DF[["IDs", "length"]].values]

random.shuffle(CULL_PDBs)
TRAIN_SET_CULL_PDB = sorted(CULL_PDBs[:int(0.8*len(CULL_PDBs))], key=lambda p: int(p[-1]))
VALID_SET_CULL_PDB = sorted(CULL_PDBs[int(0.9*len(CULL_PDBs)):], key=lambda p: int(p[-1]))
TEST_SET_CULL_PDB = sorted(CULL_PDBs[int(0.8*len(CULL_PDBs)):int(0.9*len(CULL_PDBs))], key=lambda p: int(p[-1]))

print('# TRAIN_SET_CULL_PDB: %d, # VALID_SET_CULL_PDB: %d, # TEST_SET_CULL_PDB: %d' %
      (len(TRAIN_SET_CULL_PDB), len(VALID_SET_CULL_PDB), len(TEST_SET_CULL_PDB)))

FAILED_PDBs = pd.read_csv(FAILED_PDBs_PATH)

PDB_PAIRS = pd.read_csv(PAIRS_PATH)
PDB_PAIRS = PDB_PAIRS[~PDB_PAIRS.pdb1.isin(FAILED_PDBs.pdb)]
PDB_PAIRS = PDB_PAIRS[~PDB_PAIRS.pdb2.isin(FAILED_PDBs.pdb)]
PDB_PAIRS = PDB_PAIRS[PDB_PAIRS.num_mutated == 1].values

FAILED_PDBs_LIST = FAILED_PDBs.values.tolist()
FAILED_PDBs_SET = set(FAILED_PDBs.pdb.values.tolist())

# PDB_PAIRS = PDB_PAIRS[~np.asarray([tuple(p) in set(FAILED_PDBs) for p in PDB_PAIRS[:, :2]])]
#
# random.shuffle(PDB_PAIRS)
# TRAIN_SET = sorted(PDB_PAIRS[:int(0.8*len(PDB_PAIRS))], key=lambda p: int(p[-1]))
# VALID_SET = sorted(PDB_PAIRS[int(0.9*len(PDB_PAIRS)):], key=lambda p: int(p[-1]))
# TEST_SET = sorted(PDB_PAIRS[int(0.8*len(PDB_PAIRS)):int(0.9*len(PDB_PAIRS))], key=lambda p: int(p[-1]))
#
# print('# TRAIN_SET: %d, # VALID_SET: %d, # TEST_SET: %d' % (len(TRAIN_SET), len(VALID_SET), len(TEST_SET)))
