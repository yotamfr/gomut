import os
import random
import numpy as np
import pandas as pd

random.seed(101)

REPO_PATH = os.path.join('data', 'pdbs_gz')
PAIRS_PATH = os.path.join('data', 'pairs.csv')
FPAIRS_PATH = os.path.join('data', 'failed_pairs.csv')

PDB_FPAIRS = pd.read_csv(FPAIRS_PATH).values.tolist()
PDB_FPAIRS_SET = set([(p[0], p[1]) for p in PDB_FPAIRS])

PDB_PAIRS = pd.read_csv(PAIRS_PATH)
PDB_PAIRS = PDB_PAIRS[PDB_PAIRS.num_mutated == 1].values
PDB_PAIRS = PDB_PAIRS[~np.asarray([tuple(p) in PDB_FPAIRS_SET for p in PDB_PAIRS[:, :2]])]

random.shuffle(PDB_PAIRS)
TRAIN_SET = sorted(PDB_PAIRS[:int(0.8*len(PDB_PAIRS))], key=lambda p: int(p[-1]))
VALID_SET = sorted(PDB_PAIRS[int(0.9*len(PDB_PAIRS)):], key=lambda p: int(p[-1]))
TEST_SET = sorted(PDB_PAIRS[int(0.8*len(PDB_PAIRS)):int(0.9*len(PDB_PAIRS))], key=lambda p: int(p[-1]))

print('# TRAIN_SET: %d, # VALID_SET: %d, # TEST_SET: %d' % (len(TRAIN_SET), len(VALID_SET), len(TEST_SET)))
