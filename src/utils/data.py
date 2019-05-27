from src.utils import DATA_HOME

import os
import random
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

random.seed(101)

REPO_PATH = os.path.join(DATA_HOME, 'pdbs_gz')
PAIRS_PATH = os.path.join(DATA_HOME, 'etc', 'pairs.csv')
FAILED_PDBs_PATH = os.path.join(DATA_HOME, 'etc', 'failed_pdbs.csv')
CULL_PDB40_CDHIT_PATH = os.path.join(DATA_HOME, 'etc', 'cullpdb_pc40_chains13K')
CULL_PDB25_PATH = os.path.join(DATA_HOME, 'etc', 'cullpdb_pc25_res2.5_R1.0_d190212_chains12714')
CULL_PDB40_PATH = os.path.join(DATA_HOME, 'etc', 'cullpdb_pc40_res2.5_R1.0_d190516_chains20874')
CULL_PDB_PATH = CULL_PDB40_CDHIT_PATH

XU_TRAIN_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-train-6767.release.contactFeatures.pkl')
XU_VALID_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-valid-6767.release.contactFeatures.pkl')
XU_TEST_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-test-500.release.contactFeatures.pkl')

YOTAM_TRAIN_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-train-9697.pkl')
YOTAM_VALID_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-valid-1217.pkl')
YOTAM_TEST_SET = os.path.join(DATA_HOME, 'xu', 'pdb25-test-1217.pkl')

PAIRS_TRAIN_KEYS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-train-6149-keys.pkl')
PAIRS_TRAIN_VALS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-train-7125-vals.pkl')
PAIRS_VALID_KEYS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-valid-734-keys.pkl')
PAIRS_VALID_VALS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-valid-1117-vals.pkl')
PAIRS_TEST_KEYS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-test-776-keys.pkl')
PAIRS_TEST_VALS = os.path.join(DATA_HOME, 'xu', 'pdb25-pairs-test-1159-vals.pkl')

if os.path.exists(CULL_PDB_PATH):
    if CULL_PDB_PATH == CULL_PDB40_CDHIT_PATH:
        CULL_PDBs = pd.read_csv(CULL_PDB40_CDHIT_PATH).values.tolist()
    else:
        CULL_PDB_DF = pd.read_csv(CULL_PDB25_PATH, sep=r"\s*", engine='python')
        CULL_PDBs = [["%s_%s" % (s[0:4].lower(), s[4]), l] for s, l in CULL_PDB_DF[["IDs", "length"]].values]
else:
    CULL_PDBs = None

if CULL_PDBs is not None:
    np.random.shuffle(CULL_PDBs)
    TRAIN_SET_CULL_PDB = CULL_PDBs[:int(0.8*len(CULL_PDBs))]
    VALID_SET_CULL_PDB = CULL_PDBs[int(0.9*len(CULL_PDBs)):]
    TEST_SET_CULL_PDB = CULL_PDBs[int(0.8*len(CULL_PDBs)):int(0.9*len(CULL_PDBs))]

FAILED_PDBs = pd.read_csv(FAILED_PDBs_PATH)
FAILED_PDBs_LIST = FAILED_PDBs.values.tolist()
FAILED_PDBs_SET = set(FAILED_PDBs.pdb.values.tolist())

PDB_PAIRS = pd.read_csv(PAIRS_PATH)
PDB_PAIRS = PDB_PAIRS[~PDB_PAIRS.pdb1.isin(FAILED_PDBs.pdb)]
PDB_PAIRS = PDB_PAIRS[~PDB_PAIRS.pdb2.isin(FAILED_PDBs.pdb)]
PDB_PAIRS = PDB_PAIRS[PDB_PAIRS.num_mutated == 1].values
PDB_PAIRS = PDB_PAIRS[~np.asarray([tuple(p) in set(FAILED_PDBs) for p in PDB_PAIRS[:, :2]])]

TRAIN_SET_PAIRS = PDB_PAIRS[:int(0.8*len(PDB_PAIRS))]
VALID_SET_PAIRS = PDB_PAIRS[int(0.9*len(PDB_PAIRS)):]
TEST_SET_PAIRS = PDB_PAIRS[int(0.8*len(PDB_PAIRS)):int(0.9*len(PDB_PAIRS))]
