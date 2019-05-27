from src.utils import DATA_HOME

import os
import io
import locale
import gzip
import shutil
import os.path as osp
import subprocess
import pandas as pd
from prody import fetchPDB

from src.utils import SECONDARY_STRUCTURE, AA_dict

STRIDE_HOME = '/home/yotamfr/development/gomut/stride'

STRIDE_PATH = osp.join(DATA_HOME, 'stride')

PDB_PATH = osp.join(DATA_HOME, 'pdbs_gz')


def extract_pdb_gz(pdb, stride_repo=STRIDE_PATH, pdb_repo=PDB_PATH):
    src_path = os.path.join(pdb_repo, '%s.pdb.gz' % pdb)
    if not os.path.exists(src_path):
        fetchPDB(pdb, folder=pdb_repo)
        assert os.path.exists(src_path)
    with gzip.open(osp.join(pdb_repo, '%s.pdb.gz' % (pdb,)), 'rb') as f_in:
        with open(osp.join(stride_repo, '%s.pdb' % (pdb,)), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def run_stride(pdb, chain_id, stride_home=STRIDE_HOME, stride_repo=STRIDE_PATH):
    extract_pdb_gz(pdb)
    path_to_pdb = osp.join(stride_repo, '%s.pdb' % (pdb,))
    args = ["%s/stride" % stride_home, path_to_pdb, '-r%s' % (chain_id,)]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = io.TextIOWrapper(proc.stdout, encoding=locale.getpreferredencoding(False), errors='strict')
    return parse_stride(out)


def get_stride_dataframe(pdb, chain_id, stride_repo=STRIDE_PATH):
    stride_path = osp.join(stride_repo, "%s_%s.csv" % (pdb, chain_id))
    if osp.exists(stride_path):
        df = pd.read_csv(stride_path)
    else:
        df = run_stride(pdb, chain_id)
        df.to_csv(stride_path, index=False)
    return df


def get_stride(pdb, chain_id, ss_arr=SECONDARY_STRUCTURE):
    df = get_stride_dataframe(pdb, chain_id)
    ss, seq = zip(*[(ss_arr.index(s.upper()), AA_dict[a]) for s, a in zip(df.SS, df.AA) if a in AA_dict])
    return list(ss), ''.join(seq)


def parse_stride(out):

    info = list()
    line = out.readline()

    while line:
        typ = line.split()[0]
        if typ == 'ASG':
            _, aa, chain, res, _, ss, _, phi, psi, asa, _ = line.split()
            info.append([aa, chain, res, ss, phi, psi, asa])
        line = out.readline()

    assert len(info) > 0
    aas, chains, ress, sss, phis, psis, asas = zip(*info)

    return pd.DataFrame({"AA": aas, "Chain": chains, "Res": ress, "SS": sss, "Phi": phis, "Psi": psis, "ASA": asas})
