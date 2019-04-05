from utils import DATA_HOME

import os
import pickle
import itertools
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.constants import to_one_letter
from utils.sequence import FASTA
from utils.cdhit import get_cdhit_clusters
from utils.data_utils import load_or_parse_residues, SEQs, CULL_PDBs
from utils.ccmpred import get_ccmpred, CCM_HOME
from utils.profile import read_profile_from_msa
from utils.stride import get_stride_dataframe


def get_pairs(sequence_ids, sequences):
    pairs = []
    for seq1, seq2 in itertools.combinations(sequence_ids, 2):
        if len(sequences[seq1]) != len(sequences[seq2]):
            continue
        idx = np.where(~(sequences[seq1] == sequences[seq2]))[0]
        if len(idx) == 0:
            continue
        pairs.append((seq1, seq2, tuple(idx), len(idx), len(sequences[seq1])))
    return pairs


def to_fasta(sequences, out_file):
    lines = []
    for seq in sequences:
        lines.append(">%s\n%s\n" % (seq.id, seq.seq))
    with open(out_file, "w+") as f:
        f.writelines(lines)


def run_analysis(data_home=DATA_HOME):
    df = pd.read_csv('%s/pairs.csv' % data_home)
    order, sequences = FASTA('%s/pdb_seqres.txt' % data_home)
    uids = set(df.pdb1.tolist() + df.pdb2.tolist())
    pdbs = set([uid.split('_')[0].upper() for uid in uids])
    open('%s/pdb_structs_ids.txt' % data_home, 'w+').writelines(','.join(pdbs))
    to_fasta([sequences[uid] for uid in uids], '%s/pdb_unique_seqs.fas' % data_home)
    df[df.num_mutated > 0].num_mutated.hist(range=(0, 10))
    print("# unique chains: %d, # unique structs: %d" % (len(uids), len(pdbs)))
    plt.show()

    adj = {}
    pbar = tqdm(total=len(df), desc="rows processed")
    for i, row in df.iterrows():
        if row.pdb1 not in adj:
            adj[row.pdb1] = []
        adj[row.pdb1].append(row.pdb2)
        pbar.update(1)
    pbar.close()
    num_counterparts = [len(val) for val in adj.values()]
    plt.hist(num_counterparts, bins=50, range=(0, 100))
    print(np.mean(num_counterparts),
          np.median(num_counterparts),
          np.var(num_counterparts))
    plt.show()


def generate_pairs(data_home=DATA_HOME):
    pairs = set()
    cluster_dic, reverse_dic = get_cdhit_clusters('%s/pdb_seqres.txt' % data_home)
    order, sequences = FASTA('%s/pdb_seqres.txt' % data_home)
    sequences = {seqid: np.asarray(list(seq.seq)) for seqid, seq in sequences.items()
                 if seq.mol == 'protein' and 'XX' not in seq.seq}
    for clstr in tqdm(cluster_dic, desc="clusters processed"):
        pairs |= set(get_pairs([seqid for seqid in cluster_dic[clstr] if seqid in sequences], sequences))
    df = pd.DataFrame(list(pairs), columns=['pdb1', 'pdb2', 'mutated_residues', 'num_mutated', 'length'])
    df.to_csv("%s/pairs.csv" % data_home, index=False)


def download_cullpdb(data_home=DATA_HOME, cull_fname='cullpdb_pc25_res2.5_R1.0_d190212_chains12714', lim=5):
    uids = []
    path = os.path.join(data_home, cull_fname)
    df = pd.read_csv(path, sep=r"\s*", engine='python')
    for s in tqdm(df.IDs[:lim]):
        s = s.strip()
        pdb, chain = s[0:4].lower(), s[4]
        _ = load_or_parse_residues(pdb, chain)
        uids.append("%s_%s" % (pdb, chain))
    order, sequences = FASTA('%s/pdb_seqres.txt' % data_home)
    to_fasta([sequences[uid] for uid in uids if uid in sequences], '%s.fas' % path)
    to_fasta([sequences[uid] for uid in uids
              if uid in sequences and not os.path.exists('%s/msas/%s.a3m' % (data_home, uid))],
             os.path.join(data_home, 'delta.fas'))


def build_dataset(pdbs=CULL_PDBs):
    dataset = []
    for pdb_id, length in tqdm(pdbs, desc="chains processed"):
        pdb, chain_id = pdb_id.split('_')
        pth_to_msa = glob("%s/msas/%s.a3m" % (DATA_HOME, pdb_id))
        pth_to_mat = glob("%s/ccm/%s.mat" % (DATA_HOME, pdb_id))
        if len(pth_to_msa) != 1 or len(pth_to_mat) != 1:
            continue
        try:
            ccm_mat = get_ccmpred(pdb_id, to_tensor=False)
            stride = get_stride_dataframe(pdb, chain_id)
        except AssertionError:
            continue
        dataset.append({b'name': pdb_id,
                        b'length': length,
                        b'sequence': str(SEQs[pdb_id].seq),
                        b'ccmpredZ': ccm_mat,
                        b'PSFM': read_profile_from_msa(pdb_id),
                        b'stride_seq': list(map(to_one_letter, stride.AA)),
                        b'stride_ss7': list(stride.SS)})
    if len(dataset) == 0:
        return
    with open(os.path.join('data', 'xu', 'pdb25-train-valid-test-%d.pkl' % len(dataset)), 'w+b') as f:
        pickle.dump(dataset, f,  protocol=pickle.HIGHEST_PROTOCOL)


def run_ccmpred(pdbs=CULL_PDBs, ccm_home=CCM_HOME, lim=None):
    ccm_failures = []
    for pdb_id, _ in tqdm(pdbs[:lim], desc="chains procesed"):
        try:
            get_ccmpred(pdb_id, to_tensor=False)
        except (FileNotFoundError, AssertionError) as e:
            ccm_failures.append([pdb_id, str(e)])
    pd.DataFrame(ccm_failures, columns=["pdb_id", "error"]).to_csv("%s/ccm_failures.csv" % ccm_home, index=False)


def main(data_home=DATA_HOME):
    if not os.path.exists('%s/pairs.csv' % data_home):
        generate_pairs()
    else:
        run_analysis()


if __name__ == "__main__":
    # download_cullpdb(lim=None)
    build_dataset()
    # main()
