import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.sequence import FASTA, Sequence
from utils.cdhit import get_cdhit_clusters
from utils.loader import load_or_parse_residues


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


def run_analysis():
    df = pd.read_csv('data/pairs.csv')
    order, sequences = FASTA('data/pdb_seqres.txt')
    uids = set(df.pdb1.tolist() + df.pdb2.tolist())
    pdbs = set([uid.split('_')[0].upper() for uid in uids])
    open('data/pdb_structs_ids.txt', 'w+').writelines(','.join(pdbs))
    to_fasta([sequences[uid] for uid in uids], 'data/pdb_unique_seqs.fas')
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


def generate_pairs():
    pairs = set()
    cluster_dic, reverse_dic = get_cdhit_clusters('data/pdb_seqres.txt')
    order, sequences = FASTA('data/pdb_seqres.txt')
    sequences = {seqid: np.asarray(list(seq.seq)) for seqid, seq in sequences.items()
                 if seq.mol == 'protein' and 'XX' not in seq.seq}
    for clstr in tqdm(cluster_dic, desc="clusters processed"):
        pairs |= set(get_pairs([seqid for seqid in cluster_dic[clstr] if seqid in sequences], sequences))
    df = pd.DataFrame(list(pairs), columns=['pdb1', 'pdb2', 'mutated_residues', 'num_mutated', 'length'])
    df.to_csv("data/pairs.csv", index=False)


def download_cullpdb(path='data/cullpdb_pc25_res2.5_R1.0_d190212_chains12714', lim=5):
    df = pd.read_csv(path, sep=r"\s*", engine='python')
    for s in tqdm(df.IDs[:lim]):
        pdb, chain = s[0:4].lower(), s[4]
        _ = load_or_parse_residues(pdb, chain)


def main():
    if not os.path.exists('data/pairs.csv'):
        generate_pairs()
    else:
        run_analysis()


if __name__ == "__main__":
    download_cullpdb(lim=None)
    # main()
