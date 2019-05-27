from src.utils import DATA_HOME

import os
import pickle
import itertools
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from src.utils import to_one_letter
from src.utils.stride import get_stride_dataframe
from src.utils.sequence import FASTA, to_fasta
from src.utils.data import CULL_PDB40_CDHIT_PATH
from src.utils.cdhit import get_cdhit_clusters
from src.utils.data_utils import *
from src.utils.ccmpred import get_ccmpred, CCM_HOME
from src.utils.profile import read_profile_from_msa


def save_dataset(dataset, name_prefix, data_home=DATA_HOME):
    with open(os.path.join(data_home, 'ds', '%s-%d.pkl' % (name_prefix, len(dataset))), 'w+b') as f:
        pickle.dump(dataset, f,  protocol=pickle.HIGHEST_PROTOCOL)


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


def generate_pairs(fasta_filename, data_home=DATA_HOME):
    pairs = set()
    cluster_dic, reverse_dic = get_cdhit_clusters(fasta_filename, 0.99)
    order, sequences = FASTA('%s/etc/pdb_seqres.txt' % data_home)
    sequences = {seqid: np.asarray(list(seq.seq)) for seqid, seq in sequences.items()
                 if seq.mol == 'protein' and 'XX' not in seq.seq}
    for clstr in tqdm(cluster_dic, desc="clusters processed"):
        pairs |= set(get_pairs([seqid for seqid in cluster_dic[clstr] if seqid in sequences], sequences))
    df = pd.DataFrame(list(pairs), columns=['pdb1', 'pdb2', 'mutated_residues', 'num_mutated', 'length'])
    df.to_csv("%s/pairs.csv" % data_home, index=False)


def get_pdb_seqres_with_msa_fasta_filename(data_home=DATA_HOME):
    order, sequences = FASTA('%s/etc/pdb_seqres.txt' % data_home)
    fasta_filename = '%s/etc/pdb_seqres_with_msa.fas' % data_home
    if os.path.exists(fasta_filename):
        return fasta_filename
    to_fasta([seq for seq in sequences.values() if has_msa(seq.id)], fasta_filename)
    return fasta_filename


def generate_pairs_with_cdhit():
    generate_pairs(get_pdb_seqres_with_msa_fasta_filename())


def generate_cullpdb(data_home=DATA_HOME, cullpdb_path=CULL_PDB40_CDHIT_PATH, similaritity_threshold=0.4):
    _, sequences = FASTA('%s/etc/pdb_seqres.txt' % data_home)
    fasta_fname = get_pdb_seqres_with_msa_fasta_filename()
    cluster_dic, reverse_dic = get_cdhit_clusters(fasta_fname, similaritity_threshold)
    data_cullpdb = [[clstr[0], len(SEQs[clstr[0]])] for clstr in cluster_dic.values()]
    pd.DataFrame(data_cullpdb, columns=["IDs", "length"]).to_csv(cullpdb_path, index=False)


def download_cullpdb(data_home=DATA_HOME,
                     repo_path=REPO_PATH,
                     cullpdb_fname=CULL_PDB25_PATH,
                     seqres_fname='etc/pdb_seqres.txt', lim=5):
    uids = []
    path = os.path.join(data_home, cullpdb_fname)
    df = pd.read_csv(path, sep=r"\s*", engine='python')
    for s in tqdm(df.IDs[:lim], desc="sequences downloaded"):
        s = s.strip()
        pdb, chain = s[0:4].lower(), s[4:]
        src_path = os.path.join(repo_path, '%s.pdb.gz' % pdb)
        if not os.path.exists(src_path):
            fetchPDB(pdb, folder=os.path.dirname(src_path))
        uids.append("%s_%s" % (pdb, chain))
    order, sequences = FASTA(os.path.join(data_home, seqres_fname))
    to_fasta([sequences[uid] for uid in uids if uid in sequences], '%s.fas' % path)
    to_fasta([sequences[uid] for uid in uids
              if uid in sequences and not os.path.exists('%s/msas/%s.a3m' % (data_home, uid))],
             os.path.join(data_home, 'etc', 'delta.fas'))


def get_dictionary(pdb_id, data_home=DATA_HOME, include_ccm=False):
    pdb, chain_id = pdb_id.split('_')
    pth_to_msa = glob("%s/msas/%s.a3m" % (data_home, pdb_id))
    if len(pth_to_msa) != 1:
        raise FileNotFoundError("Could not find MSA!")

    sequence = str(SEQs[pdb_id].seq)
    residues = load_or_parse_residues(pdb, chain_id)
    if residues is None:
        return None

    prof = read_profile_from_msa(pdb_id)
    ccmpred = get_ccmpred(pdb_id, compute=False, to_tensor=False) if include_ccm else None
    seq, residues, cmat, prof = align_helper(sequence, prof, residues, ccmpred)
    coords = toX(residues, get_center=get_center2)

    if len(seq) < MIN_LENGTH:
        return None

    rec = {
        b'seq_id': hash(seq),
        b'name': "{}{}".format(pdb, chain_id).encode('utf-8'),
        b'length': len(seq),
        b'seqres': seq.encode('utf-8'),
        b'PSFM': prof,
        b'dist_matrix': get_distance_matrix(coords).cpu().data.numpy()
        }

    if ccmpred is not None:
        rec[b'ccmpredZ'] = ccmpred

    return rec


def build_dataset(dataset_name, cull_pdbs=CULL_PDBs):
    dataset = []
    for pdb_id, length in tqdm(cull_pdbs, desc="chains processed"):
        try:
            rec = get_dictionary(pdb_id)
            if rec is None: continue
            dataset.append(rec)
        except (FileNotFoundError, AssertionError) as e:
            continue
    if len(dataset) == 0:
        return
    save_dataset(dataset, dataset_name)


def build_pairs_dataset(dataset_name, pdb_pairs=PDB_PAIRS, cull_pdbs=CULL_PDBs, save_to_file=True):

    dataset_pairs, cache = [], defaultdict(get_dictionary)
    s_cull_pdbs = set([a[0] for a in cull_pdbs])
    for pdb1, pdb2, mut_indices, n_mut, length in tqdm(pdb_pairs, desc="chains processed"):

        if (cull_pdbs is not None) and (pdb2 not in s_cull_pdbs) and (pdb1 not in s_cull_pdbs):
            continue

        try:
            if pdb1 not in cache:
                cache[pdb1] = get_dictionary(pdb1)
            if pdb2 not in cache:
                cache[pdb2] = get_dictionary(pdb2)
        except FileNotFoundError:
            continue

        val1, val2 = cache[pdb1], cache[pdb2]

        if val1 is None or val2 is None:
            continue

        name1, seqres1, prof1, dmat1 = \
            val1[b'name'].decode('utf8'), val1[b'seqres'].decode('utf8'), val1[b'PSFM'], val1[b'dist_matrix']
        name2, seqres2, prof2, dmat2 = \
            val2[b'name'].decode('utf8'), val2[b'seqres'].decode('utf8'), val2[b'PSFM'], val2[b'dist_matrix']

        a, b, size, diff = find_maximal_matching_shift(seqres1, seqres2, k=1)
        if (size <= 1) or (len(diff) != 1):
            continue

        seq1, dmat1, prof1 = seqres1[a: a + size], dmat1[a: a + size, a: a + size], prof1[a: a + size, :]
        seq2, dmat2, prof2 = seqres2[b: b + size], dmat2[b: b + size, b: b + size], prof2[b: b + size, :]

        assert len(seq1) == len(seq2)
        assert dmat1.shape == dmat2.shape
        assert len(seq1) == dmat1.shape[0]
        assert len(seq2) == dmat2.shape[0]
        assert len(seq1) == prof1.shape[0]
        assert len(seq2) == prof2.shape[0]

        dataset_pairs.append({
            b'pdb1': val1[b'name'],
            b'pdb2': val2[b'name'],
            b'pair_id': (val1[b'seq_id'], val2[b'seq_id']),
            b'seq1': seq1.encode('utf-8'),
            b'seq2': seq2.encode('utf-8'),
            b'dmat1': dmat1,
            b'dmat2': dmat2,
            b'prof1': prof1,
            b'prof2': prof2,
            b'mutix': diff,
            b'length': len(seq1)})

    if save_to_file:
        save_dataset([d for d in cache.values() if d is not None], 'singles-%s' % dataset_name)
        save_dataset(dataset_pairs, 'pairs-%s' % dataset_name)

    return cache, dataset_pairs


def filter_pair_dataset(dataset_pairs, threshold):

    def subsample(arr, n=2):
        return arr[0:len(arr):n]

    def compute_correlation(d1, d2):
        a1, a2 = subsample(d1.ravel()), subsample(d2.ravel())
        return pearsonr(a1, a2)

    pair_id_to_list_of_pairs = defaultdict(list)
    for rec in tqdm(dataset_pairs, desc="pairs processed"):
        pair_id_to_list_of_pairs[rec[b'pair_id']].append(rec)

    dataset_filtered = []
    for pair_id, records in tqdm(pair_id_to_list_of_pairs.items(), desc="unique seq-pairs filtered"):
        deltas = [rec[b'dmat1'] - rec[b'dmat2'] for rec in records]
        if len(deltas) == 1:
            dataset_filtered.extend(records)
            continue
        cors = [compute_correlation(p1, p2) for (p1, p2) in combinations(deltas, 2)]
        avg_pearsonr = np.mean(cors)
        if avg_pearsonr < threshold:
            continue
        dataset_filtered.extend(records)

    return dataset_filtered


def run_ccmpred(pdbs=CULL_PDBs, ccm_home=CCM_HOME, lim=None):
    ccm_failures = []
    for pdb_id, _ in tqdm(pdbs[:lim], desc="chains processed"):
        try:
            get_ccmpred(pdb_id, to_tensor=False, compute=False)
        except (FileNotFoundError, AssertionError) as e:
            ccm_failures.append([pdb_id, str(e)])
    pd.DataFrame(ccm_failures, columns=["pdb_id", "error"]).to_csv("%s/ccm_failures.csv" % ccm_home, index=False)


def add_arguments(parser):
    parser.add_argument("-a", "--action", type=str,
                        choices=["download_cullpdb",
                                 "generate_cullpdb",
                                 "build_singles_datasets",
                                 "build_pairs_datasets",
                                 "generate_pairs",
                                 "run_pairs_analysis"],
                        required=True,
                        help="Choose what loss function to use.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.action == "generate_pairs":
        generate_pairs_with_cdhit()
    elif args.action == "run_pairs_analysis":
        run_analysis()
    elif args.action == "download_cullpdb":
        download_cullpdb(lim=None)
    elif args.action == "generate_cullpdb":
        generate_cullpdb()
    elif args.action == "build_singles_datasets":
        from src.utils.data_utils import TRAIN_SET_CULL_PDB, VALID_SET_CULL_PDB, TEST_SET_CULL_PDB
        build_dataset('train', cull_pdbs=TRAIN_SET_CULL_PDB)
        build_dataset('valid', cull_pdbs=VALID_SET_CULL_PDB)
        build_dataset('test', cull_pdbs=TEST_SET_CULL_PDB)
    elif args.action == "build_pairs_datasets":
        from src.utils.data_utils import TRAIN_SET_PAIRS, VALID_SET_PAIRS, TEST_SET_PAIRS
        build_pairs_dataset('train', pdb_pairs=TRAIN_SET_PAIRS)
        build_pairs_dataset('valid', pdb_pairs=VALID_SET_PAIRS)
        build_pairs_dataset('test', pdb_pairs=TEST_SET_PAIRS)
    else:
        raise ValueError("Unrecognized action")


if __name__ == "__main__":
    main()
