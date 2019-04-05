import os
import itertools


def get_cdhit_clusters(fasta_filename, parse=lambda seq: seq.split('>')[1].split('...')[0].split('|')[0]):

    if not os.path.exists("%s_99.clstr" % fasta_filename):
        cline = '/usr/bin/cdhit -i %s -o %s_99 -c 0.99 -n 5' % (fasta_filename, fasta_filename)
        assert os.WEXITSTATUS(os.system(cline)) == 0
    cluster_file, cluster_dic, reverse_dic = open("%s_99.clstr" % fasta_filename), {}, {}

    print("Reading cluster groups...")
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    for cluster in cluster_groups:
        name = int(next(cluster).strip().split()[-1])
        ids = [parse(seq) for seq in next(cluster_groups)]
        cluster_dic[name] = ids
    for cluster, ids in cluster_dic.items():
        for seqid in ids:
            reverse_dic[seqid] = cluster
    print("Detected %s clusters (>%s%% similarity) groups..." % (len(cluster_dic), 30))

    return cluster_dic, reverse_dic
