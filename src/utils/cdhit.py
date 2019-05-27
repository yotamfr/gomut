import os
import itertools

CDHIT_HOME = '/home/yotamfr/development/gomut/cd-hit-v4.8.1-2019-0228'


def get_word_size(similarity_threshold):
    """
    http://www.bioinformatics.org/cd-hit/cd-hit-user-guide.pdf

    -n 5 for thresholds 0.7 ~ 1.0
    -n 4 for thresholds 0.6 ~ 0.7
    -n 3 for thresholds 0.5 ~ 0.6
    -n 2 for thresholds 0.4 ~ 0.5
    """
    if similarity_threshold >= 0.7:
        return 5
    elif similarity_threshold >= 0.6:
        return 4
    elif similarity_threshold >= 0.5:
        return 3
    else:
        return 2


def get_cdhit_clusters(fasta_filename, similarity_frac,
                       parse=lambda seq: seq.split('>')[1].split('...')[0].split('|')[0],
                       cdhit_home=CDHIT_HOME):

    similarity_perc = int(similarity_frac * 100)
    cluster_fname = "%s_%d.clstr" % (fasta_filename, similarity_perc)
    n = get_word_size(similarity_frac)

    if not os.path.exists(cluster_fname):
        cline = '%s/cd-hit -i %s -o %s_%d -c %.2f -n %d' % (cdhit_home, fasta_filename, fasta_filename,
                                                            similarity_perc, similarity_frac, n)
        assert os.WEXITSTATUS(os.system(cline)) == 0
    cluster_file, cluster_dic, reverse_dic = open(cluster_fname), {}, {}

    print("Reading cluster groups...")
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    for cluster in cluster_groups:
        name = int(next(cluster).strip().split()[-1])
        ids = [parse(seq) for seq in next(cluster_groups)]
        cluster_dic[name] = ids
    for cluster, ids in cluster_dic.items():
        for seqid in ids:
            reverse_dic[seqid] = cluster
    print("Detected %s clusters (>%s%% similarity) groups..." % (len(cluster_dic), similarity_perc))

    return cluster_dic, reverse_dic
