"""
Modified from https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/utils.py 
"""

from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def lprint(string, f):
    print(string)
    f.write(string+'\n')


def load_fasta(path):
    """
    Extend to handle case where path is a folder with multiple a3ms from different databases
    """
    seqs, IDs =[], []
    seen_sequences = set()
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_name, ext = os.path.splitext(file)
            if ext not in [".a3m", ".fasta"]:
                continue
            file_path = os.path.join(path, file)
            with open(file_path) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq = ''.join([x for x in record.seq])
                    if seq in seen_sequences:
                        continue
                    seen_sequences.add(seq)
                    IDs.append(record.id)
                    seqs.append(seq)
    elif os.path.isfile(path):
        file_name, ext = os.path.splitext(os.path.basename(path))
        if ext not in [".a3m", ".fasta"]:
            raise KeyError("Input alignment should be a3m or fasta format")
        with open(path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = ''.join([x for x in record.seq])
                if seq in seen_sequences:
                    continue
                seen_sequences.add(seq)
                IDs.append(record.id)
                seqs.append(seq)
    return IDs, seqs


def write_fasta(names, seqs, outfile='tmp.fasta'):
        with open(outfile,'w') as f:
                for nm, seq in list(zip(names, seqs)):
                        f.write(">%s\n%s\n" % (nm, seq))


def encode_seqs(seqs, max_len=108, alphabet=None):
    
    if alphabet is None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    
    arr = np.zeros([len(seqs), max_len, len(alphabet)])
    for j, seq in enumerate(seqs):
        for i,char in enumerate(seq):
            for k, res in enumerate(alphabet):
                if char==res:
                    arr[j,i,k]+=1
    return arr.reshape([len(seqs), max_len*len(alphabet)])


def consensusVoting(seqs):
    ## Find the consensus sequence
    consensus = ""
    residues = "ACDEFGHIKLMNPQRSTVWY-"
    n_chars = len(seqs[0])
    for i in range(n_chars):
        baseArray = [x[i] for x in seqs]
        baseCount = np.array([baseArray.count(a) for a in list(residues)])
        vote = np.argmax(baseCount)
        consensus += residues[vote]

    return consensus


def plot_landscape(x, y, df, query_, plot_type, args):

    plt.figure(figsize=(5,5))
    tmp = df.loc[df.dbscan_label==-1]
    plt.scatter(tmp[x], tmp[y], color='lightgray', marker='x', label='unclustered')

    tmp = df.loc[df.dbscan_label>9]
    plt.scatter(tmp[x],tmp[y], color='black', label='other clusters')

    tmp = df.loc[df.dbscan_label>=0][df.dbscan_label<=9]
    sns.scatterplot(x=x,y=y, hue='dbscan_label', data=tmp, palette='tab10',linewidth=0)

    plt.scatter(query_[x],query_[y], color='red', marker='*', s=150, label='Ref Seq')
    plt.legend(bbox_to_anchor=(1,1), frameon=False)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

    plt.savefig(args.o+"/"+args.keyword+'_'+plot_type+'.pdf', bbox_inches='tight')
