"""
Cluster sequences in a MSA using DBSCAN algorithm and write .a3m file for each cluster.
Assumes first sequence in fasta is the query sequence.

rk.msacluster v1 -i alignments/ -o msaclusters --run_TSNE

This script is modified from AF_cluster repo:
https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/ClusterMSA.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from polyleven import levenshtein
from sklearn.cluster import DBSCAN

from rocket.msacluster_utils import (
    consensusVoting,
    encode_seqs,
    load_fasta,
    lprint,
    plot_landscape,
    write_fasta,
)


def main():
    p = argparse.ArgumentParser(
        description="""
    Cluster sequences in a MSA using DBSCAN algorithm and write .a3m file for each cluster.
    Assumes first sequence in fasta is the query sequence.

    H Wayment-Steele, 2022

    Modified by Minhuan Li for ROCKET, 2024
    """  # noqa: E501
    )

    p.add_argument(
        "keyword", action="store", help="Keyword to call all generated MSAs."
    )
    p.add_argument(
        "-i",
        action="store",
        help="fasta/a3m file of original alignment, or path containing fasta/a3m files",
    )
    p.add_argument(
        "-o", action="store", help="name of output directory to write MSAs to."
    )
    p.add_argument(
        "--n_controls",
        action="store",
        default=10,
        type=int,
        help="Number of control msas to generate (Default 10)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print cluster info as they are generated.",
    )

    p.add_argument(
        "--scan", action="store_true", help="Select eps value on 1/4 of data, shuffled."
    )
    p.add_argument(
        "--eps_val",
        action="store",
        type=float,
        help="Use single value for eps instead of scanning.",
    )
    p.add_argument(
        "--resample",
        action="store_true",
        help="If included, will resample the original MSA with replacement before writing.",  # noqa: E501
    )
    p.add_argument(
        "--gap_cutoff",
        action="store",
        type=float,
        default=0.25,
        help="Remove sequences with gaps representing more than this frac of seq.",
    )
    p.add_argument(
        "--min_eps",
        action="store",
        default=3,
        help="Min epsilon value to scan for DBSCAN (Default 3).",
    )
    p.add_argument(
        "--max_eps",
        action="store",
        default=20,
        help="Max epsilon value to scan for DBSCAN (Default 20).",
    )
    p.add_argument(
        "--eps_step",
        action="store",
        default=0.5,
        help="step for epsilon scan for DBSCAN (Default 0.5).",
    )
    p.add_argument(
        "--min_samples",
        action="store",
        default=10,
        help="Default min_samples for DBSCAN (Default 3, recommended no lower than that).",  # noqa: E501
    )

    p.add_argument(
        "--run_PCA",
        action="store_true",
        help="Run PCA on one-hot embedding of sequences and store in output_cluster_metadata.tsv",  # noqa: E501
    )
    p.add_argument(
        "--run_TSNE",
        action="store_true",
        help="Run TSNE on one-hot embedding of sequences and store in output_cluster_metadata.tsv",  # noqa: E501
    )

    args = p.parse_args()

    if args.run_PCA:
        from sklearn.decomposition import PCA

    if args.run_TSNE:
        from sklearn.manifold import TSNE

    os.makedirs(args.o, exist_ok=True)
    f = open(f"msacluster_{args.keyword}.log", "w")  # noqa: SIM115
    IDs, seqs = load_fasta(args.i)

    seqs = [
        "".join([x for x in s if x.isupper() or x == "-"]) for s in seqs
    ]  # remove lowercase letters in alignment

    df = pd.DataFrame({"SequenceName": IDs, "sequence": seqs})

    query_ = df.iloc[:1]
    df = df.iloc[1:]

    if args.resample:
        df = df.sample(frac=1)

    L = len(df.sequence.iloc[0])
    N = len(df)  # noqa: F841

    df["frac_gaps"] = [x.count("-") / L for x in df["sequence"]]

    former_len = len(df)
    df = df.loc[df.frac_gaps < args.gap_cutoff]

    new_len = len(df)
    lprint(args.keyword, f)
    lprint(
        f"{former_len - new_len} seqs removed for containing more than {int(args.gap_cutoff * 100)}% gaps, {new_len} remaining",  # noqa: E501
        f,
    )

    ohe_seqs = encode_seqs(df.sequence.tolist(), max_len=L)

    n_clusters = []
    eps_test_vals = np.arange(args.min_eps, args.max_eps + args.eps_step, args.eps_step)

    if args.eps_val is None:  # performing scan
        lprint("eps\tn_clusters\tn_not_clustered", f)

        for eps in eps_test_vals:
            testset = encode_seqs(df.sample(frac=0.25).sequence.tolist(), max_len=L)
            clustering = DBSCAN(eps=eps, min_samples=args.min_samples).fit(testset)
            n_clust = len(set(clustering.labels_))
            n_not_clustered = len(
                clustering.labels_[np.where(clustering.labels_ == -1)]
            )
            lprint("%.2f\t%d\t%d" % (eps, n_clust, n_not_clustered), f)  # noqa: UP031
            n_clusters.append(n_clust)
            if eps > 10 and n_clust == 1:
                break

        eps_to_select = eps_test_vals[np.argmax(n_clusters)]
    else:
        eps_to_select = args.eps_val

    # perform actual clustering

    clustering = DBSCAN(eps=eps_to_select, min_samples=args.min_samples).fit(ohe_seqs)

    lprint("Selected eps={:.2f}".format(eps_to_select), f)  # noqa: UP032

    lprint("%d total seqs" % len(df), f)  # noqa: UP031

    df["dbscan_label"] = clustering.labels_

    clusters = [x for x in df.dbscan_label.unique() if x >= 0]
    unclustered = len(df.loc[df.dbscan_label == -1])

    lprint(
        "%d clusters, %d of %d not clustered (%.2f)"  # noqa: UP031
        % (len(clusters), unclustered, len(df), unclustered / len(df)),
        f,
    )

    avg_dist_to_query = np.mean([
        1 - levenshtein(x, query_["sequence"].iloc[0]) / L
        for x in df.loc[df.dbscan_label == -1]["sequence"].tolist()
    ])
    lprint("avg identity to query of unclustered: %.2f" % avg_dist_to_query, f)  # noqa: UP031

    avg_dist_to_query = np.mean([
        1 - levenshtein(x, query_["sequence"].iloc[0]) / L
        for x in df.loc[df.dbscan_label != -1]["sequence"].tolist()
    ])
    lprint("avg identity to query of clustered: %.2f" % avg_dist_to_query, f)  # noqa: UP031

    cluster_metadata = []
    for clust in clusters:
        tmp = df.loc[df.dbscan_label == clust]

        cs = consensusVoting(tmp.sequence.tolist())

        avg_dist_to_cs = np.mean([
            1 - levenshtein(x, cs) / L for x in tmp.sequence.tolist()
        ])
        avg_dist_to_query = np.mean([
            1 - levenshtein(x, query_["sequence"].iloc[0]) / L
            for x in tmp.sequence.tolist()
        ])

        if args.verbose:
            print("Cluster %d consensus seq, %d seqs:" % (clust, len(tmp)))  # noqa: UP031
            print(cs)
            print("#########################################")
            for _, row in tmp.iterrows():
                print(row["SequenceName"], row["sequence"])
            print("#########################################")

        tmp = pd.concat([query_, tmp], axis=0)

        cluster_metadata.append({
            "cluster_ind": clust,
            "consensusSeq": cs,
            "avg_lev_dist": "%.3f" % avg_dist_to_cs,  # noqa: UP031
            "avg_dist_to_query": "%.3f" % avg_dist_to_query,  # noqa: UP031
            "size": len(tmp),
        })

        write_fasta(
            tmp.SequenceName.tolist(),
            tmp.sequence.tolist(),
            outfile=args.o + "/" + args.keyword + "_" + "%03d" % clust + ".a3m",  # noqa: UP031
        )

    print(f"writing {args.n_controls} size-10 uniformly sampled clusters", flush=True)
    for i in range(args.n_controls):
        tmp = df.sample(n=10)
        tmp = pd.concat([query_, tmp], axis=0)
        write_fasta(
            tmp.SequenceName.tolist(),
            tmp.sequence.tolist(),
            outfile=args.o + "/" + "U10-" + args.keyword + "_" + "%03d" % i + ".a3m",  # noqa: UP031
        )
    if len(df) > 100:
        print(
            f"writing {args.n_controls} size-100 uniformly sampled clusters", flush=True
        )
        for i in range(args.n_controls):
            tmp = df.sample(n=100)
            tmp = pd.concat([query_, tmp], axis=0)
            write_fasta(
                tmp.SequenceName.tolist(),
                tmp.sequence.tolist(),
                outfile=args.o
                + "/"
                + "U100-"
                + args.keyword
                + "_"
                + "%03d" % i  # noqa: UP031
                + ".a3m",
            )

    if args.run_PCA:
        lprint("Running PCA ...", f)
        ohe_vecs = encode_seqs(df.sequence.tolist(), max_len=L)
        mdl = PCA()
        embedding = mdl.fit_transform(ohe_vecs)

        query_embedding = mdl.transform(
            encode_seqs(query_.sequence.tolist(), max_len=L)
        )

        df["PC 1"] = embedding[:, 0]
        df["PC 2"] = embedding[:, 1]

        query_["PC 1"] = query_embedding[:, 0]
        query_["PC 2"] = query_embedding[:, 1]

        plot_landscape("PC 1", "PC 2", df, query_, "PCA", args)

        lprint("Saved PCA plot to " + args.o + "/" + args.keyword + "_PCA.pdf", f)

    if args.run_TSNE:
        lprint("Running TSNE ...", f)
        ohe_vecs = encode_seqs(
            df.sequence.tolist() + [query_.sequence.tolist()], max_len=L
        )
        # different than PCA because tSNE doesn't have .transform attribute

        mdl = TSNE()
        embedding = mdl.fit_transform(ohe_vecs)

        df["TSNE 1"] = embedding[:-1, 0]
        df["TSNE 2"] = embedding[:-1, 1]

        query_["TSNE 1"] = embedding[-1:, 0]
        query_["TSNE 2"] = embedding[-1:, 1]

        plot_landscape("TSNE 1", "TSNE 2", df, query_, "TSNE", args)

        lprint("Saved TSNE plot to " + args.o + "/" + args.keyword + "_TSNE.pdf", f)

    outfile = args.o + "/" + args.keyword + "_clustering_assignments.tsv"
    lprint("wrote clustering data to %s" % outfile, f)  # noqa: UP031
    df.to_csv(outfile, index=False, sep="\t")

    metad_outfile = args.o + "/" + args.keyword + "_cluster_metadata.tsv"
    lprint("wrote cluster metadata to %s" % metad_outfile, f)  # noqa: UP031
    metad_df = pd.DataFrame.from_records(cluster_metadata)
    metad_df.to_csv(metad_outfile, index=False, sep="\t")

    print("Saved this output to msacluster_%s.log" % args.keyword)  # noqa: UP031
    f.close()
