"""
Functions for reading from files used by PROTAX
"""
import model
import numpy as np
from taxon import TaxNode
import time
import jax
import jax.numpy as jnp


def read_params(pdir):
    """
    Read parameters from file
    """
    print("reading parameters")
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(jnp.fromstring(l, sep=" "))
    return jnp.array(res)


def read_scalings(pdir):
    print("reading scalings")
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))

    res = np.array(res)[:, 1::2].astype("float64")
    return jnp.array(res)


def read_taxonomy(tdir):
    print("reading taxonomy file")
    f = open(tdir)
    node_dat = f.readlines()

    res = {"prior": np.zeros((len(node_dat))),
           "prob": np.zeros((len(node_dat))),
           "children": [[] for i in range(len(node_dat))],
           "node_refs": [jnp.array([]) for i in range(len(node_dat))],
           "u_z": np.zeros((len(node_dat))),
           "unk": np.zeros((len(node_dat))).astype(bool),
           "layer": np.zeros((len(node_dat))).astype(np.int8),
           "refs": None,
           "ref_lens": None
           }

    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, d = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        name = name.split(",")[-1]

        # setting data
        res["prior"][nid] = prior
        res["layer"][nid] = lvl
        res["unk"][nid] = name == 'unk'

        if nid != pid:
            res["children"][pid].append(nid)

    # converting to jax arrays
    for i in range(len(res["children"])):
        res["children"][i] = jnp.array(np.array(res["children"][i]))
    res["prior"] = jnp.array(res["prior"])
    res["layer"] = jnp.array(res["layer"])
    res["unk"] = jnp.array(res["unk"])
    return res


def read_refs(ref_dir):
    print("reading reference sequences")
    f = open(ref_dir)
    ref_list = []
    ref_lens = []

    i = 1
    while True:
        name = f.readline().strip('\n').split('\t')[0]
        seqs = f.readline().strip('\n')
        seq_bits = get_seq_bits(seqs)

        if not seqs:
            break  # EOF
        ref_list.append(seq_bits)
        ref_lens.append(len(seqs))
        print('\r' + str(i), end='')
        i += 1
    return np.array(ref_list), np.array(ref_lens)


def get_seq_bits(seq_str):
    seq_chars = np.frombuffer(seq_str.encode('ascii'), np.int8)
    a = seq_chars == 65
    t = seq_chars == 84
    g = seq_chars == 71
    c = seq_chars == 67

    # reduce memory usage
    seq_bits = np.packbits(np.array([a, t, g, c]), axis=None)
    return seq_bits


def assign_refs(tree, seq2tax_dir):
    print("assigning reference sequences to taxa")
    f = open(seq2tax_dir)
    for l in f.readlines():
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)
        seqs = jnp.array(np.fromstring(ref_idx, sep=" ").astype(int))
        tree["node_refs"][nid] = jax.device_put(seqs)


if __name__ == "__main__":
    testdir = r"C:\Users\mli\Documents\roy\modelCOIfull"

    # reading model info
    beta = read_params(testdir + "\\model.pars")
    scalings = read_scalings(testdir + "\\model.scs")
    tree = read_taxonomy(testdir + "\\taxonomy.priors")
    refs, ref_lens = read_refs(testdir + "\\refs.aln")
    tree["refs"] = jax.device_put(refs)
    tree["ref_lens"] = jax.device_put(ref_lens)
    tree["prior"] = jax.device_put(tree["prior"])
    tree["prob"] = jnp.array(tree["prob"])
    tree["prob"] = tree["prob"].at[0].set(1)
    assign_refs(tree, testdir + "\\model.rseqs.numeric")

    params = {"beta": beta, "scaling": scalings}

    # test query
    q = "-ACATTATATTTTATATTTGGAGCTTGAGCTGGGATAGTTGGAACAAGATTAAGAATTCTTATCCGAACTGAACTTGGTACCCCCGGGTCACTTATTGGAGATGACCAGATTTATAATGTAATTGTTACAGCTCACGCTTTTGTTATAATTTTTTTTATAGTTATACCAATTTTAATTGGTGGTTTCGGAAATTGACTTGTCCCATTAATATTAGGGGCACCTGATATAGCCTTCCCCCGAATAAATAACATAAGATTCTGGTTACTCCCCCCATCATTAACCCTTCTTTTAATAAGAAGAATAGTAGAAAGAGGAGCAGGAACAGGTTGAACAGTTTATCCTCCCTTGGCCTCAAATATTGCACATGGAGGGGCATCTGTCGATTTAGCAATTTTTAGTTTACATCTAGCAGGAATCTCCTCTATTTTAGGAGCAGTAAATTTTATTACAACAATTATCAATATACGAGCCCCTCAAATAAGGTTTGACCAAATACCTCTTTTTGTTTGAGCTGTGGGAATCACAGCTCTCCTTCTTCTTCTTTCTCTTCCAGTTTTAGCCGGAGCTATCACTATATTATTAACAGACCGGAATTTAAATACATCATTTTTTGACCCAGCAGGAGGTGGTGATCCTATTTTATACCAACATTTATTT"
    q = jnp.array(get_seq_bits(q))
    model.seq_dist(q, refs, ref_lens)
    start_time = time.time()
    model.classify(q, 0, params, tree)
    # model.seq_dist(q, refs, ref_lens)
    print("classification took " + str(time.time() - start_time))


    h = 3
