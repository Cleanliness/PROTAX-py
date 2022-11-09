"""
Functions for reading from files used by PROTAX
"""
import model
from model import Protax
import numpy as np
from taxon import taxNode
import time


def read_params(pdir):
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))
    return np.array(res).astype("float64")


def read_scalings(pdir):
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))

    res = np.array(res)[:, 1::2].astype("float64")
    return res


def read_taxonomy(tdir):
    f = open(tdir)
    node_dat = f.readlines()
    nodes = [None for i in range(len(node_dat))]
    for l in node_dat:
        l = l.strip("\n")
        nid, pid, lvl, name, prior, d = l.split("\t")
        nid, pid, prior = (int(nid), int(pid), float(prior))
        name = name.split(",")[-1]

        curr = taxNode(prior, name == 'unk', name)
        nodes[nid] = curr
        nodes[pid].add_child(curr)
    return nodes


def read_refs(ref_dir):
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
    return ref_list, ref_lens


def get_seq_bits(seq_str):
    seq_chars = np.frombuffer(seq_str.encode('ascii'), np.int8)
    a = seq_chars == 65
    t = seq_chars == 84
    g = seq_chars == 71
    c = seq_chars == 67

    # reduce memory usage
    seq_bits = np.packbits(np.array([a, t, g, c]), axis=None)
    return seq_bits


def assign_refs(nodes, seq2tax_dir):
    f = open(seq2tax_dir)
    for l in f.readlines():
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)
        num_refs = int(num_refs)
        seqs = np.array(ref_idx.split(" ")).astype(int)

        nodes[nid].set_ref_seqs(seqs)
    bb = 3


if __name__ == "__main__":
    testdir = r"C:\Users\mli\Documents\roy\modelCOIfull"

    # reading model info
    beta = read_params(testdir + "\\model.pars")
    scalings = read_scalings(testdir + "\\model.scs")
    all_nodes = read_taxonomy(testdir + "\\taxonomy.priors")
    refs, ref_lens = read_refs(testdir + "\\refs.aln")
    assign_refs(all_nodes, testdir + "\\model.rseqs.numeric")

    # set up model
    m = Protax()
    m.set_params(beta, 0)
    m.set_scaling(scalings)
    m.set_reference_sequences(np.array(refs), np.array(ref_lens))

    # test query
    q = "-ACATTATATTTTATATTTGGAGCTTGAGCTGGGATAGTTGGAACAAGATTAAGAATTCTTATCCGAACTGAACTTGGTACCCCCGGGTCACTTATTGGAGATGACCAGATTTATAATGTAATTGTTACAGCTCACGCTTTTGTTATAATTTTTTTTATAGTTATACCAATTTTAATTGGTGGTTTCGGAAATTGACTTGTCCCATTAATATTAGGGGCACCTGATATAGCCTTCCCCCGAATAAATAACATAAGATTCTGGTTACTCCCCCCATCATTAACCCTTCTTTTAATAAGAAGAATAGTAGAAAGAGGAGCAGGAACAGGTTGAACAGTTTATCCTCCCTTGGCCTCAAATATTGCACATGGAGGGGCATCTGTCGATTTAGCAATTTTTAGTTTACATCTAGCAGGAATCTCCTCTATTTTAGGAGCAGTAAATTTTATTACAACAATTATCAATATACGAGCCCCTCAAATAAGGTTTGACCAAATACCTCTTTTTGTTTGAGCTGTGGGAATCACAGCTCTCCTTCTTCTTCTTTCTCTTCCAGTTTTAGCCGGAGCTATCACTATATTATTAACAGACCGGAATTTAAATACATCATTTTTTGACCCAGCAGGAGGTGGTGATCCTATTTTATACCAACATTTATTT"
    q = get_seq_bits(q)
    start_time = time.time()
    m.classify(all_nodes[0], q)
    print("classification took " + str(time.time() - start_time))


    h = 3
