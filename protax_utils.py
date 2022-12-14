"""
Functions for reading from files used by PROTAX
"""
import model
import numpy as np
import time
import jax
import jax.numpy as jnp
from tree import TaxTree


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
    """
    Read scalings from file
    """
    print("reading scalings")
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))

    res = np.array(res)[:, 1::2].astype("float64")
    return jnp.array(res)


def read_taxonomy(tdir):
    """
    Read taxonomy tree from file
    """
    print("reading taxonomy file")
    f = open(tdir)
    node_dat = f.readlines()

    # making result arrays
    N = len(node_dat)
    N_packed = np.packbits(np.zeros((1, N), dtype=bool), axis=1).shape[1]
    res = np.zeros((N, N_packed), dtype=np.uint8)
    unks = np.zeros(N, dtype=bool)
    priors = np.zeros(N)
    layers = np.zeros(N, dtype=int)
    
    
    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, d = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        name = name.split(",")[-1]

        if nid != pid:
            row = np.unpackbits(res[pid])
            row[nid] = 1
            res[pid] = np.packbits(row)
        
        # information about node
        unks[nid] = name=="unk"
        layers[nid] = lvl
        priors[nid] = prior
    
    # converting to jax
    res = jnp.array(res)
    unks = jnp.array(unks)
    layers = jnp.array(layers)
    priors = jnp.array(priors)
    return res, unks, layers, priors


def read_refs(ref_dir):
    """
    Read reference sequences from file
    """
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

    # converting to jax array
    return jnp.array(np.array(ref_list)), jnp.array(np.array(ref_lens))


def assign_refs(seq2tax_dir, N, R):
    """
    Assign reference sequences to nodes from file
    """
    print("assigning reference sequences to taxa")
    f = open(seq2tax_dir)

    R_packed = np.packbits(np.zeros((1, R), dtype=bool), axis=1).shape[1]
    res = np.zeros((N, R_packed), dtype=np.uint8)
    ref_mask = np.zeros(N, dtype=bool)

    # assigning ref seq indices
    for n, l in enumerate(f.readlines()):
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)

        ref_mask[nid] = int(num_refs) > 0
        seqs = np.fromstring(ref_idx, sep=" ").astype(int)
        row = np.unpackbits(res[n])
        row[seqs] = 1
        res[n] = np.packbits(row)

    return jnp.array(res), ref_mask

def get_seq_bits(seq_str):
    """
    Convert seqence string to bit representation
    """
    seq_chars = np.frombuffer(seq_str.encode('ascii'), np.int8)
    a = seq_chars == 65
    t = seq_chars == 84
    g = seq_chars == 71
    c = seq_chars == 67

    # reduce memory usage
    seq_bits = np.packbits(np.array([a, t, g, c]), axis=None)
    return seq_bits


def run_inference_tests():
    testdir = r"/h/royga/Documents/PROTAX-dsets/30k_small"

    # reading model info
    beta = read_params(testdir + "/model.pars")
    scalings = read_scalings(testdir + "/model.scs")
    adj, unk, layer, pr = read_taxonomy(testdir + "/taxonomy.priors")
    refs, ref_lens = read_refs(testdir + "/refs.aln")
    N = adj.shape[0]
    R = refs.shape[0]
    n_refs, has_refs = assign_refs(testdir + "/model.rseqs.numeric", N, R)

    start_prob = jnp.zeros(N).at[0].set(1)
    # creating taxonomic tree data object
    tax_tree = TaxTree(
        refs = refs,
        ref_lens=ref_lens,
        node_refs=n_refs,
        layer=layer,
        prior=pr,
        prob=start_prob,
        children=adj,
        has_refs=has_refs,
        unk = unk,
        visit_q=jnp.zeros(N, dtype=int),  # assume the root index is 0
        q_end=1,
        visited=0
    )

    # test query
    q = "-ACATTATATTTTATATTTGGAGCTTGAGCTGGGATAGTTGGAACAAGATTAAGAATTCTTATCCGAACTGAACTTGGTACCCCCGGGTCACTTATTGGAGATGACCAGATTTATAATGTAATTGTTACAGCTCACGCTTTTGTTATAATTTTTTTTATAGTTATACCAATTTTAATTGGTGGTTTCGGAAATTGACTTGTCCCATTAATATTAGGGGCACCTGATATAGCCTTCCCCCGAATAAATAACATAAGATTCTGGTTACTCCCCCCATCATTAACCCTTCTTTTAATAAGAAGAATAGTAGAAAGAGGAGCAGGAACAGGTTGAACAGTTTATCCTCCCTTGGCCTCAAATATTGCACATGGAGGGGCATCTGTCGATTTAGCAATTTTTAGTTTACATCTAGCAGGAATCTCCTCTATTTTAGGAGCAGTAAATTTTATTACAACAATTATCAATATACGAGCCCCTCAAATAAGGTTTGACCAAATACCTCTTTTTGTTTGAGCTGTGGGAATCACAGCTCTCCTTCTTCTTCTTTCTCTTCCAGTTTTAGCCGGAGCTATCACTATATTATTAACAGACCGGAATTTAAATACATCATTTTTTGACCCAGCAGGAGGTGGTGATCCTATTTTATACCAACATTTATTT" 
    q2 = "-AAAATATATTTTATATTTGGAGCTTGAGCTGGGATAGTTGGAACAAGATTAAGAATTCTTATGGGAACTGAACTTGGTACCCCCGGGTCACTTATTGGAGATGACCAGATTTATAATGTAATTGTTACAGCTCACGCTTTTGTTATAATTTTTTTTATAGTTATACCAATTTTAATTGGTGGTTTCGGAAATTGACTTGTCCCATTAATATTAGGGGCACCTGATATAGCCTTCCCCCGAATAAATAACATAAGATTCTGGTTACTCCCCCCATCATTAACCCTTCTTTTAATAAGAAGAATAGTAGAAAGAGGAGCAGGAACAGGTTGAACAGTTTATCCTCCCTTGGCCTCAAATATTGCACATGGAGGGGCATCTGTCGATTTAGCAATTTTTAGTTTACATCTAGCAGGAATCTCCTCTATTTTAGGAGCAGTAAATTTTATTACAACAATTATCAATATACGAGCCCCTCAAATAAGGTTTGACCAAATACCTCTTTTTGTTTGAGCTGTGGGAATCACAGCTCTCCTTCTTCTTCTTTCTCTTCCAGTTTTAGCCGGAGCTATCACTATATTATTAACAGACCGGAATTTAAATACATCATTTTTTGACCCAGCAGGAGGTGGTGATCCTATTTTATACCAACATTTATTT"
    q = jnp.array(get_seq_bits(q))
    q2 = jnp.array(get_seq_bits(q2))
    R = tax_tree.refs.shape[0]             # num references
    N = tax_tree.children.shape[0]         # num nodes

    print(f"beginning classification for {N} Nodes")
    start_time = time.time()
    res = model.classify(q, tax_tree, beta, scalings, N, R)
    res.block_until_ready()
    print("classification took " + str(time.time() - start_time))
    print(res.at[:10].get())

    start_time = time.time()
    res = model.classify(q, tax_tree, beta, scalings, N, R)
    res.block_until_ready()
    print("classification took " + str(time.time() - start_time))
    print(res.at[:10].get())

    # 2nd classification without compilation overhead time
    tax_tree = TaxTree(
        refs = refs,
        ref_lens=ref_lens,
        node_refs=n_refs,
        layer=layer,
        prior=pr,
        prob=start_prob,
        children=adj,
        has_refs=has_refs,
        unk = unk,
        visit_q=jnp.zeros(N, dtype=int),  # assume the root index is 0
        q_end=1,
        visited=0
    )
    start_time = time.time()
    res = model.classify(q2, tax_tree, beta, scalings, N, R)
    res.block_until_ready()
    print("classification took " + str(time.time() - start_time))
    print(res.at[:10].get())


if __name__ == "__main__":
    testdir = r"/h/royga/Documents/PROTAX-dsets/30k_small"

    # reading refs
    refs, ref_lens = read_refs(testdir + "/refs.aln")

    # test query
    q = "-ACATTATATTTTATATTTGGAGCTTGAGCTGGGATAGTTGGAACAAGATTAAGAATTCTTATCCGAACTGAACTTGGTACCCCCGGGTCACTTATTGGAGATGACCAGATTTATAATGTAATTGTTACAGCTCACGCTTTTGTTATAATTTTTTTTATAGTTATACCAATTTTAATTGGTGGTTTCGGAAATTGACTTGTCCCATTAATATTAGGGGCACCTGATATAGCCTTCCCCCGAATAAATAACATAAGATTCTGGTTACTCCCCCCATCATTAACCCTTCTTTTAATAAGAAGAATAGTAGAAAGAGGAGCAGGAACAGGTTGAACAGTTTATCCTCCCTTGGCCTCAAATATTGCACATGGAGGGGCATCTGTCGATTTAGCAATTTTTAGTTTACATCTAGCAGGAATCTCCTCTATTTTAGGAGCAGTAAATTTTATTACAACAATTATCAATATACGAGCCCCTCAAATAAGGTTTGACCAAATACCTCTTTTTGTTTGAGCTGTGGGAATCACAGCTCTCCTTCTTCTTCTTTCTCTTCCAGTTTTAGCCGGAGCTATCACTATATTATTAACAGACCGGAATTTAAATACATCATTTTTTGACCCAGCAGGAGGTGGTGATCCTATTTTATACCAACATTTATTT" 
    q = jnp.array(get_seq_bits(q))

    jit_times = []
    nojit_times = []
    for times in range(4):
        for i in range(times):
            refs = jnp.concatenate((refs, refs), axis=0)
            ref_lens = jnp.concatenate((ref_lens, ref_lens), axis=0)

        # with jit compilation time
        jitted = jax.jit(model.seq_dist)
        start_time = time.time()
        jitted(q, refs, ref_lens).block_until_ready()
        res_time = time.time() - start_time
        jit_times.append(res_time)
        print("distance (with jit) took " + str(res_time))

        # without jit
        start_time = time.time()
        jitted(q, refs, ref_lens).block_until_ready()
        res_time = time.time() - start_time
        nojit_times.append(res_time)
        print("distance (without jit) took " + str(res_time))
    
    print(f"jit times: {jit_times}")
    print(f"no jit times {nojit_times}")