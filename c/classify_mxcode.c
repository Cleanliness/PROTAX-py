#include "defs.h"

int compute_cnode_probs_best2(TaxonomyNode *node, int nid, double prevprob, SequenceSet *rseq, Model *m, double **scs, char *seq, double pth, PtrCountItem *ptrcount, int oklen, char *seqid) {
  int i,j,cid,k;
  double dist,mindist1, mindist2, maxz,ezsum, *beta, *sc;
  
  beta = m->params[node[nid].level];
  sc = scs[node[nid].level];
  maxz = 0.0;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    mindist1 = 1.0;
    mindist2 = 1.0;
    for (j=0; j<node[cid].num_rseqs; j++) {
      k = node[cid].rseq_index[j];
      /* dist = pdist(seq, rseq->seq[k], rseq->alen); */
      dist = get_pdist(&(ptrcount[k]), oklen);
      if (dist < mindist1) {
	mindist2 = mindist1;
	mindist1 = dist;
      }
      else if (dist < mindist2) {
	mindist2 = dist;
      }
    }

    /* printf("  %s %f %f\n",node[cid].name,mindist,avedist); */
    
    /* use prob temporarily to store z */
    if (node[cid].isunk) {     
      node[cid].prob = 0.0;
      node[cid].no_rseqs = 1;
    }
    else if (node[cid].num_rseqs) {
      if (node[cid].num_rseqs==1) mindist2=mindist1;
      node[cid].prob = beta[1] + beta[2]*(mindist1-sc[0])/sc[1] + beta[3]*(mindist2 - mindist1 - sc[2])/sc[3];
      node[cid].no_rseqs = 0;
    }
    else {
      node[cid].prob = beta[0];
      node[cid].no_rseqs = 1;
    }

    if (node[cid].prob > maxz)
      maxz = node[cid].prob;
  }
  
  ezsum = 1e-100;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    node[cid].prob = node[cid].prior * exp(node[cid].prob - maxz);
    ezsum += node[cid].prob;
  }

  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    node[cid].prob /= ezsum;
  }

  node[nid].sumcprob_no_rseqs = 0.0;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    node[cid].prob *= prevprob;
    if (node[cid].no_rseqs)
      node[nid].sumcprob_no_rseqs += node[cid].prob;
    if ((node[cid].no_rseqs == 0) && (node[cid].prob >= pth)) {
      printf("%s %d %s %s %f",seqid,node[cid].level,node[cid].mxcode,node[cid].name, node[cid].prob);
      if (node[cid].num_cnodes)
	compute_cnode_probs_best2(node, cid, node[cid].prob, rseq, m, scs, seq, pth, ptrcount, oklen, seqid);
    }
  }
  if (node[nid].sumcprob_no_rseqs >= pth) {
    if (node[nid].level == 0)
      printf("%s %d NA %s %f\n",seqid,node[nid].level+1,UNKNAME, node[nid].sumcprob_no_rseqs);
    else
      printf("%s %d NA %s,%s %f\n",seqid,node[nid].level+1,node[nid].name, UNKNAME, node[nid].sumcprob_no_rseqs);
  }

  return(0);
}

double **read_level_scalings(char *filename, int *num_levels) {
  FILE *fp;
  char line[MAXLINE], *token;
  double **sc;
  int i, j, nlev;
  
  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR: cannot read scalingfile '%s'.\n",filename);
    perror(""); exit(-1);
  }

  nlev=0;
  while (fgets(line,MAXLINE,fp) != NULL) {
    nlev++;
  }

  if ((sc = (double **) malloc (nlev*sizeof(double *))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d double ptrs.\n", nlev);
    perror(""); exit(-1);
  }
  for (j=0; j<nlev; j++) {
    if ((sc[j] = (double *) malloc (4*sizeof(double))) == NULL) {
      fprintf(stderr,"ERROR: cannot malloc %d*4 doubles.\n",nlev);
      perror(""); exit(-1);
    }
  }
  
  rewind(fp);

  for (j=0; j<nlev; j++) {
    if (fgets(line,MAXLINE,fp) == NULL) {
      fprintf(stderr,"ERROR: cannot read scaling factors %d/%d from '%s'.\n",j+1,nlev,filename);
      perror(""); exit(-1);
    }
      
    if ((token = strtok(line," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read level %d 1st token from file '%s'.\n",nlev,filename);
      perror(""); exit(-1);
    }
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read level %d 2nd token from file '%s'.\n",nlev,filename);
      perror(""); exit(-1);
    }
    sc[j][0] = atof(token);
    for (i=1; i<4; i++) {
      if ((token = strtok(NULL," \t")) == NULL) {
	fprintf(stderr,"ERROR: cannot read level %d token %d from file '%s'.\n",nlev,2*i+1,filename);
	perror(""); exit(-1);
      }
      if ((token = strtok(NULL," \t")) == NULL) {
	fprintf(stderr,"ERROR: cannot read level %d token %d from file '%s'.\n",nlev,2*i+2,filename);
	perror(""); exit(-1);
      }
      sc[j][i] = atof(token);
    }
  }
  fclose(fp);
  
  *num_levels = nlev;
  return (sc);
}

int main (int argc, char **argv) {
  int i,j, num_tnodes, num_sclevels;
  SequenceSet *rseq,*iseq;
  TaxonomyNode *taxonomy;
  Model *model;
  double pth, **scs;
  PtrSeqItem *ptrseq;
  PtrCountItem *ptrcount;
  int global_num_notok_positions, oklen;  

  if (argc < 7) {
    fprintf(stderr,"usage: classify taxonomy rseqFASTA taxonomy2rseq modelparameters scalingfile probability_threshold inputFASTA\n");
    exit(0);	    
  }

  taxonomy = read_taxonomy(argv[1], &num_tnodes);
  rseq = read_aligned_sequences(argv[2]);
  add_rseq2taxonomy(argv[3], taxonomy);
  model = read_model(argv[4]);
  scs=read_level_scalings(argv[5], &num_sclevels);

  if (model->num_levels != num_sclevels) {
    fprintf(stderr,"ERROR: %d model levels but %d scaling levels, files '%s' and '%s'.\n", model->num_levels, num_sclevels, argv[4], argv[5]);
    exit(0);
  }
  
  pth = atof(argv[6]);  
  iseq = read_aligned_sequences(argv[7]);
  
  if (rseq->alen != iseq->alen) {
    fprintf(stderr,"ERROR: sequence lengths different in two files (%d,%d), files '%s','%s'.\n",rseq->alen,iseq->alen,argv[2],argv[6]);
    exit(0);
  }

  ptrseq = create_ptrseq(rseq);
  ptrcount = create_ptrcount(rseq->num_seqs);

  /*
    print_taxonomy(taxonomy, num_tnodes);
    print_model(model);
  */
  
  for (i=0; i<iseq->num_seqs; i++) {
    reset_ptrcount(ptrcount,rseq->num_seqs);
    ptrseq_pdist(ptrseq, iseq->seq[i], iseq->alen, ptrcount, &global_num_notok_positions);
    oklen = iseq->alen - global_num_notok_positions;
    compute_cnode_probs_best2(taxonomy, 0, 1.0, rseq, model, scs, iseq->seq[i], pth, ptrcount, oklen, iseq->id[i]);
  }
  
  return(0);
}
