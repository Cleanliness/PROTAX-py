#include "defs.h"

#define NSEQ2 3
#define NSEQ1 2
#define NSEQ0 1
#define UNK 0

char *NODE_TYPE[4] = {"unk", "nseq0", "nseq1", "nseq2"};

typedef struct {
  double weight, prior;
  int onode, node, nodetype, rseq1ind, trainseqind;
} TrainData;

TrainData *read_traindata(char *filename, int *num_trdat) {
  TrainData *tr;
  FILE *fp;
  char line[MAXLINE], *token;
  int i;

  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR: cannot read file '%s'.\n",filename);
    perror(""); exit(-1);
  }
  i=0;
  while (fgets(line,MAXLINE,fp))
    i++;
  
  *num_trdat = i;
  if ((tr = (TrainData *) malloc (i * sizeof(TrainData))) == NULL){
    fprintf(stderr,"ERROR: cannot malloc %d TrainData.\n",i);
    perror(""); exit(-1);
  }

  rewind(fp);

  i=0;
  while (fgets(line,MAXLINE,fp)) {
    if ((token = strtok(line," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 1st token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    tr[i].weight = atof(token);
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 2nd token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    tr[i].onode = atoi(token);
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 3rd token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    tr[i].prior = atof(token);
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 4th token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    if (!strcmp(token,"nseq2")) 
      tr[i].nodetype = NSEQ2;
    else if (!strncmp(token,"nseq1,",6)) {
      tr[i].nodetype = NSEQ1;
      tr[i].rseq1ind=atoi(token+6);
    }
    else if (!strcmp(token,"nseq0")) 
      tr[i].nodetype = NSEQ0;
    else if (!strcmp(token,"unk")) 
      tr[i].nodetype = UNK;
    else {
      fprintf(stderr,"ERROR: unknown nodetype '%s' (4th token) in line %d/%d file '%s'.\n",token,i+1,*num_trdat,filename);
      exit(-1);
    }

    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 5th token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    tr[i].node = atoi(token);
    if ((token = strtok(NULL," \t\n")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 6th token from line %d/%d file '%s'.\n",i+1,*num_trdat,filename);
      perror(""); exit(-1);
    }
    tr[i].trainseqind = atoi(token);    
    i++;
  }

  fclose(fp);
  return(tr);
}

double *malloc_double_array(int n) {
  double *a;
  if ((a = (double *) malloc(n * sizeof(double))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d doubles.\n",n);
    perror("");exit(-1);
  }
  return(a);
}

typedef struct {
  int num_nodes;
  int *node;
  double *prob;
} StartNode;

StartNode *read_startnodes(char *filename, int *num) {
  FILE *fp;
  char line[MAXLINE], *token;
  int i,j, n;
  StartNode *s;
  
  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR: cannot read '%s'\n",filename);
    perror(""); exit(-1);
  }
  n=0;
  while (fgets(line,MAXLINE,fp))
    n++;
  *num = n;

  if ((s = (StartNode *) malloc (n*sizeof(StartNode))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc %d StartNodes\n",n);
    perror(""); exit(-1);
  }

  rewind(fp);
  i=0;
  while (fgets(line,MAXLINE,fp)) {
    if ((token = strtok(line," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 1st token from line %d/%d file '%s'.\n",i+1,*num,filename);
      perror(""); exit(-1);
    }
    /* don't save 1st token (nodetype) */
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 2nd token from line %d/%d file '%s'.\n",i+1,*num,filename);
      perror(""); exit(-1);
    }
    /* don't save 2nd token (nodetype) */
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 3rd token from line %d/%d file '%s'.\n",i+1,*num,filename);
      perror(""); exit(-1);
    }
    /* don't save 3rd token (nodetype) */
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read 4th token from line %d/%d file '%s'.\n",i+1,*num,filename);
      perror(""); exit(-1);
    }    
    s[i].num_nodes = atoi(token);
    s[i].node = malloc_int_array(s[i].num_nodes);
    s[i].prob = malloc_double_array(s[i].num_nodes);
    for (j=0; j<s[i].num_nodes; j++) {
      if ((token = strtok(NULL," \t")) == NULL) {
	fprintf(stderr,"ERROR: cannot read startnode %d/%d from line %d/%d file '%s'.\n",j+1,s[i].num_nodes,i+1,*num,filename);
	perror(""); exit(-1);
      }
      s[i].node[j] = atoi(token);
      if ((token = strtok(NULL," \t\n")) == NULL) {
	fprintf(stderr,"ERROR: cannot read startprob %d/%d from line %d/%d file '%s'.\n",j+1,s[i].num_nodes,i+1,*num,filename);
	perror(""); exit(-1);
      }
      s[i].prob[j] = atof(token);
    }
    i++;
  }
  fclose(fp);
  return(s);
}

typedef struct {
  int num_rseqs;
  char *computed, *dont_use;      
  double *dist;  
} RseqDist;

RseqDist *create_rseqdist(int num) {
  RseqDist *r;
  int i;
  char *thisfunction = "create_rseqdist";

  if ((r = (RseqDist *) malloc (sizeof(RseqDist))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc RseqDist.\n",thisfunction);
    perror(""); exit(-1);    
  }
  r->num_rseqs = num;

  if ((r->computed = (char *) malloc (num * sizeof(char))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d chars.\n",thisfunction, num);
    perror(""); exit(-1);    
  }
  if ((r->dont_use = (char *) malloc (num * sizeof(char))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d chars.\n",thisfunction, num);
    perror(""); exit(-1);    
  }
  if ((r->dist = (double *) malloc (num * sizeof(double))) == NULL) {
    fprintf(stderr,"ERROR (%s): cannot malloc %d doubles.\n",thisfunction, num);
    perror(""); exit(-1);    
  }
  
  return(r);
}

int reset_rseqdist(RseqDist *r) {
  memset(r->computed,0,r->num_rseqs);
  memset(r->dont_use,0,r->num_rseqs);
  return(0);
}

int train_compute_cnode_probs(TaxonomyNode *node, int nid, double prevprob, SequenceSet *rseq, Model *m, double *sc, int seqind, double pth, RseqDist *r, int dont_use_node) {
  int i,j,cid,k, num_ok_rseqs;
  double mindist1, mindist2, maxz,ezsum, *beta;

  /* beta = m->params[node[nid].level]; */
  beta = m->params[0];
  maxz = 0.0;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    if (cid != dont_use_node) {
      num_ok_rseqs=0;
      mindist1 = 1.0;
      mindist2 = 1.0;
      for (j=0; j<node[cid].num_rseqs; j++) {
	k = node[cid].rseq_index[j];
	if (r->dont_use[k] == 0) {
	  if (!r->computed[k]) {
	    r->dist[k] = pdist(rseq->seq[seqind], rseq->seq[k], rseq->alen);
	    r->computed[k] = 1;
	  }
	  if (r->dist[k] < mindist1) {
	    if (num_ok_rseqs>0) mindist2=mindist1;
	    mindist1 = r->dist[k];
	  }
	  else if (r->dist[k] < mindist2)
	    mindist2=r->dist[k];
	  num_ok_rseqs++;
	}
      }
    
      /* use prob temporarily to store z */
      if (node[cid].isunk) {     
	node[cid].prob = 0.0;
	node[cid].no_rseqs = 1;
      }
      else if (num_ok_rseqs) {
	if (num_ok_rseqs==1) mindist2=mindist1;
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
  }
  
  ezsum = 1e-100;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    if (cid != dont_use_node) {
      node[cid].prob = node[cid].prior * exp(node[cid].prob - maxz);
      ezsum += node[cid].prob;
    }
  }

  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    if (cid != dont_use_node)
      node[cid].prob /= ezsum;
  }

  node[nid].sumcprob_no_rseqs = 0.0;
  for (i=0; i<node[nid].num_cnodes; i++) {
    cid = node[nid].cnode_index[i];
    if (cid != dont_use_node) {
      node[cid].prob *= prevprob;
      if (node[cid].no_rseqs)
	node[nid].sumcprob_no_rseqs += node[cid].prob;
      if ((node[cid].no_rseqs == 0) && (node[cid].prob >= pth)) {
	printf(" %s %f",node[cid].name, node[cid].prob);
	if (node[cid].num_cnodes)
	  train_compute_cnode_probs(node, cid, node[cid].prob, rseq, m, sc, seqind, pth, r, dont_use_node);
      }
    }
  }
  if (node[nid].sumcprob_no_rseqs >= pth) {
    if (node[nid].level == 0)
      printf(" %s %f",UNKNAME, node[nid].sumcprob_no_rseqs);
    else
      printf(" %s,%s %f",node[nid].name, UNKNAME, node[nid].sumcprob_no_rseqs);
  }
  return(0);
}

double *read_scaling(char *filename) {
  FILE *fp;
  char line[MAXLINE], *token;
  double *sc;
  int i;
  
  if ((fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr,"ERROR: cannot read scalingfile '%s'.\n",filename);
    perror(""); exit(-1);
  }

  if (fgets(line,MAXLINE,fp) == NULL) {
    fprintf(stderr,"ERROR: cannot read scaling factors from '%s'.\n",filename);
    perror(""); exit(-1);
  }
  fclose(fp);

  if ((sc = (double *) malloc (4*sizeof(double))) == NULL) {
    fprintf(stderr,"ERROR: cannot malloc 4 doubles.\n");
    perror(""); exit(-1);
  }
  if ((token = strtok(line," \t")) == NULL) {
    fprintf(stderr,"ERROR: cannot read 1st token from file '%s'.\n",filename);
    perror(""); exit(-1);
  }
  if ((token = strtok(NULL," \t")) == NULL) {
    fprintf(stderr,"ERROR: cannot read 2nd token from file '%s'.\n",filename);
    perror(""); exit(-1);
  }
  sc[0] = atof(token);
  for (i=1; i<4; i++) {
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read token %d from file '%s'.\n",2*i+1,filename);
      perror(""); exit(-1);
    }
    if ((token = strtok(NULL," \t")) == NULL) {
      fprintf(stderr,"ERROR: cannot read token %d from file '%s'.\n",2*i+2,filename);
      perror(""); exit(-1);
    }
    sc[i] = atof(token);
  }
  
  return (sc);
}

int main (int argc, char **argv) {
  int i,j,k,nid, avenum, num_cnodes, num_tnodes, num_trdat,targetlevel;
  SequenceSet *rseq;
  TaxonomyNode *taxonomy;
  Model *model;
  TrainData *trdat;
  double dist,mindist,avedist,pth,*sc;
  RseqDist *rd;
  int dont_use_node;
  StartNode *startprob;

  if (argc < 9) {
    fprintf(stderr,"usage: trainclassify_best2 taxonomy rseqFASTA taxonomy2rseq modelparameters scalingfile probability_threshold numeric_traindata startnodefile\n");
    exit(0);	    
  }

  taxonomy = read_taxonomy(argv[1], &num_tnodes);
  rseq = read_aligned_sequences(argv[2]);
  add_rseq2taxonomy(argv[3], taxonomy);
  model=read_model(argv[4]);
  sc=read_scaling(argv[5]);
  pth=atof(argv[6]);
  trdat = read_traindata(argv[7], &num_trdat);
  startprob = read_startnodes(argv[8], &k);

  /* printf("%f %f %f %f\n",sc[0],sc[1],sc[2],sc[3]); */
  
  if (k != num_trdat) {
    fprintf(stderr,"ERROR: %d trainitems but %d startnodes, files '%s', '%s'.\n",num_trdat,k,argv[6],argv[7]);
    exit(-1);
  }

  rd = create_rseqdist(rseq->num_seqs);
  
  for (i=0; i<num_trdat; i++) {
    nid = trdat[i].node;
    printf("%s %d %s",NODE_TYPE[trdat[i].nodetype], trdat[i].node, rseq->id[trdat[i].trainseqind]);
    reset_rseqdist(rd);

    rd->dont_use[trdat[i].trainseqind] = 1;
    dont_use_node = -1;

    if (nid < num_tnodes) {    
      if (trdat[i].nodetype == UNK) {
	for (j=0; j<taxonomy[nid].num_rseqs; j++)
	  rd->dont_use[taxonomy[nid].rseq_index[j]] = 1;
	dont_use_node = nid;
      }
      else if (trdat[i].nodetype == NSEQ0) {
	for (j=0; j<taxonomy[nid].num_rseqs; j++)
	  rd->dont_use[taxonomy[nid].rseq_index[j]] = 1;
	dont_use_node = -1;
      }
    }

    for (j=0; j<startprob[i].num_nodes; j++) 
      train_compute_cnode_probs(taxonomy, startprob[i].node[j], startprob[i].prob[j], rseq, model, sc, trdat[i].trainseqind, pth, rd, dont_use_node);
    
    printf("\n");
  }
  
  return(0);
}
