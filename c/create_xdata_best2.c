#include "defs.h"

#define NSEQ2 3
#define NSEQ1 2
#define NSEQ0 1
#define UNK 0

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

int main (int argc, char **argv) {
  int i,j,k,pid,cid, num_cnodes, num_tnodes, num_trdat, ok;
  SequenceSet *rseq;
  TaxonomyNode *taxonomy;
  TrainData *trdat;
  double dist,mindist1,mindist2;

  if (argc < 5) {
    fprintf(stderr,"usage: create_xdata_best2 taxonomy rseqFASTA taxonomy2rseq numeric_traindata\n");
    exit(0);	    
  }

  taxonomy = read_taxonomy(argv[1], &num_tnodes);
  rseq = read_aligned_sequences(argv[2]);
  add_rseq2taxonomy(argv[3], taxonomy);

  trdat = read_traindata(argv[4], &num_trdat);
  
  for (i=0; i<num_trdat; i++) {
    fprintf(stderr,"create xdata %d/%d\n",i+1,num_trdat);
    pid=taxonomy[trdat[i].node].pid;
    k=-1;
    if (trdat[i].nodetype == UNK) {
      k=1;
      num_cnodes = taxonomy[pid].num_cnodes -1;
    }
    else {
      num_cnodes = taxonomy[pid].num_cnodes;
      for (j=0; j<taxonomy[pid].num_cnodes; j++) 
	if (taxonomy[pid].cnode_index[j] == trdat[i].node) {
	  k=j+1;
	  break;
	}
    }

    printf("%f\t%f\t1,%d,%d,4",trdat[i].weight,trdat[i].prior,k,num_cnodes);

    for (j=0; j<taxonomy[pid].num_cnodes; j++) {
      cid=taxonomy[pid].cnode_index[j];
      if (taxonomy[cid].isunk)
	printf(",0,0,0,0");
      else if (cid == trdat[i].node) {
	if (trdat[i].nodetype == NSEQ0) 
	  printf(",1,0,0,0");
	else if (trdat[i].nodetype == NSEQ1) {
	  dist=pdist(rseq->seq[trdat[i].trainseqind], rseq->seq[trdat[i].rseq1ind], rseq->alen);
	  printf(",0,1,%f,0",dist);
	} 
	else if (trdat[i].nodetype == NSEQ2) {
	  mindist1=1.0;
	  mindist2=1.0;
	  ok=0;
	  for (k=0; k<taxonomy[cid].num_rseqs; k++) {
	    if (taxonomy[cid].rseq_index[k] != trdat[i].trainseqind) {
	      dist=pdist(rseq->seq[trdat[i].trainseqind], rseq->seq[taxonomy[cid].rseq_index[k]], rseq->alen);
	      if (dist < mindist1) {
		if (ok) mindist2=mindist1;
		mindist1 = dist;
	      }
	      else if (dist < mindist2) {
		mindist2 = dist;
	      }
	      ok++;
	    }
	  }
	  if (ok>1)
	    printf(",0,1,%f,%f",mindist1,mindist2-mindist1);
	  else
	    printf(",0,1,%f,0",mindist1);
	}
	/* the remaining case is UNK which has already been taken care of (first item) */
      }
      else {
	if (taxonomy[cid].num_rseqs == 0)
	  printf(",1,0,0,0");
	else {
	  mindist1=1.0;
	  mindist2=1.0;
	  ok=0;
	  for (k=0; k<taxonomy[cid].num_rseqs; k++) {
	    dist=pdist(rseq->seq[trdat[i].trainseqind], rseq->seq[taxonomy[cid].rseq_index[k]], rseq->alen);
	    if (dist < mindist1) {
	      if (ok) mindist2=mindist1;
	      mindist1 = dist;
	    }
	    else if (dist < mindist2) {
	      mindist2 = dist;
	    }
	    ok++;
	  }
	  if (ok>1)
	    printf(",0,1,%f,%f",mindist1,mindist2-mindist1);
	  else
	    printf(",0,1,%f,0",mindist1);
	}
      }
    }
    printf("\n");
  }
  
  return(0);
}
