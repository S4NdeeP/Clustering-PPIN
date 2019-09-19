/*
 * File:   main.c
 * Author: Sandeep Narayan P, MithunMohan K.M
 *
 * Created on 5 June, 2014, 6:49 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include "uthash.h"
#include <string.h>

struct HashTable {
    char key[10];             /* key */
    int value;
    UT_hash_handle hh;        /* makes this structure Hash Table */
};

struct Matrix_CSR {
    int* columnIndices,*rowOffsets;
    float* values;
    int nonZeroValueCount;
    int rowCount;
};

struct Edge {
    int nodeNo;
    float weight;
    struct Edge* next;
};

struct Graph {
    int noOfNodes;
    int noOfEdges;
    struct Edge* node[12000];
};

void createGraph(struct Graph** graph) {
    struct Graph* g=(struct Graph*)malloc(sizeof(struct Graph));
    g->noOfNodes=0;
    g->noOfEdges=0;
    int i;
    for(i=0;i<12000;++i) {
        g->node[i]=NULL;
    }
    *graph=g;
}

struct Edge* createEdge(int nodeNo,float weight) {
    struct Edge* x=(struct Edge*)malloc(sizeof(struct Edge));
    x->nodeNo=nodeNo;
    x->weight=weight;
    x->next=NULL;
    return x;
}

void createHashTable(struct HashTable** table) {
    *table=NULL;
}

void addEntry(struct HashTable** table,char key[],int value) {
    struct HashTable* s= (struct HashTable*)malloc(sizeof(struct HashTable));
    strcpy(s->key,key);
    s->value = value;
    HASH_ADD_STR(*table,key,s);
}

int getValue(struct HashTable** table,char key[]) {
    struct HashTable* s= (struct HashTable*)malloc(sizeof(struct HashTable));
    HASH_FIND_STR(*table,key,s);
    if(s) {
        return (s->value);
    }
    return -1;
}

float calculateWeight(char confidenceScores[]) {
    return 1.0;
}

int insert(struct Edge** head,struct Edge* e) {
    struct Edge* i,*temp;
    if(*head==NULL) {
        *head=e;
    }
    else if(e->nodeNo<(*head)->nodeNo) {
        e->next=*head;
        *head=e;
    }
    else {
        i=*head;
        while(i!=NULL&&i->nodeNo<e->nodeNo) {
        temp=i;
        i=i->next;
        }

        if(i==NULL) {
            temp->next=e;
        }
        else {
            if(e->nodeNo==i->nodeNo) {
                //printf("Problem\n");
                return 0;
            }
            temp->next=e;
            e->next=i;
        }
    }
    return 1;
}

void addEdge(struct Graph* graph,int index1,int index2,int weight) {
    struct Edge* e=createEdge(index2,weight);
    if(insert(&graph->node[index1],e)) {
        graph->noOfEdges++;
    }
}


struct Matrix_CSR* constructMatrix_CSR(struct Graph* graph) {
    int i;
    struct Edge *e;
    struct Matrix_CSR* m=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));
    m->columnIndices=(int*)malloc(sizeof(int)*graph->noOfEdges);
    m->values=(float*)malloc(sizeof(float)*(graph->noOfEdges));
    m->rowOffsets=(int*)malloc(sizeof(int)*(graph->noOfNodes+1));

    int c=0,r=0,count=0;
    for(i=0;i<graph->noOfNodes;++i) {
        e=graph->node[i];
        m->rowOffsets[r++]=count;
        while(e!=NULL) {
            count++;
            m->values[c]=e->weight;
            m->columnIndices[c++]=e->nodeNo;
            e=e->next;
        }
    }
    m->rowOffsets[r]=count;
    m->nonZeroValueCount=graph->noOfEdges;
    m->rowCount=graph->noOfNodes;
    return m;
}

void printAdjMat(struct Graph* graph) {
    int i;
    struct Edge* e;
    for(i=0;i<graph->noOfNodes;++i) {
        e=graph->node[i];
        printf("\nNode %d->",i);
        while(e!=NULL) {
            printf("(%d,%f)",e->nodeNo,e->weight);
            e=e->next;
        }
    }
}

void deleteGraph(struct Graph* graph) {
    int i;
    struct Edge* e,*temp;
    for(i=0;i<graph->noOfNodes;++i) {
        e=graph->node[i];
        while(e!=NULL) {
            temp=e;
            e=e->next;
            free(temp);
        }
    }
}

/*
 * @Input Name of file in DIP format
 * @Output A CSR Matrix of type Matrix_CSR
 * Function converts a input file  into a matrix in CSR format
 */


struct Matrix_CSR* convertDIPtoMat_CSR(char* DIP_FileName,char NodeIdTable[10000][20]) {
    FILE* fileFromDatabase=fopen(DIP_FileName,"r");

    char line[5000];
    char interactorA[15],interactorB[15];
    char confidenceScores[5000];
    int nodeNo=0;

    struct HashTable* IdNodeTable;
    createHashTable(&IdNodeTable);

    struct Graph* PPIN;
    createGraph(&PPIN);
    fgets(line, sizeof(line), fileFromDatabase);
    int linecounter=2;
    while(fgets(line, sizeof(line), fileFromDatabase)) {
    	int interactorA_i,interactorB_i,confidenceScore_i,line_i;

        /*
         * Setting Interactor A
         */
        for(interactorA_i=0,line_i=0;line[line_i]!='\t'&&line[line_i]!='|';++interactorA_i,++line_i) {
            interactorA[interactorA_i]=line[line_i];
        }
        interactorA[interactorA_i]='\0';

        //fprintf(stderr,"\n%d: A is %s",linecounter,interactorA);
        /*
         * Skipping till beginning of interactorB's field
         */

        while(line[line_i]!='\t')
            line_i++;

        /*
         * Setting Interactor B
         */
        for(interactorB_i=0,line_i=line_i+1;line[line_i]!='\t'&&line[line_i]!='|';++interactorB_i,++line_i) {
            interactorB[interactorB_i]=line[line_i];
        }
        interactorB[interactorB_i]='\0';

        //fprintf(stderr,"\n%d: B is %s",linecounter,interactorB);

        /*
         * Skipping till beginning of third field
         */
        while(line[line_i]!='\t')
            line_i++;

        /*
         * Skipping till beginning of the 15th field
         */

        int fieldNumber;
        for(fieldNumber=3,line_i=line_i+1;fieldNumber<15;++line_i) {
            if(line[line_i]=='\t') {
                fieldNumber++;
            }
        }

        /*
         * Setting confidence score
         */

        //fprintf(stderr,"\nA Befor adding in tables %s",interactorA);

        for(confidenceScore_i=0;line[line_i]!='\n';++confidenceScore_i,++line_i) {
            confidenceScores[confidenceScore_i]=line[line_i];
        }
        confidenceScores[confidenceScore_i]='\0';
        //printf("\n%s\n",confidenceScores);
        //fprintf(stderr,"\nA After adding in tables %s",interactorA);

        /*
         * Adds a mapping from DIP unique identifier to a node
         */
        int index1,index2;
        float weight;

        index1=getValue(&IdNodeTable,interactorA);
        if(index1==-1) {

            index1=nodeNo;
            addEdge(PPIN,index1,index1,1.0);
            addEntry(&IdNodeTable,interactorA,nodeNo);
            //NodeIdTable[nodeNo]=(char*)malloc(20*sizeof(char));
            strcpy(NodeIdTable[nodeNo],interactorA);
            //fprintf(stderr,"\nA in Table%s",NodeIdTable[nodeNo]);
            nodeNo++;
        }

        index2=getValue(&IdNodeTable,interactorB);
        if(index2==-1) {
            index2=nodeNo;
            addEdge(PPIN,index2,index2,1.0);
            addEntry(&IdNodeTable,interactorB,nodeNo);
            //NodeIdTable[nodeNo]=(char*)malloc(20*sizeof(char));
            strcpy(NodeIdTable[nodeNo],interactorB);
            //fprintf(stderr,"\nB in Table%s",NodeIdTable[nodeNo]);
            nodeNo++;
        }

        weight=calculateWeight(confidenceScores);
        //fprintf(stderr,"\n%d: %d && %d && %f\n",linecounter++,index1,index2,weight);
        addEdge(PPIN,index1,index2,weight);
        addEdge(PPIN,index2,index1,weight);
    }
    printf("No of nodes= %d\n",nodeNo);
    printf("No of edges= %d\n",PPIN->noOfEdges);
    PPIN->noOfNodes=nodeNo;

    struct Matrix_CSR* PPIN_CSR;
    PPIN_CSR=constructMatrix_CSR(PPIN);
    deleteGraph(PPIN);
    printf("Protein-Protein Interaction Network Created...\n");
    return PPIN_CSR;
}

