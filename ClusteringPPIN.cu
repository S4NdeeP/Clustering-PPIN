/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include "convertDIPtoMat_CSR.cu"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*
struct Matrix_CSR {
    int* columnIndices,*rowOffsets;
    float* values;
    int nonZeroValueCount;
    int rowCount;
};
*/

/**
 * CUDA kernel function that squares each element of the array.
 */
__global__ void square_row(struct Matrix_CSR* d_m,int inflationParameter) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<d_m->nonZeroValueCount) {
		d_m->values[i]=pow(d_m->values[i],inflationParameter);
		d_m->values[i]=((int)(d_m->values[i]*1000.0))/1000.0;

	}
}

__global__ void normalize_row(struct Matrix_CSR* d_m) {
	int i,j;
	float sum;
	i=blockIdx.x*blockDim.x+threadIdx.x;

	if(i<d_m->rowCount) {
		sum=0.0;

		for(j=d_m->rowOffsets[i];j<d_m->rowOffsets[i+1];j++) {
			sum=sum+d_m->values[j];
		}

		for(j=d_m->rowOffsets[i];j<d_m->rowOffsets[i+1];j++) {
			d_m->values[j]=d_m->values[j]/sum;
			d_m->values[i]=((int)(d_m->values[i]*1000.0))/1000.0;

		}
	}
}


Matrix_CSR * inflateRow(struct Matrix_CSR *d_m,int nonZeroValueCount,int rowCount,int inflationParameter) {
	square_row<<<((nonZeroValueCount+512)/512),512>>>(d_m,inflationParameter);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete

	normalize_row<<<((rowCount+512)/512),512>>>(d_m);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete

	return d_m;
}


__global__ void row_based_mult(int row,struct Matrix_CSR* primeValue,struct Matrix_CSR *subValue,float *tempColumn){
	int j,k,l;
	float tempSum=0.0;
	j=blockIdx.x*blockDim.x+threadIdx.x;
	if(j<primeValue->rowCount){

		for(k=primeValue->rowOffsets[row];k<primeValue->rowOffsets[row+1];k++) {
				for(l=subValue->rowOffsets[primeValue->columnIndices[k]];l<subValue->rowOffsets[primeValue->columnIndices[k]+1];l++  ) {
					if(j==subValue->columnIndices[l]) {
						tempSum=tempSum+primeValue->values[k]*subValue->values[l];
						break;
					}
					if(j<subValue->columnIndices[l]) {
						break;
					}
				}
		}
		tempColumn[j]=tempSum;
		}
	}

	__global__ void row_based_induction(int row,struct Matrix_CSR* d_m,float *tempValues,int *tempColumnIndices,int *tempRowOffsets,float *tempColumn,int *countResult){
	int i,count;
	tempRowOffsets[0]=0;
	count=tempRowOffsets[row];
		for(i=0;i<d_m->rowCount;i++){
			if(tempColumn[i]>=0.00001){
				tempValues[count]=tempColumn[i];
				tempValues[count]=((int)(tempValues[count]*1000.0))/1000.0;
				tempColumnIndices[count]=i;
				count=count+1;
			}
		}
		tempRowOffsets[row+1]=count;
	*countResult=count;
}

__global__ void copy_function(int row,float *newTempValues, int *newTempColumnIndices, float *tempValues, int *tempColumnIndices,int currentSize){
int i=blockIdx.x*blockDim.x+threadIdx.x;
	 if(i<currentSize){
		newTempValues[i]=tempValues[i];
		newTempColumnIndices[i]=tempColumnIndices[i];
		}
}

__global__ void reinitialize_function(struct Matrix_CSR* d_m,float *tempValues,int *tempColumnIndices,int *tempRowOffsets,int nonZeroValueCount){

	d_m->values=tempValues;
	d_m->columnIndices=tempColumnIndices;
	d_m->rowOffsets=tempRowOffsets;
	d_m->nonZeroValueCount=nonZeroValueCount;

}


int matrix_multiplication(struct Matrix_CSR *primeValue,struct Matrix_CSR *subValue,int rowCount,int clearFlag)
{
	int i,assignedSize,currentSize,startFree=0;
	float *tempColumn, *tempValues, *newTempValues, *flTempFree;
	int *tempColumnIndices, *newTempColumnIndices, *tempRowOffsets, *intTempFree1, *countResult;
	assignedSize=rowCount;

	struct Matrix_CSR *temp;
	temp=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));


	CUDA_CHECK_RETURN(cudaMalloc((void**) &tempColumn, sizeof(float)*rowCount));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &tempValues, sizeof(float)*rowCount));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &tempColumnIndices, sizeof(int)*rowCount));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &tempRowOffsets, sizeof(int)*(rowCount+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &countResult, sizeof(int)));


	for(i=0;i<rowCount;i++){
		cudaMemset(tempColumn,0,sizeof(float)*rowCount);
		row_based_mult<<<((rowCount+512)/512),512>>>(i,primeValue,subValue,tempColumn);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		row_based_induction<<<1,1>>>(i,primeValue,tempValues,tempColumnIndices,tempRowOffsets,tempColumn,countResult);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaMemcpy(&currentSize,countResult,sizeof(int),cudaMemcpyDeviceToHost));
		if((assignedSize-currentSize)<(rowCount)){
			assignedSize=2*assignedSize;
			CUDA_CHECK_RETURN(cudaMalloc((void**) &newTempValues, sizeof(float)*assignedSize));
			CUDA_CHECK_RETURN(cudaMalloc((void**) &newTempColumnIndices, sizeof(int)*assignedSize));
			copy_function<<<((currentSize+512)/512),512>>>(i,newTempValues,newTempColumnIndices,tempValues,tempColumnIndices,currentSize);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
			flTempFree=tempValues;
			intTempFree1=tempColumnIndices;
			tempValues=newTempValues;
			tempColumnIndices=newTempColumnIndices;
			CUDA_CHECK_RETURN(cudaFree((void*) flTempFree));
			CUDA_CHECK_RETURN(cudaFree((void*) intTempFree1));
			}

	}

	CUDA_CHECK_RETURN(cudaMemcpy(temp,primeValue,sizeof(struct Matrix_CSR), cudaMemcpyDeviceToHost));

	reinitialize_function<<<1,1>>>(primeValue,tempValues,tempColumnIndices,tempRowOffsets,currentSize);

	if(clearFlag)
{
	CUDA_CHECK_RETURN(cudaFree((void*) temp->columnIndices));
	CUDA_CHECK_RETURN(cudaFree((void*) temp->rowOffsets));
	CUDA_CHECK_RETURN(cudaFree((void*) temp->values));
	CUDA_CHECK_RETURN(cudaFree((void*) tempColumn));
	CUDA_CHECK_RETURN(cudaFree((void*) countResult));
}
	return currentSize;

}


int expand(struct Matrix_CSR *d_m,int rowCount,int expansionParameter) {
	struct Matrix_CSR *temp,*variable;
	int binary[8],i=0;
	int currentSize,fastExpQuotient,clearFlag=0;
	temp=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &variable,sizeof(struct Matrix_CSR)));
	CUDA_CHECK_RETURN(cudaMemcpy(temp,d_m,sizeof(struct Matrix_CSR), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(variable,temp,sizeof(struct Matrix_CSR), cudaMemcpyHostToDevice));

	fastExpQuotient=expansionParameter;

	while(fastExpQuotient)
	{
		binary[i++]=fastExpQuotient%2;
		fastExpQuotient=fastExpQuotient/2;
	}

	i=i-2;
	while(i>=0)
	{
		currentSize=matrix_multiplication(d_m,d_m,rowCount,clearFlag);
		clearFlag=1;
		if(binary[i]==1)
		{
			currentSize=matrix_multiplication(d_m,variable,rowCount,clearFlag);
		}
		i--;
	}

	return currentSize;
}



__global__ void converge(struct Matrix_CSR *d_m,int *convergeResult) {
	int i,j;
	float temp_val;

	i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<d_m->rowCount) {
		temp_val=d_m->values[d_m->rowOffsets[i]];
		for(j=d_m->rowOffsets[i];j<d_m->rowOffsets[i+1];j++) {
			if(temp_val!=d_m->values[j]) {
				convergeResult[i]=0;
				return;
			}
		}
		convergeResult[i]=1;
	}
}

Matrix_CSR * MCL(struct Matrix_CSR *m,int inflationParameter,int expansionOperator) {
	int noOfNonZeroValues, *converged;
	struct Matrix_CSR *d_m = NULL;
	struct Matrix_CSR *cluster=NULL;
	int* d_rowOffsets,*d_columnIndices, *convergeResult;
	float *d_values;

	cluster=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_rowOffsets, sizeof(int)*(m->rowCount+1)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_rowOffsets, m->rowOffsets,sizeof(int)*(m->rowCount+1),cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_columnIndices, sizeof(int)*m->nonZeroValueCount));
	CUDA_CHECK_RETURN(cudaMemcpy(d_columnIndices, m->columnIndices,sizeof(int)*m->nonZeroValueCount,cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_values, sizeof(float)*m->nonZeroValueCount));
	CUDA_CHECK_RETURN(cudaMemcpy(d_values, m->values,sizeof(float)*m->nonZeroValueCount,cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &convergeResult, sizeof(int)*m->rowCount));
	converged=(int*)malloc(sizeof(int)*m->rowCount); //????

	Matrix_CSR* temp=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));
	temp->columnIndices=d_columnIndices;
	temp->rowOffsets=d_rowOffsets;
	temp->values=d_values;
	temp->rowCount=m->rowCount;
	temp->nonZeroValueCount=m->nonZeroValueCount;

	noOfNonZeroValues=m->nonZeroValueCount;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_m,sizeof(struct Matrix_CSR)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_m,temp,sizeof(struct Matrix_CSR),cudaMemcpyHostToDevice));
	normalize_row<<<((m->rowCount+512)/512),512>>>(d_m);

	int i,kill=0,flag=0;

	do {
		kill++;
		printf("Iteration:%d   Non zero value count:%d\n",kill,noOfNonZeroValues);
		noOfNonZeroValues=expand(d_m,m->rowCount,expansionOperator);
		inflateRow(d_m,noOfNonZeroValues,m->rowCount,inflationParameter);
		converge<<<((m->rowCount+512)/512),512>>>(d_m,convergeResult);
		CUDA_CHECK_RETURN(cudaMemcpy(converged,convergeResult,sizeof(int)*m->rowCount,cudaMemcpyDeviceToHost));
		for(i=0;i<m->rowCount;i++){
			if(converged[i]==0){
				flag=0;
				break;
			}
			flag=1;
		}
		if(kill>100) {
			printf("Didnt converge....try with other parameters...");
			break;
		}
} while(!flag);

	free(m->values);
	free(m->columnIndices);

	m->values=(float*)malloc(sizeof(float)*noOfNonZeroValues);
	m->columnIndices=(int*)malloc(sizeof(int)*noOfNonZeroValues);


	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaMemcpy(cluster,d_m,sizeof(struct Matrix_CSR), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(m->values,cluster->values,sizeof(float)*noOfNonZeroValues, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(m->columnIndices,cluster->columnIndices,sizeof(int)*noOfNonZeroValues, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(m->rowOffsets,cluster->rowOffsets,sizeof(int)*(m->rowCount+1), cudaMemcpyDeviceToHost));
	m->nonZeroValueCount=noOfNonZeroValues;

	CUDA_CHECK_RETURN(cudaFree((void*) d_m));
	CUDA_CHECK_RETURN(cudaFree((void*) cluster->columnIndices));
	CUDA_CHECK_RETURN(cudaFree((void*) cluster->rowOffsets));
	CUDA_CHECK_RETURN(cudaFree((void*) cluster->values));
	CUDA_CHECK_RETURN(cudaFree((void*) convergeResult));
	return m;
}

void writeCSRToFile(struct Matrix_CSR* m) {
    FILE* fp=fopen("cluster.mcsr","w");
    printf("Cluster wrote to file...\n");
    int i;
    //printf("\n Non Zero Count %d",m->nonZeroValueCount);
    for(i=0;i<m->nonZeroValueCount;++i) {
        fprintf(fp,"%f ",m->values[i]);
    }

    fprintf(fp,"\n");
    for(i=0;i<m->nonZeroValueCount;++i) {
        fprintf(fp,"%d ",m->columnIndices[i]);
    }

    fprintf(fp,"\n");
    for(i=0;i<(m->rowCount)+1;++i) {
        fprintf(fp,"%d ",m->rowOffsets[i]);
    }
}

int convertToInt(char* string) {
	int i,value=0;
	for(i=0;string[i]!='\0';++i) {
		value=value*10+((int)string[i]-48);
	}
	return value;
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */

/*
struct Matrix_CSR* readFromFile(char* file) {
	FILE* fp=fopen(file,"r");
	Matrix_CSR* m=(struct Matrix_CSR*)malloc(sizeof(struct Matrix_CSR));
	printf("\nIn read");
	fscanf(fp,"%d",&m->nonZeroValueCount);
	fscanf(fp,"%d",&m->rowCount);


	m->columnIndices=(int*)malloc(sizeof(int)*m->nonZeroValueCount);
	m->rowOffsets=(int*)malloc(sizeof(int)*(m->rowCount+1));
	m->values=(float*)malloc(sizeof(float)*m->nonZeroValueCount);
	int i;
	for(i=0;i<m->nonZeroValueCount;++i) {
		fscanf(fp,"%f",&m->values[i]);
	}

	for(i=0;i<m->nonZeroValueCount;++i) {
			fscanf(fp,"%d",&m->columnIndices[i]);
		}

	for(i=0;i<=m->rowCount;++i) {
			fscanf(fp,"%d",&m->rowOffsets[i]);
		}
	printf("\nOut read");
	return m;
}
*/

void writeClusterToFile(struct Matrix_CSR* cluster,char NodeIdTable[][20]) {

	struct bucket {
		int rowNumber;
		struct bucket *next;
	};

	struct bucket *array[cluster->rowCount], *temp;
	int i,j;

	for(i=0;i<cluster->rowCount;i++) {
		array[i]=NULL;
	}

	for(i=0;i<cluster->rowCount;i++)
	{
		for(j=cluster->rowOffsets[i];j<cluster->rowOffsets[i+1];j++)
		{
			if(array[cluster->columnIndices[j]]==NULL)
			{
				array[cluster->columnIndices[j]]=(struct bucket*)malloc(sizeof(struct bucket));
				array[cluster->columnIndices[j]]->rowNumber=i;
				array[cluster->columnIndices[j]]->next=NULL;
			}
			else
			{
				temp=(struct bucket*)malloc(sizeof(struct bucket));
				temp->rowNumber=i;
				temp->next=array[cluster->columnIndices[j]];
				array[cluster->columnIndices[j]]=temp;
			}

		}
	}

	for(i=0;i<cluster->rowCount;++i) {
		for(j=i+1;j<cluster->rowCount;++j) {
			struct bucket* node1,*node2;
			for(node1=array[i],node2=array[j];node1!=NULL&&node2!=NULL;node1=node1->next,node2=node2->next) {
				if(node1->rowNumber!=node2->rowNumber) {
					break;
				}
			}
			if(node1==NULL&&node2==NULL) {
				array[j]=NULL;
			}
		}
	}

	FILE* fp=fopen("ClusterSets","w");

	for(i=0;i<cluster->rowCount;++i) {
		struct bucket* node;
		if(array[i]==NULL) {
			continue;
		}
		for(node=array[i];node!=NULL;node=node->next) {
			fprintf(fp,"%s ",NodeIdTable[node->rowNumber]);
			//fprintf(fp,"%d ",node->rowNumber);
		}
		fprintf(fp,"\n");
	}
	printf("Finished Writing Clusters to File...\n");
}

int main(int argc, char** argv) {
	int inflationParameter,expansionOperator;
	struct Matrix_CSR *m,*cluster;
	char NodeIdTable[10000][20];
	m=convertDIPtoMat_CSR(argv[1],NodeIdTable);

	if(argc<4) {
		printf("Too few arguments");
		exit(0);
	}

	inflationParameter=convertToInt(argv[2]);
	expansionOperator=convertToInt(argv[3]);

	cluster=MCL(m,inflationParameter,expansionOperator);
	printf("Clusters created...");
	writeClusterToFile(cluster,NodeIdTable);
	//writeCSRToFile(cluster);
	return 0;
}
