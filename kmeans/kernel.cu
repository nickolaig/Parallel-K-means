#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "Header.h"


cudaError_t groupPointsToClustersCuda(POINT_STR* &points, int points_size, CLUSTER_STR* &clusters, int clusters_size, int nMaxIterations);
void freeDevBuffers(POINT_STR *dev_points, CLUSTER_STR *dev_clusters);
__device__ void assignPointToClusterDevice(POINT_STR p, CLUSTER_STR* clusters, int clusters_size);
__device__ double getDistanceBetweenPoints(POINT_STR p1, POINT_STR p2);

// calculate point distance to each one of the clusters centroid and join it to the closest cluster
__device__ void assignPointToClusterDevice(POINT_STR *p, CLUSTER_STR* clusters, int clusters_size)
{
	int clusterID = clusters[0].id;
	double minDistance = getDistanceBetweenPoints(*p, clusters[0].centroid);


	for (int i = 1; i<clusters_size; i++) {
		double dist = getDistanceBetweenPoints(*p, clusters[i].centroid);
		if (dist<minDistance) {
			minDistance = dist;
			clusterID = clusters[i].id;
		}
	}
	//assign point p to the cluster.
	p->clusterID = clusterID;
}

__device__ double getDistanceBetweenPoints(POINT_STR p1, POINT_STR p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}
// take single point from points array, checks its distance to clusters and move to closest
__global__ void AssignPointsToClosestClusters(POINT_STR *dev_points, CLUSTER_STR *dev_clusters, int clusters_size, int nThreadsInBlock, int startIndexOffset)
{
	int tID = threadIdx.x;
	int bID = blockIdx.x;
	int pointIndex = startIndexOffset + ((bID * nThreadsInBlock) + tID);

	assignPointToClusterDevice(&dev_points[pointIndex], dev_clusters, clusters_size);

}

// kernel function where each thread takes range of points from points array , for each point checks the distance to each cluster and assigns it to closest
__global__ void AssignRangeOfPointsToClosestClusters(POINT_STR *dev_points, int points_size, CLUSTER_STR *dev_clusters, int clusters_size, int pointsRangeForThread, int pointsRangeForBlock)
{
	int tID = threadIdx.x;
	int bID = blockIdx.x;
	int startIndexOffset = bID*pointsRangeForBlock + tID*pointsRangeForThread;

	// check if cudaOccupancyMaxPotentialBlockSize overfeeded our needs
	if (startIndexOffset>points_size - 1) {
		return;
	}
	//move each point to closest cluster
	for (int i = startIndexOffset; i<(startIndexOffset + pointsRangeForThread); i++) {
		assignPointToClusterDevice(&dev_points[i], dev_clusters, clusters_size);
	}

}


// For given array of points and clusters, calculates the distance for each point to each cluster
cudaError_t groupPointsToClustersCuda(POINT_STR* &points, int points_size, CLUSTER_STR* &clusters, int clusters_size, int nMaxIterations)
{
	POINT_STR *dev_points = 0;
	CLUSTER_STR *dev_clusters = 0;

	cudaError_t cudaStatus;
	int numBlocks, nThreadsForBlock, minGridSize;

	//  calculates number of threads for block size that achieves the maximum multiprocessor-level occupancy. Device specs is recieved automatically
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &nThreadsForBlock, AssignRangeOfPointsToClosestClusters, 0, points_size);

	// Round up numBlocks to use kernel function
	numBlocks = (points_size + nThreadsForBlock - 1) / nThreadsForBlock;


	// each thread will make calculation to range of points from points array
	// calculate the length of range which each thread should work.
	int pointsRangeForThread;
	if (numBlocks*nThreadsForBlock>points_size) {
		pointsRangeForThread = 1;
	}
	else {
		pointsRangeForThread = points_size / (numBlocks*nThreadsForBlock);
	}


	// calculate the total range size which each block will work on 
	int pointsRangeForBlock = pointsRangeForThread*nThreadsForBlock;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n%s", cudaGetErrorString(cudaStatus));
		freeDevBuffers(dev_points, dev_clusters);
	}

	// Allocate GPU buffers for points and clusters array
	cudaStatus = cudaMalloc((void**)&dev_points, points_size * sizeof(POINT_STR));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed! Allocate GPU buffer for points array \n%s", cudaGetErrorString(cudaStatus));
		freeDevBuffers(dev_points, dev_clusters);
	}


	cudaStatus = cudaMalloc((void**)&dev_clusters, clusters_size * sizeof(CLUSTER_STR));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed! Allocate GPU buffer for clusters array \n%s", cudaGetErrorString(cudaStatus));
		freeDevBuffers(dev_points, dev_clusters);
	}


	// Copy points and clusters array to alocated GPU buffers
	cudaStatus = cudaMemcpy(dev_points, points, points_size * sizeof(POINT_STR), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed! Copy points alocated GPU buffers\n%s", cudaGetErrorString(cudaStatus));
		freeDevBuffers(dev_points, dev_clusters);
	}



	for (int i = 0; i<nMaxIterations; i++) {

		cudaStatus = cudaMemcpy(dev_clusters, clusters, clusters_size * sizeof(CLUSTER_STR), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed! Copy points alocated GPU buffers\n%s", cudaGetErrorString(cudaStatus));
			freeDevBuffers(dev_points, dev_clusters);
		}

		//run kernel function which will asign each point to closest clusters
		AssignRangeOfPointsToClosestClusters << <numBlocks, nThreadsForBlock >> >(dev_points, points_size, dev_clusters, clusters_size, pointsRangeForThread, pointsRangeForBlock);

		// wait for the kernel to finish.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize failed! AssignRangeOfPointsToClosestClusters\n%s", cudaGetErrorString(cudaStatus));
			freeDevBuffers(dev_points, dev_clusters);
		}

		// special case where not all points got assign to clusters due to reminder

		if (points_size % pointsRangeForThread != 0) {
			printf("reminder case\n");
			int nRemindPoints = points_size % pointsRangeForThread;
			int startIndexOffset = points_size - nRemindPoints;

			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &nThreadsForBlock, AssignPointsToClosestClusters, 0, nRemindPoints);

			numBlocks = (nRemindPoints + nThreadsForBlock - 1) / nThreadsForBlock;

			AssignPointsToClosestClusters << <numBlocks, nThreadsForBlock >> >(dev_points, dev_clusters, clusters_size, nThreadsForBlock, startIndexOffset);

			// wait for the kernel to finish.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				printf("cudaDeviceSynchronize failed! AssignRangeOfPointsToClosestClusters\n%s", cudaGetErrorString(cudaStatus));
				freeDevBuffers(dev_points, dev_clusters);
			}
		}


		// Copy results of sorted points per clusters
		cudaStatus = cudaMemcpy(points, dev_points, points_size * sizeof(POINT_STR), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("Copy results of found clusters from device to host failed!\n%s", cudaGetErrorString(cudaStatus));
			freeDevBuffers(dev_points, dev_clusters);
		}


		recalculateClusterCentroids(clusters, clusters_size, points, points_size);

		// stop K Means when all clusters centeroids stays the same
		if (!isClustersCentroidsHasChanged(clusters, clusters_size)) {
			break;
		}

	}


	freeDevBuffers(dev_points, dev_clusters);

	return cudaStatus;
}

void freeDevBuffers(POINT_STR *dev_points, CLUSTER_STR *dev_clusters) {

	cudaFree(dev_points);
	cudaFree(dev_clusters);
}

