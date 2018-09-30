#pragma once
#include <mpi.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct TIME_INFOS {
	double tStart;
	double tEnd;
	double deltaT;
	double timeInterval;
	int nMaxIteration;
	int nClusters;
	int nCoordinates;
	double QM;

} timeInfo;

typedef struct POINT_STR {
	double x;
	double y;
	int clusterID;
} point;

typedef struct COORDINATE_STR {
	double X;
	double Y;
	double VX;
	double VY;
} coordinate_str;

typedef struct CLUSTER_STR {
	int id;
	int points_size;
	POINT_STR centroid;
	POINT_STR oldCentroid;
} cluster;


typedef struct STATE_SNAPSHOT_STR {
	double time;
	double qm;
	CLUSTER_STR* clusters;
} state_snapshot;



//MPI OpenMP
void createTimeInformations(TIME_INFOS*, double, int, double, int, int, int, double);
void K_Means(CLUSTER_STR* & clusters, int, POINT_STR*, int, int);
void createPoints(POINT_STR* points, COORDINATE_STR* coordinates, int coordinates_size, double);
void readFile(char * fullFilePath, COORDINATE_STR* &coordinates, int *nCoordinates, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations, double *qm);
double getDistanceBetweenCoordinates(POINT_STR p1, POINT_STR p2);
void initClusters(CLUSTER_STR* &clusters, int nClusters, POINT_STR* points);
void writeOutFile(char* fileFullPath, STATE_SNAPSHOT_STR goal_time_snapshot, int nClusters);
void recalculateClusterCentroids(CLUSTER_STR* &clusters, int clusters_size, POINT_STR* points, int points_size);
bool isClustersCentroidsHasChanged(CLUSTER_STR* &clusters, int clusters_size);
double findDiameter(POINT_STR* , int );
double calculateQM(TIME_INFOS, POINT_STR* , CLUSTER_STR*);
MPI_Datatype createMPICoordinateType();
MPI_Datatype createMPIPointType();
MPI_Datatype createMPISnapShotType();
MPI_Datatype createMPIRangeType();
MPI_Datatype createMPIClusterType(MPI_Datatype);

//CUDA
cudaError_t groupPointsToClustersCuda(POINT_STR* &points, int points_size, CLUSTER_STR* &clusters, int clusters_size, int nMaxIterations);
void freeDevBuffers(POINT_STR *dev_points, CLUSTER_STR *dev_clusters);
__device__ void assignPointToClusterDevice(POINT_STR p, CLUSTER_STR* clusters, int clusters_size);
__device__ double getDistanceBetweenPoints(POINT_STR p1, POINT_STR p2);

