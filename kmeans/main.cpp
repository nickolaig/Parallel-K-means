#include <stdlib.h>
#include <stdio.h>
#include "Header.h"
#include "math.h"
#include <string.h>
#include <omp.h>
#include <mpi.h>

char*   inputFilePath = "D:\\input.txt";
char*   outputFilePath = "D:\\output.txt";

#define MASTER 0

int main(int argc, char *argv[])
{
	MPI_Status status;
	int num_procs, processID;
	double startTime, finishTime;
	//-------------------------MPI---------------------------
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processID);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	// Create struct types in MPI system.
	MPI_Datatype CoordinateMPIType = createMPICoordinateType();
	MPI_Datatype PointMPIType = createMPIPointType();
	MPI_Datatype RangeMPIType = createMPIRangeType();
	MPI_Datatype ClusterMPIType = createMPIClusterType(PointMPIType);
	MPI_Datatype SnapShotMPIType = createMPISnapShotType();
	//-----------------------------------ENDMPIINIT--------------------------------

	double deltaT = 0, timeIntervalT = 0, QM = 0;
	int numCoordinates = 0, numClusters = 0, numMaxIterations = 0;
	bool isMasterFoundGoal = false;
	bool flag = false;
	double global_min_time;


	TIME_INFOS slaveTi;
	COORDINATE_STR* coordinates = NULL;
	TIME_INFOS* time_informations = (TIME_INFOS*)malloc((num_procs) * sizeof(TIME_INFOS));


	//Init Master
	if (processID == MASTER) {
		readFile(inputFilePath, coordinates, &numCoordinates, &numClusters, &deltaT, &timeIntervalT, &numMaxIterations, &QM);
		// Split time info for each slave.
		createTimeInformations(time_informations, timeIntervalT, num_procs, deltaT, numMaxIterations, numCoordinates, numClusters, QM);
	}

	//work start time
	startTime = MPI_Wtime();

	// send to each slave his time information struct
	MPI_Scatter(time_informations, 1, RangeMPIType, &slaveTi, 1, RangeMPIType, MASTER, MPI_COMM_WORLD);
	// send to each slave coordinates from input file
	if (processID != MASTER) { coordinates = (COORDINATE_STR*)malloc(slaveTi.nCoordinates * sizeof(COORDINATE_STR)); }
	MPI_Bcast(coordinates, slaveTi.nCoordinates, CoordinateMPIType, MASTER, MPI_COMM_WORLD);

	//Init Master Finish

	//Slave work start

	int number_snapshots = (slaveTi.tEnd - slaveTi.tStart) / slaveTi.deltaT;
	STATE_SNAPSHOT_STR goalSnapshot;
	goalSnapshot.clusters = (CLUSTER_STR*)malloc(slaveTi.nClusters * sizeof(CLUSTER_STR));
	goalSnapshot.time = INT_MAX;
#pragma omp parallel for
	for (int tIterarionIndex = 0; tIterarionIndex < number_snapshots; tIterarionIndex++) {
		//current t for check	
		double current_t = slaveTi.tStart + tIterarionIndex*slaveTi.deltaT;

		POINT_STR* points_slave = (POINT_STR*)malloc(slaveTi.nCoordinates * sizeof(POINT_STR));
		createPoints(points_slave, coordinates, slaveTi.nCoordinates, current_t);

		//init clusters
		CLUSTER_STR* clusters = NULL;
		initClusters(clusters, slaveTi.nClusters, points_slave);

		double singleTstartTime;
		if (processID == MASTER && tIterarionIndex == 0) {
			singleTstartTime = MPI_Wtime();
		}

		// perform KMean with limit of numMaxIterations using CUDA.
		K_Means(clusters, slaveTi.nClusters, points_slave, slaveTi.nCoordinates, slaveTi.nMaxIteration);


		if (processID == MASTER && tIterarionIndex == 0) {
			printf("first thread finished first iter : %lf\n", MPI_Wtime() - singleTstartTime);
		}
		double calcQm = calculateQM(slaveTi, points_slave, clusters);

		if (calcQm <= slaveTi.QM)//&&!flag
		{
			goalSnapshot.qm = calcQm;
			goalSnapshot.time = current_t;
			memcpy(goalSnapshot.clusters, clusters, slaveTi.nClusters * sizeof(CLUSTER_STR));
			free(points_slave);
			free(clusters);
			break;
			//flag = true;
		}
		free(points_slave);
		free(clusters);
	}

	// reducing time minimum found
	MPI_Allreduce(&goalSnapshot.time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);


	if (global_min_time == goalSnapshot.time) {

		// if minimum time found by non master process , send best result to master
		if (processID != MASTER) {
			MPI_Send(&goalSnapshot, 1, SnapShotMPIType, MASTER, 0, MPI_COMM_WORLD);
			MPI_Send(goalSnapshot.clusters, slaveTi.nClusters * sizeof(CLUSTER_STR), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);
		}
		else {
			isMasterFoundGoal = true;
		}

	}
	//SLAVE OVER


	//MASTER
	if (processID == MASTER) {
		STATE_SNAPSHOT_STR final_snapshot;
		final_snapshot.clusters = (CLUSTER_STR*)malloc(numClusters * sizeof(CLUSTER_STR));

		if (!isMasterFoundGoal) {
			MPI_Recv(&final_snapshot, 1, SnapShotMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(final_snapshot.clusters, slaveTi.nClusters * sizeof(CLUSTER_STR), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}
		else {
			final_snapshot = goalSnapshot;
		}

		finishTime = MPI_Wtime();
		printf("Total Time Taken: %lf\n", finishTime - startTime);
		writeOutFile(outputFilePath, final_snapshot, numClusters);

	}

	//MASTER FINISH

	//Free some memory
	free(goalSnapshot.clusters);

	MPI_Finalize();

	return 0;
}


// Creates Time information array 
void createTimeInformations(TIME_INFOS* time_informations, double timeIntervalT, int nProcesses, double deltaT, int nMaxIterations, int nCoordinates, int nClusters, double qm)
{

	int totalNumberOfSystemSnapShots = timeIntervalT / deltaT;
	int nSystemSnapShootsForProcess = totalNumberOfSystemSnapShots / nProcesses;
	int reminder = totalNumberOfSystemSnapShots % nProcesses;
	double tRangeForProcess = nSystemSnapShootsForProcess*deltaT;


	int nSplites = nProcesses;

	if (reminder != 0) {
		nSplites -= 1;
	}

	double startT = 0;

	for (int i = 0; i < nSplites; i++) {
		time_informations[i].tStart = startT;
		time_informations[i].tEnd = startT + tRangeForProcess;
		time_informations[i].deltaT = deltaT;
		time_informations[i].timeInterval = timeIntervalT;
		time_informations[i].nMaxIteration = nMaxIterations;
		time_informations[i].nCoordinates = nCoordinates;
		time_informations[i].nClusters = nClusters;
		time_informations[i].QM = qm;

		startT += tRangeForProcess;
	}

	if (reminder != 0) {
		time_informations[nSplites].tStart = startT;
		time_informations[nSplites].tEnd = startT + (reminder + nSystemSnapShootsForProcess)*deltaT;
		time_informations[nSplites].deltaT = deltaT;
		time_informations[nSplites].timeInterval = timeIntervalT;
		time_informations[nSplites].nMaxIteration = nMaxIterations;
		time_informations[nSplites].nCoordinates = nCoordinates;
		time_informations[nSplites].nClusters = nClusters;
		time_informations[nSplites].QM = qm;
	}


}
MPI_Datatype createMPIPointType()
{
	MPI_Datatype PointType;
	POINT_STR point;
	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklen[3] = { 1,1,1 };
	MPI_Aint disp[3];
	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.clusterID - (char *)&point;
	MPI_Type_create_struct(3, blocklen, disp, type, &PointType);
	MPI_Type_commit(&PointType);
	return PointType;
}

MPI_Datatype createMPISnapShotType()
{
	MPI_Datatype SnapShotMPIType;
	STATE_SNAPSHOT_STR snapShot;
	MPI_Datatype type[2] = { MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[2] = { 1,1 };
	MPI_Aint disp[2];
	disp[0] = (char *)&snapShot.qm - (char *)&snapShot;
	disp[1] = (char *)&snapShot.time - (char *)&snapShot;
	//disp[2] = (char *) &snapShoot.clusters -	 (char *) &snapShoot;
	MPI_Type_create_struct(2, blocklen, disp, type, &SnapShotMPIType);
	MPI_Type_commit(&SnapShotMPIType);
	return SnapShotMPIType;
}
MPI_Datatype createMPIClusterType(MPI_Datatype PointMPIType)
{
	MPI_Datatype ClusterMPIType;
	CLUSTER_STR cluster;
	MPI_Datatype type[4] = { MPI_INT, MPI_INT,PointMPIType,PointMPIType };
	int blocklen[4] = { 1,1,2,2 };
	MPI_Aint disp[4];
	disp[0] = (char *)&cluster.id - (char *)&cluster;
	disp[1] = (char *)&cluster.points_size - (char *)&cluster;
	disp[2] = (char *)&cluster.centroid - (char *)&cluster;
	disp[3] = (char *)&cluster.oldCentroid - (char *)&cluster;
	MPI_Type_create_struct(4, blocklen, disp, type, &ClusterMPIType);
	MPI_Type_commit(&ClusterMPIType);
	return ClusterMPIType;
}

MPI_Datatype createMPIRangeType()
{
	MPI_Datatype RangeMPIType;
	TIME_INFOS range;
	MPI_Datatype type[8] = { MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT,MPI_INT,MPI_INT,MPI_DOUBLE };
	int blocklen[8] = { 1,1,1,1,1,1,1,1 };
	MPI_Aint disp[8];
	disp[0] = (char *)&range.tStart - (char *)&range;
	disp[1] = (char *)&range.tEnd - (char *)&range;
	disp[2] = (char *)&range.deltaT - (char *)&range;
	disp[3] = (char *)&range.timeInterval - (char *)&range;
	disp[4] = (char *)&range.nMaxIteration - (char *)&range;
	disp[5] = (char *)&range.nClusters - (char *)&range;
	disp[6] = (char *)&range.nCoordinates - (char *)&range;
	disp[7] = (char *)&range.QM - (char *)&range;

	MPI_Type_create_struct(8, blocklen, disp, type, &RangeMPIType);
	MPI_Type_commit(&RangeMPIType);
	return RangeMPIType;
}

// each K-Means iteration recalculate new clusters centroids.
void recalculateClusterCentroids(CLUSTER_STR* &clusters, int clusters_size, POINT_STR* points, int points_size)
{
	double* arr = (double*)calloc(clusters_size * 3, sizeof(double));

	for (int i = 0; i < points_size; i++) {
		arr[points[i].clusterID * 3] += points[i].x;
		arr[points[i].clusterID * 3 + 1] += points[i].y;
		arr[points[i].clusterID * 3 + 2] += 1;
	}

#pragma omp parallel for
	for (int i = 0; i < clusters_size; i++) {
		if (arr[i * 3 + 2] != 0) {

			double newCentroidX = arr[i * 3] / arr[i * 3 + 2];
			double newCentroidY = arr[i * 3 + 1] / arr[i * 3 + 2];

			clusters[i].centroid.x = newCentroidX;
			clusters[i].centroid.y = newCentroidY;
		}
	}

	free(arr);

}

// check if centroids changed
bool isClustersCentroidsHasChanged(CLUSTER_STR* &clusters, int clusters_size)
{
	bool isClustersHasChanged = false;

#pragma omp parallel for
	for (int i = 0; i < clusters_size; i++) {

		double newCentroidX = clusters[i].centroid.x;
		double newCentroidY = clusters[i].centroid.y;

		if (newCentroidX != clusters[i].oldCentroid.x || newCentroidY != clusters[i].oldCentroid.y) {
			isClustersHasChanged = true;
		}
		clusters[i].oldCentroid.x = newCentroidX;
		clusters[i].oldCentroid.y = newCentroidY;
	}
	return isClustersHasChanged;

}

MPI_Datatype createMPICoordinateType()
{
	MPI_Datatype CoordinateType;
	COORDINATE_STR coord;
	MPI_Datatype type[4] = { MPI_DOUBLE, MPI_DOUBLE ,MPI_DOUBLE,MPI_DOUBLE };
	int blocklen[4] = { 1, 1, 1 ,1 };
	MPI_Aint disp[4];
	disp[0] = (char *)&coord.X - (char *)&coord;
	disp[1] = (char *)&coord.Y - (char *)&coord;
	disp[2] = (char *)&coord.VX - (char *)&coord;
	disp[3] = (char *)&coord.VY - (char *)&coord;
	MPI_Type_create_struct(4, blocklen, disp, type, &CoordinateType);
	MPI_Type_commit(&CoordinateType);
	return CoordinateType;
}

void writeOutFile(char* fileFullPath, STATE_SNAPSHOT_STR goal_state_snapshot, int nClusters)
{
	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error writing file!\n");
		exit(1);
	}
	if (goal_state_snapshot.time == INT_MAX)
		fprintf(f, "We didn't encounter goal qm in provided limits try another settings.\n");
	else {
		fprintf(f, "Goal QM : %lf\n", goal_state_snapshot.qm);
		fprintf(f, "Occurred at time: %lf\n", goal_state_snapshot.time);
		fprintf(f, "Centers of the clusters:\n");

		for (int i = 0; i < nClusters; i++) {
			fprintf(f, "(%lf, %lf)\n", goal_state_snapshot.clusters[i].centroid.x, goal_state_snapshot.clusters[i].centroid.y);
		}
	}
	fclose(f);
}

// For given array of points performs K-Means algorithm and returns found clusters centroids.
void K_Means(CLUSTER_STR* & clusters, int nClusters, POINT_STR *points, int nPoints, int nMaxIterations)
{
	groupPointsToClustersCuda(points, nPoints, clusters, nClusters, nMaxIterations);
}

// Initiate empty clusters 
void initClusters(CLUSTER_STR* &clusters, int nClusters, POINT_STR* points)
{
	clusters = (CLUSTER_STR*)malloc(nClusters * sizeof(CLUSTER_STR));

	for (int i = 0; i < nClusters; i++) {
		clusters[i].centroid.x = points[i].x;
		clusters[i].centroid.y = points[i].y;
		clusters[i].points_size = 0;
		clusters[i].id = i;
	}
}
// Create points via the formula
void createPoints(POINT_STR* points, COORDINATE_STR* coordinates, int coord_size, double current_t)
{
#pragma omp parallel for
	for (int i = 0; i < coord_size; i++) {
		points[i].x = coordinates[i].X + coordinates[i].VX*current_t;
		points[i].y = coordinates[i].Y + coordinates[i].VY*current_t;
	}
}

double getDistanceBetweenCoordinates(POINT_STR p1, POINT_STR p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}

// Reads input file
void readFile(char * fullFilePath, COORDINATE_STR* &coordinates, int *nCoordinates, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations, double *qm)
{
	FILE *f;

	errno_t errorCode = fopen_s(&f, fullFilePath, "r");

	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	int row = fscanf_s(f, "%d %d %lf %lf %d %lf", nCoordinates, nClusters, timeIntervalT, deltaT, nMaxIterations, qm);
	coordinates = (COORDINATE_STR*)malloc(*nCoordinates * sizeof(COORDINATE_STR));

	int pointId = 0;
	double X, Y, VX, VY;
	while (row != EOF&&pointId < (*nCoordinates)) {
		row = fscanf_s(f, "%lf %lf %lf %lf\n", &X, &Y, &VX, &VY);
		coordinates[pointId].X = X;
		coordinates[pointId].Y = Y;
		coordinates[pointId].VX = VX;
		coordinates[pointId].VY = VY;
		pointId++;
		//printf("%lf %lf %lf %lf\n", X, Y, VX, VY);
	}
	fclose(f);
}
// dirty solution better results can be achieved on cuda for each point find its diameter and then we can find diameter of whole cluster 
double calculateQM(TIME_INFOS ti, POINT_STR* points, CLUSTER_STR* clusters)
{
	double qM = 0;
	int* sizes = (int*)calloc(ti.nClusters, sizeof(int));
	for (int i = 0; i < ti.nCoordinates; i++)
	{
		sizes[points[i].clusterID]++;
	}
	POINT_STR** pt = (POINT_STR**)malloc(sizeof(POINT_STR*)*ti.nClusters);

#pragma omp parralel for
	for (int i = 0; i < ti.nClusters; i++)
	{
		pt[i] = (POINT_STR*)malloc(sizes[i] * sizeof(POINT_STR));
	}
	for (int i = 0; i < ti.nCoordinates; i++)
	{
		pt[points[i].clusterID][sizes[points[i].clusterID] - 1] = points[i];
		sizes[points[i].clusterID]--;
	}
	for (int i = 0; i < ti.nCoordinates; i++)
	{
		sizes[points[i].clusterID]++;
	}

#pragma omp parralel for
	for (int i = 0; i < ti.nClusters; i++)
	{
		sizes[i] = findDiameter(pt[i], sizes[i]);
	}

	for (int i = 0; i < ti.nClusters; i++)
	{
		for (int j = 0; j < ti.nClusters; j++)
		{
			if (i != j)
				qM += (sizes[i] / (getDistanceBetweenCoordinates(clusters[j].centroid, clusters[i].centroid)));
		}
	}
	qM = qM / (ti.nClusters*(ti.nClusters - 1));
	//printf("QM: %lf\n", qM);
	//free(sizes);
	//free(pt);
	return qM;
}

double findDiameter(POINT_STR* pt, int size) {
	double diameter = 0;
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			if (diameter < getDistanceBetweenCoordinates(pt[i], pt[j]))
				diameter = getDistanceBetweenCoordinates(pt[i], pt[j]);
		}
	}
	return diameter;
}
