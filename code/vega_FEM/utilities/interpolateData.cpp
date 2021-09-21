#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "float.h"
#include "volumetricMesh.h"
#include "cubicMesh.h"
#include "tetMesh.h"
#include "objMesh.h"
#include "matrixIO.h"
#include "matrixMacros.h"
#include "getopts.h"

int main(int argc, char ** argv)
{
  if (argc < 5)
  {
    printf("Interpolates given volumetric mesh datamatrix to vertices of the given mesh.\n");
    printf("Usage: %s <volumetric mesh file> <target obj file> <data matrix> <output matrix> [-i interpolant file] [-z threshold]\n",argv[0]);
    printf("-i : use the specified interpolant (default: build interpolant)\n");
    printf("-z : assign zero mode to vertices too far away from the volumetric mesh\n");
    return 0;
  }

  char * meshFile = argv[1];
  char * objMeshname = argv[2];
  char * dataMatrixFilename = argv[3];
  char * outputMatrixFilename = argv[4];

  char zeroThresholdString[4096] = "__none";
  char interpolantFile[4096] = "__none";
  opt_t opttable[] =
  {
    { "i", OPTSTR, &interpolantFile },
    { "z", OPTSTR, &zeroThresholdString },
    { NULL, 0, NULL }
  };

  argv += 4;
  argc -= 4;
  int optup = getopts(argc,argv,opttable);
  if (optup != argc)
  {
    printf("Error parsing options. Error at option %s.\n",argv[optup]);
    return 1;
  }

  double threshold;
  if (strcmp(zeroThresholdString,"__none") == 0)
    threshold = -1;
  else
    threshold = strtod(zeroThresholdString, NULL);

  VolumetricMesh * volumetricMesh = NULL;
  CubicMesh * cubicMesh = NULL;
  TetMesh * tetMesh = NULL;
  VolumetricMesh::elementType eType = volumetricMesh->getElementType(meshFile);

  if (eType == VolumetricMesh::CUBIC)
  {
    printf("Loading cubic mesh...\n");
    cubicMesh = new CubicMesh(meshFile);
    volumetricMesh = cubicMesh;
  }

  if (eType == VolumetricMesh::TET)
  {
    printf("Loading tet mesh...\n");
    tetMesh = new TetMesh(meshFile);
    volumetricMesh = tetMesh;
  }

  if (volumetricMesh == NULL)
  {
    printf("Error: unknown volumetric mesh type encountered.\n");
    exit(1);
  }
 
  int n = volumetricMesh->getNumVertices();
  int nel = volumetricMesh->getNumElements();
  printf("Info on %s:\n", meshFile);
  printf("Num vertices: %d\n", n);
  printf("Num elements: %d\n", nel);
  //double voxelSpacing = volumetricMesh->getCubeSize();
  //printf("Voxel spacing: %.15f\n", voxelSpacing);

  ObjMesh * objMesh = new ObjMesh(objMeshname);

  double * vertexData;
  int mData, r;
  ReadMatrixFromDisk_(dataMatrixFilename, &mData, &r, &vertexData);

  if (mData != 3*n)
  {
    printf("Error: datamatrix has %d rows, but 3*#vertices=%d\n", mData, 3*n);
    exit(1);
  }

  int numInterpolationLocations = objMesh->getNumVertices();
  double * interpolationLocations = (double*) malloc (sizeof(double) * 3 * numInterpolationLocations);
  for(int i=0; i< numInterpolationLocations; i++)
  {
    Vec3d pos = objMesh->getPosition(i);
    interpolationLocations[3*i+0] = pos[0];
    interpolationLocations[3*i+1] = pos[1];
    interpolationLocations[3*i+2] = pos[2];
  }

  double * destMatrix = (double*) malloc (sizeof(double) * 3 * numInterpolationLocations * r);

  int * vertices;
  double * weights;

  if (strcmp(interpolantFile, "__none") == 0)
  {
    printf("Building interpolation weights...\n");
    int numExternalVertices = volumetricMesh->generateInterpolationWeights(numInterpolationLocations, interpolationLocations, &vertices, &weights, threshold);
    printf("Encountered %d vertices not belonging to any voxel.\n", numExternalVertices);
  }
  else
  {
    printf("Loading interpolation weights from %s...\n", interpolantFile);
    int code = volumetricMesh->loadInterpolationWeights(interpolantFile, numInterpolationLocations, volumetricMesh->getNumElementVertices(), &vertices, &weights);
    if (code != 0)
    {
      printf("Error loading interpolation weights.\n");
      exit(1);
    }    
  }

  printf("Interpolating...\n");
  for(int i=0; i<r; i++)
    VolumetricMesh::interpolate(&vertexData[ELT(3*n,0,i)], &destMatrix[ELT(3 * numInterpolationLocations,0,i)], numInterpolationLocations, volumetricMesh->getNumElementVertices(), vertices, weights);

  //int numExternalVertices = volumetricMesh->interpolate(vertexData, numInterpolationLocations, r, interpolationLocations, destMatrix, threshold);

  WriteMatrixToDisk_(outputMatrixFilename, 3 * numInterpolationLocations, r, destMatrix);

  return 0;
}
