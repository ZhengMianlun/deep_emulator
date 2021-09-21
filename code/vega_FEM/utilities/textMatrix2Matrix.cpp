#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "matrixIO.h"


int main(int argc, char ** argv)

{
	
  if ( argc != 6 || (argv[1])[0] == '-')
  { 
	
	  
    printf("Converts an ascii text file into the binary matrix format.\n");
    printf("The input file is a text matrix of numbers.\n");
    printf("The number of lines/columns of the text file must equal the product of parameters 'number of rows' and 'number of columns'.\n");
    printf("All the input numbers will be multiplied with scaling factor before writing to the output matrix.\n");
    printf("Set scaling factor to 1 to preserve the original values.\n");
    printf("Usage: %s <input file> <output file> <number of rows> <number of columns> <scaling factor>\n",argv[0]);
    return -1;
  }
  
  // open input
  FILE * file;
  file = fopen(argv[1],"r"); 
  if (!file)
  {
    printf("Error: couldn't open input file %s.\n",argv[1]);
    return -1;
  }

  int m,n;
  // parse m
  m = strtol(argv[3],NULL,10);
  if (m <= 0)
  {
    printf("Error: bad parameter for the number of rows: %s.\n",argv[3]);
    return -1;
  }

  // parse n
  n = strtol(argv[4],NULL,10);
  if (n <= 0)
  {
    printf("Error: bad parameter for the number of columns: %s.\n",argv[4]);
    return -1;
  }

  // parse scaling factor
  double scaling;
  scaling = strtod(argv[5],NULL);
  if (fabs(scaling) > 1E8)
    printf("Warning: scaling factor very large: %f .\n",scaling);

  // allocate space
  double * matrix;
  matrix = (double *)malloc(sizeof(double)*m*n);
  double data;

  int i=0;

  int bufSize = n * 30;
  char * buf = (char*) malloc (sizeof(char) * bufSize);
  while(fgets(buf,bufSize,file) != NULL)
  {
    if (i % 100 == 1)
    {
      printf("%d ",i);
      fflush(NULL);
    } 

    if (i == m)
    {
      break;
      printf("Error: the number of data rows in file %s is greater than the declared dimension %d x %d .\n",argv[1],m,n);
      return -1;
    }

    // remove multiple white space characters from line
    char * w = buf;
    while (*w != '\0')
    {
      while ((*w == ' ') && (*(w+1) == ' ')) // erase second blank
      {
        char * u = w+1;
        while (*u != '\0') // shift everything left one char
        {
          *u = *(u+1);
          u++;
        }
      }
      w++;
    }

    // remove whitespace at beginning of line
    if (*buf == ' ')
    {
      char * u = buf;
      while (*u != '\0') // shift everything left one char
      {
        *u = *(u+1);
        u++;
      }
    }

    //printf("%s\n",buf);

    char * token = strtok(buf," ");

    for(int j=0; j<n; j++)
    {
      if (token == NULL)
      {
        printf("Token is NULL. Error parsing line %d.\n",i+1);
        return  1 ;
      }

      if (sscanf(token,"%lf",&data) < 1)
      {
        printf("Cannot parse data from current token. Error parsing line %d.\n",i);
        return  1 ;
      }

      //printf("%G\n",data);

      matrix[ELT(m,i,j)] = data * scaling;

      token = strtok(NULL," ");

    }
    i++;
  }

  fclose(file);

  free(buf);

  if (i < m) // not enough data has been read
  {
    printf("Error: the number of data rows in file %s is smaller than the declared dimension %d x %d .\n",argv[1],m,n);
    return -1;
  }

  // now, matrix contains the input data
  
  // write the data out to the file
  if (WriteMatrixToDisk(argv[2], m, n, matrix) != 0)
  {
    printf("Error writing to the output file %s.\n",argv[2]);
    return -1;
  }

  printf("\n");

  return 0;

   
}
