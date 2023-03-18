#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <divsufsort.h>

int main() {
	// input data

	clock_t start, stop;
	start = clock();
	FILE *fp;
	fp=fopen("pattern_small.txt", "r");

	//clock_t t;
	//t = clock();

	char Text[256];

	fscanf(fp, "%[^\n]", Text);
	//printf("Data from the file:\n%s\n", Text);

	fclose(fp);

	int n = strlen(Text);
	int i;

	// allocate
	int *SA = (int *)malloc(n * sizeof(int));
	
	// sort
	divsufsort((unsigned char *)Text, SA, n);
	
 	FILE *fp2;
	fp2=fopen("out_small.txt", "w");

	// output
	for(i = 0; i < n; ++i) {
		//printf("SA[%2d] = %2d: ", i, SA[i]);
		fprintf(fp2, "%i\n", SA[i]);
		//for(j = SA[i]; j < n; ++j) {
		//	printf("%c", Text[j]);
		//}
		//printf("$\n");
	}

	stop = clock();
	double time_taken;
	time_taken = (double)(stop-start)/(double)CLOCKS_PER_SEC;
	fprintf(stdout, "Completed in %f sec\n", time_taken);
	// deallocate
	free(SA);

	return 0;

}	
