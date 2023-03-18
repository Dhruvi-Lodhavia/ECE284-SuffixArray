#include <zlib.h>
#include <stdio.h>
#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

int main(int argc, char *argv[])
{
	FILE *fp1, *fp2;
	char num1[100];
	char num2[100];
	int n1, n2;
	int flag = 0;
	fp1 = fopen(argv[1], "r");
	fp2 = fopen(argv[2], "r");
	int i = 0;
	while (fgets(num1, sizeof(num1), fp1) && fgets(num2, sizeof(num2), fp2)) {
		n1 = atoi(num1);
		n2 = atoi(num2);
		//printf("Num1 = %i\n", n1);
		//printf("Num2 = %i\n", n2);
		if(n1 != n2) {
			flag = 1;
			printf("Error at Position: %i\n", i);
			break;
		}
		i=i+1;
	}	
	
	if(flag)
		printf("FAIL: Sequential vs Parallel implementation functionality NOT MATCHING!\n");
	else
		printf("PASS: Sequential vs Parallel implementation functionality MATCHING!\n");


	fclose(fp1);
	fclose(fp2);
	return 0;
}
