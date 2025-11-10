#include <stdio.h>
#include <stdint.h>
#include "unif01.h"
#include "bbattery.h"

static FILE *fp;

unsigned int Read32FromFile(void) {
	uint32_t x;
	size_t n = fread(&x, sizeof(uint32_t), 1, fp);
	if (n != 1) {
		// Restart or terminate when EOF
		rewind(fp);
		fread(&x, sizeof(uint32_t), 1, fp);
	}
	return x;
}

int main(void) {
	fp = fopen("noverlap_seed1931571603.bin", "rb");
	if (!fp) {
		perror("Error opening file");
		return 1;
	}

	unif01_Gen *gen = unif01_CreateExternGenBits("LCG_LH_TEST", Read32FromFile);
	bbattery_SmallCrush(gen);
	unif01_DeleteExternGenBits(gen);

	fclose(fp);
	return 0;
}

