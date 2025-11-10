#include "unif01.h"
#include "bbattery.h"
#include <stdio.h>

// TestU01 expects a function taking no arguments for bit generators.
static unsigned int read32(void) {
	unsigned int x;

	// Try to read 4 bytes from stdin (binary)
	if (fread(&x, sizeof(x), 1, stdin) != 1) {
		// Reached EOF: try to rewind (optional)
		clearerr(stdin);
		rewind(stdin);
		fread(&x, sizeof(x), 1, stdin);
	}

	return x;
}

int main(void) {
	setvbuf(stdin, NULL, _IONBF, 0);   // disable buffering on stdin
	setvbuf(stdout, NULL, _IONBF, 0);  // disable buffering on stdout

	unif01_Gen *gen = unif01_CreateExternGenBits("pipe_gen", read32);

	bbattery_Crush(gen);   // or bbattery_Crush / bbattery_BigCrush

	unif01_DeleteExternGenBits(gen);
	return 0;
}

