#include "unif01.h"
#include "bbattery.h"
#include <stdio.h>

static uint32_t buffer[8192];
static size_t buf_pos = 0;
static size_t buf_len = 0;

static unsigned int read32(void) {
    if (buf_pos >= buf_len) {
        buf_len = fread(buffer, sizeof(uint32_t), 8192, stdin);
        buf_pos = 0;

        if (buf_len == 0) {
            clearerr(stdin);
            rewind(stdin);
            buf_len = fread(buffer, sizeof(uint32_t), 8192, stdin);
        }
    }
    return buffer[buf_pos++];
}

int main(void) {
	setvbuf(stdout, NULL, _IONBF, 0);

	unif01_Gen *gen = unif01_CreateExternGenBits("pipe_gen", read32);

	bbattery_Crush(gen);
	unif01_DeleteExternGenBits(gen);
	return 0;
}

