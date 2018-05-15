#include "utils.h"

int LayerDimension::getTotalSize() {
	return N * C * H * W;
}

void outOfMemory() {
	std::cout << "Out of Memory\n";
	exit(0);
}