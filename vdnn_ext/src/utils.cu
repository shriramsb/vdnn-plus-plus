#include "utils.h"

int LayerDimension::getTotalSize() {
	return N * C * H * W;
}

void outOfMemory() {
	std::cout << "Out of Memory\n";
	exit(0);
}

CnmemSpace::CnmemSpace(long free_bytes) {
	this->free_bytes = free_bytes;
	this->initial_free_bytes = free_bytes;
} 

void CnmemSpace::updateSpace(CnmemSpace::Op op, long size) {

	if (op == ADD)
		free_bytes += ceil(1.0 * size / CNMEM_GRANULARITY) * CNMEM_GRANULARITY;
	else if (op == SUB)
		free_bytes -= ceil(1.0 * size / CNMEM_GRANULARITY) * CNMEM_GRANULARITY;
}

bool CnmemSpace::isAvailable() {
	if (free_bytes >= 0)
		return true;
	else
		return false;
}

long CnmemSpace::getConsumed() {
	return (initial_free_bytes - free_bytes);
}

void CnmemSpace::updateMaxConsume(size_t &max_consume) {
	max_consume = max_consume > (initial_free_bytes - free_bytes) ? max_consume : (initial_free_bytes - free_bytes);
}