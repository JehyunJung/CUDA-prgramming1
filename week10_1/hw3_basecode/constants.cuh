#define TILE_SIZE 16
#define BLOCK_SIZE TILE_SIZE+FILTER_SIZE-1

__constant__ float Mc[FILTER_SIZE][FILTER_SIZE];