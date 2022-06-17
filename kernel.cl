#include "vars.h"
enum Type {
    EMPTY = 0,
    WALL = 1,
    SAND = 2
};

struct Offset {
    uint x;
    uint y;
};

uint getID(uint x, uint y, struct Offset off) {
    return x + off.x + (off.y + y) * MAP_WIDTH;
}

bool chkSandSwap(__global uchar* in, __global uchar* outA, __global uchar* outB) {
    if (*in == EMPTY) {
        *outA = EMPTY;
        *outB = SAND;
        return true;
    } else return false;

}

kernel void updateState(global uchar* input, global uchar* output, uchar workOffset) {
    struct Offset offset = {
        .x = get_global_id(0) * 2 + workOffset,
        .y = get_global_id(1) * 2 + workOffset
    };
    if (offset.x == MAP_WIDTH - 1 || offset.y == MAP_HEIGHT - 1) {
        return;
    }

    for (uint y = 0; y < 2; y++) {
    for (uint x = 0; x < 2; x++) {
        uint id = getID(x, y, offset);
        output[id] = input[id];
    }
    }


    for (uint x = 0; x < 2; x++) {
        uint id = getID(x, 0, offset);
        if (input[id] == SAND) {
            // piasek pion w dol
            uint newID = getID(x, 1, offset);
            if (chkSandSwap(&input[newID], &output[id], &output[newID]));
            // piasek w dol-prawo
            else if (x == 0) {
                newID = getID(x + 1, 1, offset);
                chkSandSwap(&input[newID], &output[id], &output[newID]);
            } 
            // piasek w dol-lewo
            else if (x == 1) {
                newID = getID(x - 1, 1, offset);
                chkSandSwap(&input[newID], &output[id], &output[newID]);
            }
        }
    }
}
