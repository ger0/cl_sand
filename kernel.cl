enum Type {
    EMPTY = 0,
    WALL = 1,
    SAND = 2
};

kernel void updateState(const uint width, const uint height, global int* input, global int* output) {
    uint off_x = get_global_id(0) * 2;
    uint off_y = get_global_id(1) * 2;

    for (uint y = 0; y < 1; y++) {
        for (uint x = 0; x < 1; x++) {
            uint id = x + off_x + (off_y + y) * width;
            switch (input[id]) {
                case SAND:
                output[id] = SAND; 
                break;
                case WALL:
                output[id] = WALL; 
                break;
                default:
                output[id] = EMPTY;
            }
        }
    }
}
