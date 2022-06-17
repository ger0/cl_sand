enum Type {
    EMPTY = 0,
    WALL = 1,
    SAND = 2
};

kernel void updateState(const uint width, const uint height, global int* input, global int* output) {
    uint off_x = get_global_id(0);
    uint off_y = get_global_id(1);

    uint loc_x = get_local_id(0);
    uint loc_y = get_local_id(1);

    output[(off_x * 8) + loc_x * 2 + ((off_y * 8) + loc_y * 2) * width] = SAND;
}
