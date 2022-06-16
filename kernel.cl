
__kernel void HelloWorld(__global int* data) {
    int gid = get_global_id(0);
    data[0] = 'H';
    data[1] = 'e';
    data[2] = 'l';
    data[3] = 'l';
    data[4] = 'o';
    data[5] = '!';
    data[5] = '\n';
}
