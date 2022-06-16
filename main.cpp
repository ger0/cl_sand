#include <SDL2/SDL.h>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <CL/cl.h>

typedef unsigned char byte;
using uint = unsigned;

// pixels per particle
//constexpr int SCALE = 4;
constexpr struct {int x = 1; int y = 1;} SCALE;
constexpr uint MAP_WIDTH   = 1600;
constexpr uint MAP_HEIGHT  = 1600;

constexpr uint SCREEN_WIDTH   = MAP_WIDTH * SCALE.x;
constexpr uint SCREEN_HEIGHT  = MAP_HEIGHT * SCALE.y;

cl_context          context;
cl_program          program;
cl_kernel           kernel;
cl_command_queue    command_queue;

bool isRunning    = true;

// ----------- data ------------
enum Type {
    EMPTY = 0,
    WALL  = 1,
    SAND  = 2,
    GAS   = 3,
};

// brush
Type brushState = EMPTY;
bool isMousePressed  = false;
struct {int x = 0; int y = 0;} mpos;

// -----------------------------
Type brush_state = EMPTY;

// map
Type map[MAP_WIDTH * MAP_HEIGHT] = {
    EMPTY
};

inline Type *mapGet(int x, int y) {
    if (x >= MAP_WIDTH || y >= MAP_HEIGHT ||
            x < 0 || y < 0) {
        return nullptr;
    }
    return &map[x + y * (MAP_WIDTH)];
}

void mapPutCell(int x, int y, Type cell) {
    if (x >= 0 && y >= 0 &&
            x < MAP_WIDTH && y < MAP_HEIGHT) {
        *(mapGet(x, y)) = cell;
    }
}

// --------------------------- inicjalizowanie opencl -------------------------------------
#define checkCL(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error on line %d %d\n", __LINE__, err); \
        return -1; \
    } \

int initCL() {
    const char *clSourceFile = "kernel.cl";

    cl_uint platform_id_count = 0;
    cl_int err = CL_SUCCESS;
    clGetPlatformIDs(0, nullptr, &platform_id_count);

    if(platform_id_count == 0) {
        fprintf(stderr, "no opencl platforms\n");
        return -1;
    }

    cl_platform_id *platform_ids = new cl_platform_id[platform_id_count]; 
    clGetPlatformIDs(platform_id_count, platform_ids, nullptr);

    cl_uint device_id_count = 0;
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
        &device_id_count);

    if(device_id_count == 0) {
        fprintf(stderr, "no opencl devices\n");
        return -1;
    }

    cl_device_id* device_ids = new cl_device_id[device_id_count]; 
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, device_id_count,
        device_ids, nullptr);

    const cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform_ids[0]),
        0, 0
    };

    context = clCreateContext(context_properties, device_id_count,
        device_ids, nullptr, nullptr, &err);
    checkCL(err);

    // otwieranie pliku z kodem zrodlowym kernela
    FILE* f = fopen(clSourceFile, "r");
    if(f == nullptr) {
        fprintf(stderr, "failed to load opencl program\n");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buff = new char[len];
    assert(fread(buff, 1, len, f) == len);
    fclose(f);
    
    size_t src_len[1]       = { len };
    char const *src_data[1] = { buff };

    program = clCreateProgramWithSource(context, 1, src_data, src_len, &err);
    checkCL(err);

    err = clBuildProgram(program, 1, device_ids, nullptr, nullptr, nullptr);
    checkCL(err);

    kernel = clCreateKernel(program, "HelloWorld", &err);
    checkCL(err);

    cl_char *mem = (cl_char*)malloc(sizeof(cl_char) * 8); 
    printf("Before: %s\n", mem);

    cl_mem mem_obj = nullptr;
    mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(cl_char), nullptr, &err);
    checkCL(err);

    command_queue = clCreateCommandQueueWithProperties(context, device_ids[0], nullptr, &err);
    checkCL(err);

    err = clEnqueueWriteBuffer(command_queue, mem_obj, CL_TRUE, 0, 8 * sizeof(cl_char), mem, 0, nullptr, nullptr);
    checkCL(err);

    cl_uint arg_index = 0;
    size_t arg_size = sizeof(cl_mem);

    err = clSetKernelArg(kernel, arg_index, arg_size, (void*)&mem_obj);
    checkCL(err);

    constexpr size_t global_work_size[] = {8};
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    checkCL(err);

    err = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0, 8 * sizeof(cl_char), mem, 0, nullptr, nullptr);
    checkCL(err);

    printf("After %s\n", mem);

    return 0;
}

// sekwencyjnie
void mapStateUpdate() {
    // iterating over entire map
    clock_t timePoint = clock();
    double deltaTime = 0.0;
    for (int y = MAP_HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            Type *cell = mapGet(x, y);
            if (cell == nullptr) {
                printf("ERROR DURING UPDATING: WRONG ID\n");
            }

            // update sand gravity
            if (*cell == SAND) {
                int mov = rand() % 3 - 1;
                Type *neighbour = mapGet(x, y + 1);
                for (unsigned i = 0; i < 2; i++) {
                    if (neighbour == nullptr) {
                        break;
                    } else {
                        neighbour = mapGet(x + (i * mov), y + 1);
                    }
                    if (neighbour != nullptr && *neighbour == EMPTY) {
                        *neighbour = *cell;
                        *cell = EMPTY;
                    }
                }
            }
        }
    }
    timePoint = clock() - timePoint;
    deltaTime = (double)timePoint / CLOCKS_PER_SEC;
    printf("Delta: %f\n", deltaTime);
}

void tryDrawing() {
    if (isMousePressed) {
        for (int x = mpos.x - 1; x <= mpos.x; x++) {
            for (int y = mpos.y - 1; y <= mpos.y; y++) {
                mapPutCell(x, y, brushState);
            }
        }
    }
}

void handleEvents(SDL_Event *e) {
    while (SDL_PollEvent(e) > 0) {
        switch(e->type) {
            case SDL_QUIT:
            isRunning = false;
            break;

            case SDL_MOUSEBUTTONDOWN:
            isMousePressed = true;

            case SDL_MOUSEMOTION:
            mpos.x = e->motion.x / SCALE.x;
            mpos.y = e->motion.y / SCALE.y;
            tryDrawing();
            //printf("Mouse moved to (%d, %d)\n", mpos_x, mpos_y);
            break;

            case SDL_MOUSEBUTTONUP:
            isMousePressed = false;
            break;

            case SDL_KEYDOWN:
            //printf("Scancode: 0x%02X\n", e->key.keysym.scancode);
            if (e->key.keysym.scancode == 0x1A) {
                brushState = WALL;
            } else if (e->key.keysym.scancode == 0x16) {
                brushState = SAND;
            } else {
                brushState = EMPTY;
            }
            break;
        }
    }
}

void renderMap(SDL_Window *window, SDL_Renderer *renderer) {
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    for (int y = 0; y < MAP_HEIGHT; y++) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            // drawing the cell
            SDL_Rect rect = {x * SCALE.x, y * SCALE.y, SCALE.x, SCALE.y};
            SDL_Colour col;
            switch (*mapGet(x, y)) {
                case EMPTY:
                    col = SDL_Colour{0, 0, 0, 255};
                    break;
                case WALL:
                    col = SDL_Colour{100, 100, 100, 255};
                    break;
                case SAND:
                    col = SDL_Colour{255, 255, 50, 255};
                    break;
                case GAS:
                    col = SDL_Colour{50, 20, 100, 255};
                    break;
            }

            SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, col.a);
            SDL_RenderFillRect(renderer, &rect);
        }
    }
    SDL_RenderPresent(renderer);
}
// RAII
template<typename T, auto Destructor>
struct Scope_Handle {
   T* ptr;
   Scope_Handle() {
      ptr = nullptr;
   }
   Scope_Handle(Scope_Handle&& rhs) {
      this->ptr = nullptr;
      *this = move(rhs);
   }
   ~Scope_Handle() {
      reset();
   }
   Scope_Handle& operator=(T* rhs) {
      assert(ptr == nullptr);
      ptr = rhs;
      return *this;
   }
   Scope_Handle& operator=(Scope_Handle&& rhs) {
      assert(ptr == nullptr);
      ptr = rhs.ptr;
      rhs.ptr = nullptr;
      return *this;
   }
   void reset() {
      if (ptr != nullptr) {
         Destructor(ptr);
         ptr = nullptr;
      }
   }
   operator T*()    const { return ptr; }
   T* operator->()  const { return ptr; }
};

int main() {
    srand(time(NULL));
    initCL();

    Scope_Handle<SDL_Window,   SDL_DestroyWindow>   window;
    Scope_Handle<SDL_Renderer, SDL_DestroyRenderer> renderer;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Failed to initialize the SDL2 library\n");
        return -1;
    }

    window = SDL_CreateWindow(
            "gasand",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            SCREEN_WIDTH, SCREEN_HEIGHT,
            0
            );
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        exit(1);
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Failed to create renderer\n");
        exit(1);
    }
    SDL_Event event;
    
    // game loop
    while (isRunning) {
        handleEvents(&event);
        tryDrawing();
        mapStateUpdate();
        renderMap(window, renderer);
        //SDL_Delay(16);
    }
    // clean up 
    SDL_Quit();
    return 0;
}
