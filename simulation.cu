// SPH Fluid Simulation with CUDA
// Based on Smoothed Particle Hydrodynamics method

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

// Simulation parameters
const float PARTICLE_RADIUS = 0.025f;
const float H = 4.0f * PARTICLE_RADIUS;    // Smoothing radius
const float MASS = 1.0f;
const float REST_DENSITY = 1000.0f;
const float VISCOSITY = 0.1f;
const float SURFACE_TENSION = 0.0728f;
const float GAS_CONSTANT = 2000.0f;
const float GRAVITY = -9.8f;
const float TIMESTEP = 0.001f;
const float BOUND_DAMPING = -0.5f;

// Simulation boundaries
const float DOMAIN_WIDTH = 1.0f;
const float DOMAIN_HEIGHT = 1.5f;
const float DOMAIN_DEPTH = 1.0f;
const int3 GRID_SIZE = make_int3(
    (int)(DOMAIN_WIDTH / H) + 1,
    (int)(DOMAIN_HEIGHT / H) + 1,
    (int)(DOMAIN_DEPTH / H) + 1
);

// Helper structs
struct float3 {
    float x, y, z;
    
    __host__ __device__ float3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__ float3 operator+(const float3& other) const {
        return float3(x + other.x, y + other.y, z + other.z);
    }
    
    __host__ __device__ float3 operator-(const float3& other) const {
        return float3(x - other.x, y - other.y, z - other.z);
    }
    
    __host__ __device__ float3 operator*(float scalar) const {
        return float3(x * scalar, y * scalar, z * scalar);
    }
    
    __host__ __device__ float dot(const float3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __host__ __device__ float3 normalized() const {
        float len = length();
        if (len > 1e-6f) {
            return float3(x / len, y / len, z / len);
        }
        return *this;
    }
};

struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float pressure;
    float density;
};

// Spatial hash grid for neighbor searching
struct SpatialGrid {
    int* cellStart;
    int* cellEnd;
    int* particleIndices;
    int* particleCellIndices;
    
    int numCells;
    
    SpatialGrid(int maxParticles) {
        numCells = GRID_SIZE.x * GRID_SIZE.y * GRID_SIZE.z;
        
        cudaMalloc(&cellStart, numCells * sizeof(int));
        cudaMalloc(&cellEnd, numCells * sizeof(int));
        cudaMalloc(&particleIndices, maxParticles * sizeof(int));
        cudaMalloc(&particleCellIndices, maxParticles * sizeof(int));
    }
    
    ~SpatialGrid() {
        cudaFree(cellStart);
        cudaFree(cellEnd);
        cudaFree(particleIndices);
        cudaFree(particleCellIndices);
    }
};

// Kernel functions
__device__ float3 gradSpiky(const float3& r, float h) {
    float r_len = r.x * r.x + r.y * r.y + r.z * r.z;
    if (r_len > 0.0f && r_len < h * h) {
        float d = sqrtf(r_len);
        float s = -45.0f / (M_PI * powf(h, 6)) * powf(h - d, 2);
        return float3(s * r.x / d, s * r.y / d, s * r.z / d);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

__device__ float poly6(const float3& r, float h) {
    float r_len = r.x * r.x + r.y * r.y + r.z * r.z;
    if (r_len < h * h) {
        return 315.0f / (64.0f * M_PI * powf(h, 9)) * powf(h * h - r_len, 3);
    }
    return 0.0f;
}

__device__ float laplacianViscosity(const float3& r, float h) {
    float r_len = r.x * r.x + r.y * r.y + r.z * r.z;
    if (r_len < h * h) {
        float d = sqrtf(r_len);
        return 45.0f / (M_PI * powf(h, 6)) * (h - d);
    }
    return 0.0f;
}

__device__ int3 calculateCell(float3 position, float h) {
    int x = (int)(position.x / h);
    int y = (int)(position.y / h);
    int z = (int)(position.z / h);
    
    // Clamp to grid bounds
    x = max(0, min(GRID_SIZE.x - 1, x));
    y = max(0, min(GRID_SIZE.y - 1, y));
    z = max(0, min(GRID_SIZE.z - 1, z));
    
    return make_int3(x, y, z);
}

__device__ int getCellIndex(int3 cell) {
    return cell.x + cell.y * GRID_SIZE.x + cell.z * GRID_SIZE.x * GRID_SIZE.y;
}

__global__ void calculateDensityPressure(Particle* particles, int numParticles, 
                                       SpatialGrid grid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    
    Particle& pi = particles[index];
    float sum = 0.0f;
    int3 cell = calculateCell(pi.position, H);
    
    // Loop through neighboring cells
    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 neighborCell = make_int3(
                    cell.x + offsetX,
                    cell.y + offsetY,
                    cell.z + offsetZ
                );
                
                // Skip if outside grid
                if (neighborCell.x < 0 || neighborCell.x >= GRID_SIZE.x ||
                    neighborCell.y < 0 || neighborCell.y >= GRID_SIZE.y ||
                    neighborCell.z < 0 || neighborCell.z >= GRID_SIZE.z) {
                    continue;
                }
                
                int cellIdx = getCellIndex(neighborCell);
                int start = grid.cellStart[cellIdx];
                
                if (start == -1) continue;
                
                int end = grid.cellEnd[cellIdx];
                
                for (int j = start; j < end; j++) {
                    int particleIdx = grid.particleIndices[j];
                    Particle& pj = particles[particleIdx];
                    
                    float3 r = pi.position - pj.position;
                    sum += MASS * poly6(r, H);
                }
            }
        }
    }
    
    // Update density and pressure
    pi.density = sum;
    pi.pressure = GAS_CONSTANT * (pi.density - REST_DENSITY);
}

__global__ void calculateForces(Particle* particles, int numParticles, 
                              SpatialGrid grid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    
    Particle& pi = particles[index];
    float3 pressureForce = float3(0.0f, 0.0f, 0.0f);
    float3 viscosityForce = float3(0.0f, 0.0f, 0.0f);
    float3 surfaceForce = float3(0.0f, 0.0f, 0.0f);
    float3 gradColorField = float3(0.0f, 0.0f, 0.0f);
    float laplaceColorField = 0.0f;
    
    int3 cell = calculateCell(pi.position, H);
    
    // Loop through neighboring cells
    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 neighborCell = make_int3(
                    cell.x + offsetX,
                    cell.y + offsetY,
                    cell.z + offsetZ
                );
                
                // Skip if outside grid
                if (neighborCell.x < 0 || neighborCell.x >= GRID_SIZE.x ||
                    neighborCell.y < 0 || neighborCell.y >= GRID_SIZE.y ||
                    neighborCell.z < 0 || neighborCell.z >= GRID_SIZE.z) {
                    continue;
                }
                
                int cellIdx = getCellIndex(neighborCell);
                int start = grid.cellStart[cellIdx];
                
                if (start == -1) continue;
                
                int end = grid.cellEnd[cellIdx];
                
                for (int j = start; j < end; j++) {
                    int particleIdx = grid.particleIndices[j];
                    if (particleIdx == index) continue;
                    
                    Particle& pj = particles[particleIdx];
                    
                    float3 r = pi.position - pj.position;
                    float r_len = r.length();
                    
                    if (r_len < H) {
                        // Pressure force
                        float3 gradW = gradSpiky(r, H);
                        pressureForce = pressureForce + gradW * (-MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density));
                        
                        // Viscosity force
                        float3 velDiff = pj.velocity - pi.velocity;
                        viscosityForce = viscosityForce + velDiff * (VISCOSITY * MASS * laplacianViscosity(r, H) / pj.density);
                        
                        // Surface tension
                        gradColorField = gradColorField + gradW * (MASS / pj.density);
                        laplaceColorField += MASS * laplacianViscosity(r, H) / pj.density;
                    }
                }
            }
        }
    }
    
    // Surface tension force
    float gradColorFieldLen = gradColorField.length();
    if (gradColorFieldLen > 0.0001f) {
        surfaceForce = gradColorField * (-SURFACE_TENSION * laplaceColorField / gradColorFieldLen);
    }
    
    // Gravity force
    float3 gravityForce = float3(0.0f, GRAVITY * pi.density, 0.0f);
    
    // Total force
    pi.force = pressureForce + viscosityForce + surfaceForce + gravityForce;
}

__global__ void integrate(Particle* particles, int numParticles, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    
    Particle& p = particles[index];
    
    // Euler integration
    p.velocity = p.velocity + p.force * (dt / p.density);
    p.position = p.position + p.velocity * dt;
    
    // Enforce boundary conditions
    if (p.position.x < PARTICLE_RADIUS) {
        p.velocity.x *= BOUND_DAMPING;
        p.position.x = PARTICLE_RADIUS;
    }
    if (p.position.x > DOMAIN_WIDTH - PARTICLE_RADIUS) {
        p.velocity.x *= BOUND_DAMPING;
        p.position.x = DOMAIN_WIDTH - PARTICLE_RADIUS;
    }
    
    if (p.position.y < PARTICLE_RADIUS) {
        p.velocity.y *= BOUND_DAMPING;
        p.position.y = PARTICLE_RADIUS;
    }
    if (p.position.y > DOMAIN_HEIGHT - PARTICLE_RADIUS) {
        p.velocity.y *= BOUND_DAMPING;
        p.position.y = DOMAIN_HEIGHT - PARTICLE_RADIUS;
    }
    
    if (p.position.z < PARTICLE_RADIUS) {
        p.velocity.z *= BOUND_DAMPING;
        p.position.z = PARTICLE_RADIUS;
    }
    if (p.position.z > DOMAIN_DEPTH - PARTICLE_RADIUS) {
        p.velocity.z *= BOUND_DAMPING;
        p.position.z = DOMAIN_DEPTH - PARTICLE_RADIUS;
    }
}

__global__ void resetGrid(int* cellStart, int numCells) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numCells) {
        cellStart[index] = -1;
    }
}

__global__ void identifyCells(Particle* particles, int numParticles,
                            int* particleIndices, int* particleCellIndices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    
    Particle& p = particles[index];
    int3 cell = calculateCell(p.position, H);
    int cellIndex = getCellIndex(cell);
    
    particleIndices[index] = index;
    particleCellIndices[index] = cellIndex;
}

__global__ void countingSort(int* cellStart, int* cellEnd,
                           int* particleIndices, int* particleCellIndices,
                           int numParticles, int numCells) {
    __shared__ int counter[1024]; // Assuming max numCells <= 1024
    
    if (threadIdx.x < numCells) {
        counter[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x < numParticles) {
        atomicAdd(&counter[particleCellIndices[threadIdx.x]], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < numCells) {
        int start = 0;
        for (int i = 0; i < threadIdx.x; i++) {
            start += counter[i];
        }
        cellStart[threadIdx.x] = start;
        cellEnd[threadIdx.x] = start + counter[threadIdx.x];
    }
}

__global__ void reindexParticles(int* cellStart, int* particleIndices, 
                               int* particleCellIndices, int* sortedIndices,
                               int numParticles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    
    int cellIndex = particleCellIndices[index];
    int targetIdx = atomicAdd(&cellStart[cellIndex], 1);
    sortedIndices[targetIdx] = particleIndices[index];
}

// Main simulation class
class FluidSimulator {
private:
    int numParticles;
    std::vector<Particle> hostParticles;
    Particle* deviceParticles;
    SpatialGrid* grid;
    int* tempSortedIndices;
    
public:
    FluidSimulator(int numParticles) : numParticles(numParticles) {
        // Allocate memory
        hostParticles.resize(numParticles);
        cudaMalloc(&deviceParticles, numParticles * sizeof(Particle));
        cudaMalloc(&tempSortedIndices, numParticles * sizeof(int));
        grid = new SpatialGrid(numParticles);
        
        // Initialize particles
        initializeParticles();
    }
    
    ~FluidSimulator() {
        cudaFree(deviceParticles);
        cudaFree(tempSortedIndices);
        delete grid;
    }
    
    void initializeParticles() {
        // Create a block of water particles
        float spacing = 2.0f * PARTICLE_RADIUS * 0.95f; // Slight overlap for stability
        int particlesPerSide = std::cbrt(numParticles);
        
        float xStart = DOMAIN_WIDTH * 0.25f;
        float yStart = DOMAIN_HEIGHT * 0.25f;
        float zStart = DOMAIN_DEPTH * 0.25f;
        
        int particleIndex = 0;
        for (int i = 0; i < particlesPerSide && particleIndex < numParticles; i++) {
            for (int j = 0; j < particlesPerSide && particleIndex < numParticles; j++) {
                for (int k = 0; k < particlesPerSide && particleIndex < numParticles; k++) {
                    Particle& p = hostParticles[particleIndex++];
                    p.position = float3(
                        xStart + i * spacing,
                        yStart + j * spacing,
                        zStart + k * spacing
                    );
                    p.velocity = float3(0.0f, 0.0f, 0.0f);
                    p.force = float3(0.0f, 0.0f, 0.0f);
                    p.density = REST_DENSITY;
                    p.pressure = 0.0f;
                }
            }
        }
        
        // Copy to device
        cudaMemcpy(deviceParticles, hostParticles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    }
    
    void updateSpatialGrid() {
        const int blockSize = 256;
        
        // Reset grid
        int numCells = grid->numCells;
        int gridSize = (numCells + blockSize - 1) / blockSize;
        resetGrid<<<gridSize, blockSize>>>(grid->cellStart, numCells);
        
        // Identify cells for each particle
        gridSize = (numParticles + blockSize - 1) / blockSize;
        identifyCells<<<gridSize, blockSize>>>(deviceParticles, numParticles, 
                                          grid->particleIndices, grid->particleCellIndices);
        
        // Sort particles by cell
        // Note: For simplicity, using a basic parallel counting sort.
        // For production, consider using Thrust's sort_by_key
        countingSort<<<1, numCells>>>(grid->cellStart, grid->cellEnd, 
                                grid->particleIndices, grid->particleCellIndices,
                                numParticles, numCells);
        
        // Reindex particles
        reindexParticles<<<gridSize, blockSize>>>(grid->cellStart, grid->particleIndices, 
                                            grid->particleCellIndices, tempSortedIndices,
                                            numParticles);
        
        // Copy sorted indices back
        cudaMemcpy(grid->particleIndices, tempSortedIndices, 
                   numParticles * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    void update(float dt) {
        const int blockSize = 256;
        const int gridSize = (numParticles + blockSize - 1) / blockSize;
        
        // Update spatial grid for faster neighbor searches
        updateSpatialGrid();
        
        // Calculate density and pressure
        calculateDensityPressure<<<gridSize, blockSize>>>(deviceParticles, numParticles, *grid);
        
        // Calculate forces
        calculateForces<<<gridSize, blockSize>>>(deviceParticles, numParticles, *grid);
        
        // Integrate positions and handle boundaries
        integrate<<<gridSize, blockSize>>>(deviceParticles, numParticles, dt);
    }
    
    void saveToFile(const std::string& filename) {
        // Copy data back to host
        cudaMemcpy(hostParticles.data(), deviceParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
        
        // Write to PLY file format for visualization
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open " << filename << " for writing." << std::endl;
            return;
        }
        
        // PLY header
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "element vertex " << numParticles << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "end_header\n";
        
        // Write particle data
        for (int i = 0; i < numParticles; i++) {
            const Particle& p = hostParticles[i];
            
            // Calculate color based on velocity magnitude
            float velMag = sqrtf(p.velocity.x * p.velocity.x + 
                                p.velocity.y * p.velocity.y + 
                                p.velocity.z * p.velocity.z);
            
            float normalizedVel = fmin(1.0f, velMag / 10.0f);
            int blue = (int)(255 * (1.0f - normalizedVel));
            int red = (int)(255 * normalizedVel);
            
            // Write position, normal (just using (0,1,0) for all), and color
            file << p.position.x << " " << p.position.y << " " << p.position.z << " "
                 << "0 1 0 " // Normal
                 << red << " " << 0 << " " << blue << "\n";
        }
        
        file.close();
    }
    
    void runSimulation(int numFrames, float timeStep) {
        for (int frame = 0; frame < numFrames; frame++) {
            // Update simulation
            update(timeStep);
            
            // Save frame if needed
            if (frame % 10 == 0) {
                std::string filename = "frame_" + std::to_string(frame) + ".ply";
                saveToFile(filename);
                std::cout << "Saved frame " << frame << std::endl;
            }
        }
    }
    
    void benchmarkPerformance() {
        // Warm up
        for (int i = 0; i < 10; i++) {
            update(TIMESTEP);
        }
        
        // Benchmark
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        const int numIterations = 100;
        for (int i = 0; i < numIterations; i++) {
            update(TIMESTEP);
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double timePerFrame = diff.count() / numIterations;
        
        std::cout << "Performance Benchmark:" << std::endl;
        std::cout << "Particles: " << numParticles << std::endl;
        std::cout << "Average time per frame: " << timePerFrame * 1000 << " ms" << std::endl;
        std::cout << "FPS: " << 1.0 / timePerFrame << std::endl;
    }
};

// Main function
int main(int argc, char** argv) {
    // Default parameters
    int numParticles = 20000;
    int numFrames = 200;
    bool benchmark = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-p" && i + 1 < argc) {
            numParticles = std::stoi(argv[++i]);
        } else if (arg == "-f" && i + 1 < argc) {
            numFrames = std::stoi(argv[++i]);
        } else if (arg == "-b") {
            benchmark = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -p <num>   Number of particles (default: 20000)" << std::endl;
            std::cout << "  -f <num>   Number of frames to simulate (default: 200)" << std::endl;
            std::cout << "  -b         Run performance benchmark" << std::endl;
            std::cout << "  -h, --help Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Initializing fluid simulation with " << numParticles << " particles..." << std::endl;
    FluidSimulator simulator(numParticles);
    
    if (benchmark) {
        std::cout << "Running performance benchmark..." << std::endl;
        simulator.benchmarkPerformance();
    } else {
        std::cout << "Running simulation for " << numFrames << " frames..." << std::endl;
        simulator.runSimulation(numFrames, TIMESTEP);
    }
    
    std::cout << "Simulation completed." << std::endl;
    return 0;
}
