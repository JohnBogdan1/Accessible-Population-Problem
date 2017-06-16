#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

using namespace std;

#define DEBUG_PRINT_ON	1

/**
 * Problem characteristics
 */
#define DEGREE_TO_RADIANS		0.0174533f
#define EARTH_RADIUS			6371.0f


struct INPUT_MAP
{
	float kmrange;
	vector<unsigned int> city_pop;
	vector<float> city_lon;
	vector<float> city_lat;
};

struct OUTPUT_SOLUTION
{
	vector<unsigned int> city_accpop;
};

/**
 * MACRO error check
 */
#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

#if DEBUG_PRINT_ON
#define DEBUG_PRINT(...) do{ fprintf( stdout, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif


/**
 * General functions
 */
void read_input_map(const char* file_name, INPUT_MAP &input_map);
void write_output_solution(const char* file_name, OUTPUT_SOLUTION &output_solution);

float geo_distance(float lat1, float lon1, float lat2, float lon2);

int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);

const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program, cl_device_id device);

void read_kernel(string file_name, string &str_kernel);

