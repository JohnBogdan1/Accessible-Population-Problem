#include "helper.hpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>

using namespace std;

/**
* Retrieve GPU device
*/
void gpu_find(cl_device_id &device, 
		uint platform_select, 
		uint device_select)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	/* get all available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	cout << "Platforms found: " << platform_num << endl;

	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		/* no valid platform found */
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		/* get num of available OpenCL devices type ALL on the selected platform */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_ALL, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		/* get all available OpenCL devices type ALL on the selected platform */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
			  device_num, device_list, NULL));
		cout << "\tDevices found " << device_num  << endl;

		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_NAME */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_VERSION */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data; 
			delete[] attr_data;

			/* select device based on cli arguments */
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}

void solution_opencl(INPUT_MAP &input_map,
		OUTPUT_SOLUTION &output_solution, cl_device_id device)
{
	cl_int ret;
	
	cl_context context;
	cl_command_queue cmdQueue;
	cl_program program;
	cl_kernel kernel;

	string kernel_src;
	size_t buffer_size = input_map.city_pop.size();
	cl_float kmrange = input_map.kmrange;

	/* alloc memory for the output */
	output_solution.city_accpop.resize(buffer_size);

	/* create a context for the device */
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	/* create a command queue for the device in the context */
	cmdQueue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	/* create buffer object from vector using a pointer to the vector data
	 * I use CL_MEM_COPY_HOST_PTR to copy the data (from host to device)
	 * from memory referenced by the pointer specified in the arguments.
	 */
	cl_mem lat_vec = clCreateBuffer(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * buffer_size,
		input_map.city_lat.data(), &ret);
	CL_ERR( ret );

	cl_mem lon_vec = clCreateBuffer(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * buffer_size,
		input_map.city_lon.data(), &ret);
	CL_ERR( ret );

	cl_mem in_city_pop = clCreateBuffer(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * buffer_size,
		input_map.city_pop.data(), &ret);
	CL_ERR( ret );

	cl_mem out_city_pop = clCreateBuffer(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * buffer_size,
		output_solution.city_accpop.data(), &ret);
	CL_ERR( ret );

	/* copy the vectors to VRAM using a pointer to the vector data -> v.data() */
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, lat_vec, CL_TRUE, 0,
		  sizeof(cl_float) * buffer_size, input_map.city_lat.data(), 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, lon_vec, CL_TRUE, 0,
		  sizeof(cl_float) * buffer_size, input_map.city_lon.data(), 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, in_city_pop, CL_TRUE, 0,
		  sizeof(cl_uint) * buffer_size, input_map.city_pop.data(), 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, out_city_pop, CL_TRUE, 0,
		  sizeof(cl_uint) * buffer_size, output_solution.city_accpop.data(), 0, NULL, NULL));

	/* retrieve kernel source */
	read_kernel("skl_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	/* create kernel program from source */
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	/* compile the program for the given set of devices */
	ret = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	/* create kernel associated to compiled source kernel */
	kernel = clCreateKernel(program, "mat_mul", &ret);
	CL_ERR( ret );

	/* set OpenCL kernel argument */
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&lat_vec) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&lon_vec) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&in_city_pop) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&out_city_pop) );
	CL_ERR( clSetKernelArg(kernel, 4, sizeof(cl_uint), (void *)&buffer_size) );
	CL_ERR( clSetKernelArg(kernel, 5, sizeof(cl_float), (void *)&kmrange) );

	/* event notification when function finishes */
	cl_event event;

	size_t globalSize[1] = {buffer_size};

	/* execute the command on kernel */
	ret = clEnqueueNDRangeKernel(cmdQueue, 
		kernel, 1, NULL, globalSize, 0, 0, NULL, &event);
	CL_ERR( ret );

	// wait on event
	clWaitForEvents(1, &event);

	/* get the result back from GPU */
	clEnqueueReadBuffer(cmdQueue, out_city_pop, CL_TRUE, 0,
	sizeof(cl_uint) * buffer_size, output_solution.city_accpop.data(), 0, NULL, NULL);

	/* wait for all enqueued operations to finish */
	CL_ERR( clFinish(cmdQueue) );

	/* free all resources related to GPU */
	CL_ERR( clReleaseMemObject(lat_vec) );
	CL_ERR( clReleaseMemObject(lon_vec) );
	CL_ERR( clReleaseMemObject(in_city_pop) );
	CL_ERR( clReleaseMemObject(out_city_pop) );
	CL_ERR( clReleaseCommandQueue(cmdQueue) );
	CL_ERR( clReleaseContext(context) );
}

/**
 * MAIN entry function (CPU/HOST)
 */
int main(int argc, char** argv)
{
	INPUT_MAP input_map;
	OUTPUT_SOLUTION output_solution;
	cl_device_id device;
	int platform_select = 1;
	int device_select = 1;

	/* Check for input and setting problems */
	DIE(argc != 4, "Usage: <kmrange> <infile> <outfile>");

	istringstream(string(argv[1])) >> input_map.kmrange;

	/* search and select platform/devices in OpenCL */
	gpu_find(device, platform_select, device_select);

	/* Read input map */
	read_input_map(argv[2], input_map);

	/* Process solution using OpenCL */
	solution_opencl(input_map, output_solution, device);
	//solution_opencl(input_map, output_solution);

	/* Write solution to output */
	write_output_solution(argv[3], output_solution);

	return 0;
}
