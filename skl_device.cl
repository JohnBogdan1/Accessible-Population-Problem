#define DEGREE_TO_RADIANS		0.0174533f
#define EARTH_RADIUS			6371.0f

inline float geo_distance(float lat1, float lon1, float lat2, float lon2)
{
	float dist_lat = (lat2 - lat1) * DEGREE_TO_RADIANS;
	float dist_lon = (lon2 - lon1) * DEGREE_TO_RADIANS;
	float inter = sin(dist_lat / 2) * sin(dist_lat / 2) +
		cos(lat1 * DEGREE_TO_RADIANS) * cos(lat2 * DEGREE_TO_RADIANS) * sin(dist_lon / 2) * sin(dist_lon / 2);
	return 2 * EARTH_RADIUS * atan2(sqrt(inter), sqrt(1 - inter));
}

/* Accessible population kernel */
__kernel void mat_mul(__global float* lat_vector,
		__global float* lon_vector,
		__global uint* in_city_pop,
		__global uint* out_city_pop,
		uint size,
		float kmrange)
{
	uint gid0 = get_global_id(0);

	/* save values/pointers for optimum access */
	uint reg_s = 0;

	float curr_lat_val = lat_vector[gid0];
	float curr_lon_val = lon_vector[gid0];

	__global float* lat_p = &(lat_vector[0]);
	__global float* lon_p = &(lon_vector[0]);

	/* find the accpop for the city at the index gid0 */
	/* for the current gid0 city check the distance with all the other cities */
	for (int j = 0; j < size; j++) {
		if (geo_distance(curr_lat_val, curr_lon_val,
			*lat_p, *lon_p) <= kmrange) {
			reg_s += in_city_pop[j];
		}

		/* increment pointers */
		lat_p++;
		lon_p++;
	}

	/* save the result */
	out_city_pop[gid0] = reg_s;
}
