#include <stdio.h>
#include <assert.h>
#include <splatt.h>

int main(int argc, char** argv) {
	char * fname = argv[1];

	/* allocate default options */
	double * cpd_opts = splatt_default_opts();

	/* load the tensor from a file */
	int ret;
	splatt_idx_t nmodes;
	splatt_csf * tsr;
	ret = splatt_csf_load(fname, &nmodes, &tsr, cpd_opts);
	assert(ret == 0);

	/* cleanup */
	splatt_free_csf(tsr, cpd_opts);
	splatt_free_opts(cpd_opts);

	return 0;
}