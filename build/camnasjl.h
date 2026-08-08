/*
Interface structure following "dpsim/MNASolverDynInterface.h" .
These methods get bound to the dpsim interface struct in "dpsim_wrapper.c".
*/

#include <dpsim/MNASolverDynInterface.h>

int init(struct dpsim_csr_matrix *matrix);
int decomp(struct dpsim_csr_matrix *matrix);
int solve(double *rhs_values, double *lhs_values);
void camnas_log(const char *str);
void cleanup(void);
