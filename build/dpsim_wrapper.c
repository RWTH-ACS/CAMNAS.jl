#include <stdio.h>
#include <string.h>

#include <julia_init.h>
#include <camnasjl.h>

#include <dpsim/MNASolverDynInterface.h>

int init_wrapper(struct dpsim_csr_matrix *matrix)
{
    printf("[CAMNAS] Initializing Julia...\n");
    char **argv;
    int argc = 0;

    init_julia(argc, argv);
    init(matrix);
};

void cleanup_wrapper(void)
{
    cleanup();

    printf("[CAMNAS] Shutting down Julia...\n");
    shutdown_julia(0);
}

static const char *PLUGIN_NAME = "camnasjl";
static struct dpsim_mna_plugin solver_plugin = {
    .log = log,
    .init = init_wrapper,
    .lu_decomp = decomp,
    .solve = solve,
    .cleanup = cleanup_wrapper,
};

struct dpsim_mna_plugin *get_mna_plugin(const char *name)
{
    if (name == NULL || strcmp(name, PLUGIN_NAME) != 0)
    {
        printf("error: name mismatch %s %s\n", name, PLUGIN_NAME);
        return NULL;
    }
    return &solver_plugin;
}
