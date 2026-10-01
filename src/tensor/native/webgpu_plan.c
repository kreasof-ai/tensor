/* wgpu 0.29 native prepared compute encoding. The owning Python plan validates
 * resources and captures native validation errors around this entire call.
 * Function pointers come from the already-loaded wgpu library; no extra loader,
 * library linkage, resource ownership, command-buffer replay or compiler at run.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef void (*SetPipeline)(void *, void *);
typedef void (*SetBindGroup)(void *, uint32_t, void *, size_t, const uint32_t *);
typedef void (*Dispatch)(void *, uint32_t, uint32_t, uint32_t);

static PyObject *encode(PyObject *self, PyObject *args) {
    unsigned long long pass, pipeline_fn, bind_fn, dispatch_fn;
    const char *nodes;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "Ky#KKK", &pass, &nodes, &size,
                          &pipeline_fn, &bind_fn, &dispatch_fn)) return NULL;
    if (sizeof(void *) != 8 || size % 32 || !pass || !pipeline_fn || !bind_fn || !dispatch_fn) {
        PyErr_SetString(PyExc_ValueError, "invalid native WebGPU encoder ABI");
        return NULL;
    }
    /* Validate the complete record stream before mutating the encoder. */
    for (Py_ssize_t offset=0; offset<size; offset+=32) {
        uint64_t pipeline, group;
        uint32_t grid[3];
        memcpy(&pipeline,nodes+offset,8);memcpy(&group,nodes+offset+8,8);
        memcpy(grid,nodes+offset+16,12);
        if (!pipeline || !group || !grid[0] || !grid[1] || !grid[2]) {
            PyErr_SetString(PyExc_ValueError, "invalid native WebGPU dispatch record");
            return NULL;
        }
    }
    SetPipeline set_pipeline=(SetPipeline)(uintptr_t)pipeline_fn;
    SetBindGroup set_bind=(SetBindGroup)(uintptr_t)bind_fn;
    Dispatch dispatch=(Dispatch)(uintptr_t)dispatch_fn;
    uint64_t previous=0;
    for (Py_ssize_t offset=0; offset<size; offset+=32) {
        uint64_t pipeline,group;
        uint32_t grid[3];
        memcpy(&pipeline,nodes+offset,8);memcpy(&group,nodes+offset+8,8);
        memcpy(grid,nodes+offset+16,12);
        if (pipeline != previous) set_pipeline((void *)(uintptr_t)pass,(void *)(uintptr_t)pipeline);
        previous=pipeline;
        set_bind((void *)(uintptr_t)pass,0,(void *)(uintptr_t)group,0,NULL);
        dispatch((void *)(uintptr_t)pass,grid[0],grid[1],grid[2]);
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Encode owned prepared nodes through wgpu-native."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_webgpu_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__webgpu_native(void) {return PyModule_Create(&module);}
