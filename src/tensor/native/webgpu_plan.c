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
typedef void *(*Create)(void *, const void *);
typedef void (*End)(void *);
typedef void (*Copy)(void *, void *, uint64_t, void *, uint64_t, uint64_t);
typedef void (*Submit)(void *, size_t, void *const *);
typedef void (*Release)(void *);

static int validate_nodes(const char *nodes, Py_ssize_t size) {
    for (Py_ssize_t offset=0; offset<size; offset+=32) {
        uint64_t pipeline, group;
        uint32_t grid[3];
        memcpy(&pipeline,nodes+offset,8);memcpy(&group,nodes+offset+8,8);
        memcpy(grid,nodes+offset+16,12);
        if (!pipeline || !group || !grid[0] || !grid[1] || !grid[2]) {
            PyErr_SetString(PyExc_ValueError, "invalid native WebGPU dispatch record");
            return 0;
        }
    }
    return 1;
}

static void encode_nodes(void *pass, const char *nodes, Py_ssize_t size,
                         SetPipeline set_pipeline, SetBindGroup set_bind, Dispatch dispatch) {
    uint64_t previous=0;
    for (Py_ssize_t offset=0; offset<size; offset+=32) {
        uint64_t pipeline,group;
        uint32_t grid[3];
        memcpy(&pipeline,nodes+offset,8);memcpy(&group,nodes+offset+8,8);
        memcpy(grid,nodes+offset+16,12);
        if (pipeline != previous) set_pipeline(pass,(void *)(uintptr_t)pipeline);
        previous=pipeline;
        set_bind(pass,0,(void *)(uintptr_t)group,0,NULL);
        dispatch(pass,grid[0],grid[1],grid[2]);
    }
}

static int captured_error(PyObject *probe) {
    PyObject *value=PyObject_CallNoArgs(probe);
    if (!value) return -1;
    int result=PyObject_IsTrue(value);Py_DECREF(value);return result;
}

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
    if (!validate_nodes(nodes,size)) return NULL;
    SetPipeline set_pipeline=(SetPipeline)(uintptr_t)pipeline_fn;
    SetBindGroup set_bind=(SetBindGroup)(uintptr_t)bind_fn;
    Dispatch dispatch=(Dispatch)(uintptr_t)dispatch_fn;
    encode_nodes((void *)(uintptr_t)pass,nodes,size,set_pipeline,set_bind,dispatch);
    Py_RETURN_NONE;
}

static PyObject *submit(PyObject *self, PyObject *args) {
    unsigned long long device,queue,source,destination,bytes;
    const char *nodes,*descriptors,*functions;
    Py_ssize_t size,descriptor_size,function_size;
    PyObject *probe;
    if (!PyArg_ParseTuple(args,"KKy#y#y#KKKO",&device,&queue,&nodes,&size,
                          &descriptors,&descriptor_size,&functions,&function_size,
                          &source,&destination,&bytes,&probe)) return NULL;
    if (sizeof(void *)!=8 || !device || !queue || size%32 || descriptor_size!=24 || function_size!=96 ||
        ((!source)!=(!destination)) || (!source && bytes) || (source && !bytes) || !PyCallable_Check(probe)) {
        PyErr_SetString(PyExc_ValueError,"invalid native WebGPU submission ABI");return NULL;
    }
    uint64_t desc[3],fn[12];
    memcpy(desc,descriptors,24);memcpy(fn,functions,96);
    for (int i=0;i<3;i++) if (!desc[i]) {PyErr_SetString(PyExc_ValueError,"invalid native WebGPU descriptor");return NULL;}
    for (int i=0;i<12;i++) if (!fn[i]) {PyErr_SetString(PyExc_ValueError,"invalid native WebGPU function");return NULL;}
    if (!validate_nodes(nodes,size)) return NULL;
    void *encoder=((Create)(uintptr_t)fn[0])((void *)(uintptr_t)device,(void *)(uintptr_t)desc[0]);
    if (!encoder) {PyErr_SetString(PyExc_RuntimeError,"native WebGPU encoder allocation failed");return NULL;}
    int error=captured_error(probe);
    if (error) {((Release)(uintptr_t)fn[11])(encoder);if (error<0) return NULL;Py_RETURN_NONE;}
    void *pass=((Create)(uintptr_t)fn[1])(encoder,(void *)(uintptr_t)desc[1]);
    if (!pass) {((Release)(uintptr_t)fn[11])(encoder);PyErr_SetString(PyExc_RuntimeError,"native WebGPU pass allocation failed");return NULL;}
    error=captured_error(probe);
    if (!error) {
    encode_nodes(pass,nodes,size,(SetPipeline)(uintptr_t)fn[2],(SetBindGroup)(uintptr_t)fn[3],(Dispatch)(uintptr_t)fn[4]);
    error=captured_error(probe);
    }
    if (error) {((Release)(uintptr_t)fn[10])(pass);((Release)(uintptr_t)fn[11])(encoder);if (error<0) return NULL;Py_RETURN_NONE;}
    ((End)(uintptr_t)fn[5])(pass);
    ((Release)(uintptr_t)fn[10])(pass);
    error=captured_error(probe);
    if (error) {((Release)(uintptr_t)fn[11])(encoder);if (error<0) return NULL;Py_RETURN_NONE;}
    if (source) ((Copy)(uintptr_t)fn[6])(encoder,(void *)(uintptr_t)source,0,(void *)(uintptr_t)destination,0,bytes);
    error=captured_error(probe);
    if (error) {((Release)(uintptr_t)fn[11])(encoder);if (error<0) return NULL;Py_RETURN_NONE;}
    void *command=((Create)(uintptr_t)fn[7])(encoder,(void *)(uintptr_t)desc[2]);
    ((Release)(uintptr_t)fn[11])(encoder);
    if (!command) {PyErr_SetString(PyExc_RuntimeError,"native WebGPU command allocation failed");return NULL;}
    error=captured_error(probe);
    if (error) {((Release)(uintptr_t)fn[9])(command);if (error<0) return NULL;Py_RETURN_NONE;}
    ((Submit)(uintptr_t)fn[8])((void *)(uintptr_t)queue,1,&command);
    ((Release)(uintptr_t)fn[9])(command);
    Py_RETURN_NONE;
}

static PyMethodDef methods[]={
    {"encode",encode,METH_VARARGS,"Encode owned prepared nodes through wgpu-native."},
    {"submit",submit,METH_VARARGS,"Create, encode, copy, submit and release fresh wgpu-native command handles."},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_webgpu_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__webgpu_native(void) {return PyModule_Create(&module);}
