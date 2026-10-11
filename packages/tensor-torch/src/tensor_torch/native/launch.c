/* Header-free CUDA submission shim. The wheel needs only Python headers to
 * build; consumers need the NVIDIA driver, never a host compiler or toolkit. */
#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030C0000
#include <Python.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#define CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define CUDA_CALL
#endif

typedef int (CUDA_CALL *launch_fn)(void *, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
static launch_fn launch = NULL;

static PyObject *submit(PyObject *self, PyObject *args) {
    unsigned long long fn, gx, gy, gz, bx, by, bz, shared, stream, params;
    if (!PyArg_ParseTuple(args, "KKKKKKKKKK", &fn, &gx, &gy, &gz, &bx, &by, &bz,
                          &shared, &stream, &params)) return NULL;
    if (!launch) {
#ifdef _WIN32
        HMODULE driver = GetModuleHandleA("nvcuda.dll");
        if (driver) launch = (launch_fn)GetProcAddress(driver, "cuLaunchKernel");
#else
        void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (driver) launch = (launch_fn)dlsym(driver, "cuLaunchKernel");
#endif
        if (!launch) {
            PyErr_SetString(PyExc_RuntimeError, "CUDA driver launch entry point unavailable");
            return NULL;
        }
    }
    int code;
    Py_BEGIN_ALLOW_THREADS
    code = launch((void *)(uintptr_t)fn, (unsigned)gx, (unsigned)gy, (unsigned)gz,
        (unsigned)bx, (unsigned)by, (unsigned)bz, (unsigned)shared,
        (void *)(uintptr_t)stream, (void **)(uintptr_t)params, NULL);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(code);
}

static PyMethodDef methods[] = {
    {"submit", submit, METH_VARARGS, "Submit an already validated Tensor CUDA call."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_launch", NULL, -1, methods};
PyMODINIT_FUNC PyInit__launch(void) { return PyModule_Create(&module); }
