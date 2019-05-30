#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <iostream>
#include <arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    Py_Initialize();
    import_array();

    PyRun_SimpleString("import sys,os");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('./front_position_algorithm')");
    PyRun_SimpleString("sys.path.append('./object_detection_network')");
    PyRun_SimpleString("sys.path.append('./spatial_in_seat_network')");
    PyRun_SimpleString("sys.path.append('./side_position_algorithm')");
    
    PyObject* ThreadsModule = PyImport_ImportModule("threads_start");
     if (!ThreadsModule)
    {
        PyErr_Print();
        cout << "[ERROR] Python get module failed." << endl;
        return 0;
    }
    cout << "[INFO] Python get module succeed." << endl;
  
  PyObject* workers_cluster_class = PyObject_GetAttrString(ThreadsModule, "workers_cluster");
   if (!workers_cluster_class)
    {
        PyErr_Print();
        cout << "[ERROR] Can't find class " << endl;
        return 0;
    }
    cout << "[INFO] Get class succeed." << endl;

    PyObject* workers_instance = PyObject_CallObject(workers_cluster_class, NULL);
    if (!workers_instance)
    {
        PyErr_Print();
        printf("Can't create instance./n");
        return -1;
    }


    Mat B1_image;
    Mat B2_image;
    Mat B3_image;
    Mat B4_image;
    Mat B5_image;
    Mat B6_image;
    B1_image = imread("./right.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    B2_image = imread("./left.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    B3_image = imread("./right.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    B4_image = imread("./left.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    B5_image = imread("./right.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    B6_image = imread("./left.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    npy_intp B_Dims[3] = { B1_image.rows, B1_image.cols, B1_image.channels() }; 
    PyObject *B1_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B1_image.data);
    PyObject *B2_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B2_image.data);
    PyObject *B3_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B3_image.data);
    PyObject *B4_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B4_image.data);
    PyObject *B5_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B5_image.data);
    PyObject *B6_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B6_image.data);
   

    PyObject *B123456_ArgImg = PyTuple_New(7);
    PyTuple_SetItem(B123456_ArgImg, 0, workers_instance);   //填值
    PyTuple_SetItem(B123456_ArgImg, 1, B1_PyArray);   //填值
    PyTuple_SetItem(B123456_ArgImg, 2, B2_PyArray);   //填值
    PyTuple_SetItem(B123456_ArgImg, 3, B3_PyArray);   //填值
    PyTuple_SetItem(B123456_ArgImg, 4, B4_PyArray);   //填值
    PyTuple_SetItem(B123456_ArgImg, 5, B5_PyArray);   //填值
    PyTuple_SetItem(B123456_ArgImg, 6, B6_PyArray);   //填值

    for(int i=0; i<5; i=i+1) 
    {
    PyObject* MReturn4;
    //PyObject* NothingModule = PyImport_ImportModule("DoNothingModule");
    MReturn4 = PyObject_CallMethod(ThreadsModule, "algorithm_detection_and_merge", "O", B123456_ArgImg);

    }


}
