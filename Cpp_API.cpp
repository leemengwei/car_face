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
    //先准备python环境路径:
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('./front_position_algorithm')");
    PyRun_SimpleString("sys.path.append('./object_detection_network')");
    PyRun_SimpleString("sys.path.append('./spatial_in_seat_network')");
    PyRun_SimpleString("sys.path.append('./side_position_algorithm')");
    cout<<"hello from C+"<<endl;
    PyRun_SimpleString("print('hello from python')");

    //准备一下一会儿要用的数据，需要弄成PyObject格式,并设置成要传的参数,两侧相机单独准备:
    Mat A_image;
    A_image = imread("./left_61.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! A_image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the A_image" << std::endl ;
        return -1;
    }
    npy_intp A_Dims[3] = { A_image.rows, A_image.cols, A_image.channels() }; 
    PyObject *A_PyArray = PyArray_SimpleNewFromData(3, A_Dims, NPY_UINT8, A_image.data);
    PyObject *A_root_dir = PyBytes_FromString("./");
    PyObject *A_ArgImg = PyTuple_New(1);    //准备空元组包
    PyObject *A_ArgDir = PyTuple_New(1);    //准备空元组包
    PyTuple_SetItem(A_ArgImg, 0, A_PyArray);   //填值
    PyTuple_SetItem(A_ArgDir, 0, A_root_dir);   //填值

    Mat B_image;
    B_image = imread("./right_61.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! B_image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the B_image" << std::endl ;
        return -1;
    }
    npy_intp B_Dims[3] = { B_image.rows, B_image.cols, B_image.channels() }; 
    PyObject *B_PyArray = PyArray_SimpleNewFromData(3, B_Dims, NPY_UINT8, B_image.data);
    PyObject *B_root_dir = PyBytes_FromString("./");
    PyObject *B_ArgImg = PyTuple_New(1);    //准备空元组包
    PyObject *B_ArgDir = PyTuple_New(1);    //准备空元组包
    PyTuple_SetItem(B_ArgImg, 0, B_PyArray);   //填值
    PyTuple_SetItem(B_ArgDir, 0, B_root_dir);   //填值
		cout<<"Image data got"<<endl;

    //调用Python算法总共包含一下4个步骤，其中包含了4个API，约定A表示司机侧，B表示另一侧。
    //STEP1（这个不是API, 是必要步骤）:
    //先import python模块
    PyObject* AModule = PyImport_ImportModule("A");
    PyObject* BModule = PyImport_ImportModule("B");
    PyObject* MergeModule = PyImport_ImportModule("seat_merge");
    if (!AModule||!BModule||!MergeModule)
    {
        PyErr_Print();
        cout << "[ERROR] Python get module failed." << endl;
        return 0;
    }
    cout << "[INFO] Python get module succeed." << endl;
    //再从模块import得到类或者函数
    PyObject* A_class = PyObject_GetAttrString(AModule, "A");
    PyObject* B_class = PyObject_GetAttrString(BModule, "B");
    PyObject* Merge_function = PyObject_GetAttrString(MergeModule, "seat_merge");
    if (!A_class||!B_class)
    {
        PyErr_Print();
        cout << "[ERROR] Can't find class " << endl;
        return 0;
    }
    cout << "[INFO] Get class succeed." << endl;

    //STEP2（API1加载模型）:
    //实例化, 此时完成初始化并加载模型，需要几秒种:
    PyObject* A_instance = PyObject_CallObject(A_class, A_ArgDir);
    PyObject* B_instance = PyObject_CallObject(B_class, B_ArgDir);
    if (!A_instance||!B_instance) 
    {
        PyErr_Print();
        printf("Can't create instance./n");
        return -1;
    }

    //STEP3(API2、API3两侧AB相机的检测与定位):
    //返回值为一个PyObject，实际上是一个python的list，可以通过相应方法得到list长度。
    PyObject *AReturn_listplot;
    PyObject *BReturn_listplot;
    AReturn_listplot = PyObject_CallMethod(A_instance, "self_logic", "O", A_ArgImg);
    BReturn_listplot = PyObject_CallMethod(B_instance, "self_logic", "O", B_ArgImg);
    if (!AReturn_listplot||!BReturn_listplot) 
    {
        PyErr_Print();
        printf("Can't return list and plot./n");
        return -1;
    }
    printf("Getting list.");
    PyObject *AReturn;
    PyObject *BReturn;
    AReturn = PyList_GetItem(AReturn_listplot, 0);
    BReturn = PyList_GetItem(BReturn_listplot, 0);
    if (!PyList_Check(AReturn)||!PyList_Check(BReturn))
    {
       PyErr_Print();
       cout<<"Error getting list back"<<endl;
			 return -1;
    }
    int A_list_size = PyList_Size(AReturn);  //使用PyList_Size方法得到返回值长度。
    int B_list_size = PyList_Size(BReturn);  //使用PyList_Size方法得到返回值长度。
    cout<<"***There are "<<A_list_size<<" in A position"<<endl;
    for(int i=0; i<A_list_size; i=i+1)
    {
        cout<<"Seats taken A are:"<<PyLong_AsLong(PyList_GetItem(AReturn, i))<<endl;
    }
    cout<<"***There are "<<B_list_size<<" in B position"<<endl;
    for(int i=0; i<B_list_size; i=i+1)
    {
         cout<<"Seats taken B are:"<<PyLong_AsLong(PyList_GetItem(BReturn, i))<<endl;
    }

    //STEP4(API4融合程序)
    //这个API既可以用来融合两侧AB相机的结果，也可以用来融合不同帧的结果，因为他们输入输出的物理意义都一样，即：人坐在了哪里如（1,2,5）
    PyObject *M_ArgList = PyTuple_New(2);    //准备空元组包
    PyTuple_SetItem(M_ArgList, 0, AReturn);   //填值
    PyTuple_SetItem(M_ArgList, 1, BReturn);   //填值
    PyObject *MReturn;
    MReturn = PyObject_CallMethod(MergeModule, "seat_merge", "O", M_ArgList);
    int M_list_size = PyList_Size(MReturn);  //使用PyList_Size方法得到返回值长度。
    cout<<"***There are "<<M_list_size<<" in merged position"<<endl;
    for(int i=0; i<M_list_size; i=i+1)
    {
        cout<<"Seats taken merged are:"<<PyLong_AsLong(PyList_GetItem(MReturn, i))<<endl;
    }

    Py_Finalize();
    return 0;
}
