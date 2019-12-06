#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <iostream>
#include <arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <stdio.h>
#include <QThread>
#include "globalvalue.h"


class ProcessThread: public QThread
{Q_OBJECT
public:
    ProcessThread();
    void run();
    ~ProcessThread();

private:
    bool SuanfaInit();

    PyObject* processA1;
    PyObject* processA2;
    PyObject* processA3;
    PyObject* processB1;
    PyObject* processB2;
    PyObject* processB3;
    PyObject* ThreadsModule;
    Mat QImageToMat(QImage* image);
    Mat convertTo3Channels(const Mat& binImg);
    PyObject* convertToPyObject(QByteArray imageData);
    PyObject* A_class;
    PyObject* workers_instance;
    PyObject* algorithm_detection_and_merge_function;
    PyObject* NothingModule;

signals:
    void jieguo(PyObject *MReturn4);
};

#endif // PROCESSTHREAD_H
