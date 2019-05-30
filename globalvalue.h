#ifndef GLOBALVALUE_H
#define GLOBALVALUE_H
#include <QByteArray>
#include <QTime>
#include <QtSerialPort/QtSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#define VIDEO_WIDTH 960
#define VIDEO_HEIGHT 540

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <iostream>
#include <arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include <QImage>

class GlobalValue
{
public:
    GlobalValue();
    static QByteArray imageData1;
    static QByteArray imageData2;
    static QByteArray imageData3;
    static QByteArray imageData4;
    static QByteArray imageData5;
    static QByteArray imageData6;
    static QByteArray imageDataSend1;
    static QByteArray imageDataSend2;

    //static QByteArray imageData111;

    static QTime time;
    static quint32 shibiema;

};

#endif // GLOBALVALUE_H
