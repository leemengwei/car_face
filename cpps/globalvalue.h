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

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;

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
    static QTime timebfs;
    static QTime timebfs1;
    static quint32 shibiema;
    static QTime timeserial;
    //static bool BFS1_Flag;
    //static bool BFS1_Flag1;
    static int BFS1_imagenum0;
    static int BFS1_imagenum1;

    static SystemPtr system;
    static CameraList camList;
    static bool BFSFlag_change;
    static QByteArray BFSimageData00;
    static QByteArray BFSimageData01;
    static QByteArray BFSimageData10;
    static QByteArray BFSimageData11;

    static QByteArray imageData11;
    static QByteArray imageData22;
    static QByteArray imageData33;
    static QByteArray imageData44;
    static QByteArray imageData55;
    static QByteArray imageData66;

    static QString dir_time;
    static int baoshu1;
    static int baoshu2;
    static int baoshu3;
    static int baoshu4;
    static int baoshu5;
    static int baoshu6;

    static bool Flag_receiveimagefinish;

};

#endif // GLOBALVALUE_H
