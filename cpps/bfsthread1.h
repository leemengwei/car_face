#ifndef BFSTHREAD1_H
#define BFSTHREAD1_H


#include <QThread>
#include <QDebug>
#include <QImage>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include "globalvalue.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;


//枚举触发类型
/*enum triggerType
{
    SOFTWARE,  //软触发
    HARDWARE   //硬件触发
};*/

class BFSThread1 : public QThread
{
    Q_OBJECT
public:
    BFSThread1();
    ~BFSThread1();
    int ConfigureExposurenodeMap(double exposureTimeToSet);
    int ConfigureGainnodeMap(double GainToSet);
    void init();

private:
    void run();
    //SystemPtr system;
    //CameraList camList;
    //triggerType chosenTrigger;     //这里选择了硬件触发
    int m_number;                               //多帧采集时，一个信号采集图像的个数
    double exposureTimeToSet;           //设置的曝光值，单位us
    double gainValueToSet;           //设置的曝光值，单位us

    //int ConfigureTrigger(INodeMap & nodeMap);
    int ResetTrigger(INodeMap & nodeMap);

    CameraPtr pCam;
    int ConfigureExposure(INodeMap& nodeMap,double exposureTimeToSet);
    int ConfigureGain(INodeMap& nodeMap,double GainToSet);

    int ConfigureUserSet(CameraPtr pCam);
    QImage *BFSimage;

    int ConfigureAcquisitionFrameRate(INodeMap& nodeMap,double AcquisitionFrameRateToSet);


signals:
    void sendBFSimage10();
    void sendBFSimage11();
    void sendyanshi();


private slots:

};


#endif // BFSTHREAD1_H
