#include "globalvalue.h"

GlobalValue::GlobalValue()
{

}
QByteArray GlobalValue::imageData1 = QByteArray();
QByteArray GlobalValue::imageData2 = QByteArray();
QByteArray GlobalValue::imageData3 = QByteArray();
QByteArray GlobalValue::imageData4 = QByteArray();
QByteArray GlobalValue::imageData5 = QByteArray();
QByteArray GlobalValue::imageData6 = QByteArray();
QByteArray GlobalValue::imageDataSend1 = QByteArray();
QByteArray GlobalValue::imageDataSend2 = QByteArray();
QTime GlobalValue::timebfs;
QTime GlobalValue::timebfs1;
QTime GlobalValue::timeserial;

//QByteArray GlobalValue::imageData111 = QByteArray();

QTime GlobalValue::time = QTime();
quint32 GlobalValue::shibiema = 0;

//bool GlobalValue::BFS1_Flag=false;
//bool GlobalValue::BFS1_Flag1=false;
int GlobalValue::BFS1_imagenum0=0;
int GlobalValue::BFS1_imagenum1=0;

SystemPtr GlobalValue::system = System::GetInstance();

// Retrieve list of cameras from the system
CameraList GlobalValue::camList = system->GetCameras();

bool GlobalValue::BFSFlag_change=false;
QByteArray GlobalValue::BFSimageData00 = QByteArray();
QByteArray GlobalValue::BFSimageData01 = QByteArray();
QByteArray GlobalValue::BFSimageData10 = QByteArray();
QByteArray GlobalValue::BFSimageData11 = QByteArray();

QByteArray GlobalValue::imageData11 = QByteArray();
QByteArray GlobalValue::imageData22 = QByteArray();
QByteArray GlobalValue::imageData33 = QByteArray();
QByteArray GlobalValue::imageData44 = QByteArray();
QByteArray GlobalValue::imageData55 = QByteArray();
QByteArray GlobalValue::imageData66 = QByteArray();
QString GlobalValue::dir_time = QString();
int GlobalValue::baoshu1=0;
int GlobalValue::baoshu2=0;
int GlobalValue::baoshu3=0;
int GlobalValue::baoshu4=0;
int GlobalValue::baoshu5=0;
int GlobalValue::baoshu6=0;
bool GlobalValue::Flag_receiveimagefinish=true;
