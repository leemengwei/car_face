  #ifndef MAINWINDOW_H
#define MAINWINDOW_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <iostream>
#include <arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include <QMainWindow>
#include <QTcpServer>
#include <QTcpSocket>
#include <QMessageBox>
#include <QProcess>
#include <QPainter>
#include <QTimer>
#include <stdio.h>
#include "thread1.h"
#include "thread2.h"
#include "sendthread.h"
#include "globalvalue.h"
#include "readthread.h"
#include "bfsthread.h"
#include "bfsthread1.h"
#include "savethread.h"
#include "processthread.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


private:
    Ui::MainWindow *ui;


    Thread1 thread1;//消费者线程,数据解压缩
    Thread2 thread2;//消费者线程,数据解压缩
    SendThread sendthread;
    ReadThread readthread;
    QImage *image;
    QByteArray readbuffer;
    QByteArray sendbuffer;
    unsigned short GetCRC16_CCITT(const unsigned char *, int);
    bool CRCflag;
    QTcpServer *mp_TCPServer;
    QTcpSocket *mp_TCPSocket;
    static const unsigned short CRC16Tab[256];
    void process();
    void process123456();
    bool FLAG_process123,FLAG_process456;


    bool Flag_ContinuousShooting;
    bool Flag_OneCamera;
    QTimer *timer_ContinuousShooting;
    bool Flag_TcpServerConnect;

    QTcpSocket *TCPSocket7001;
    void SocketInit7001();
    void WriteData7001();
    void WriteData70015();
    void WriteData70025();

    QTcpSocket *TCPSocket7002;
    void SocketInit7002();
    void WriteData7002();
    void PortInfo();
    void startSlave();
    void sendcamera7001(const QByteArray);
    void sendcamera7002(const QByteArray);
    FILE *fpNum;
    bool Flag_label;
    FILE *fpyuzhi;
    FILE *fpcanshu;
    unsigned int jifenshijianyuzhi;
    unsigned int jifenshijianzhenshu;
    unsigned int yanshixiangyingshijian;
    FILE *fpsuanfa;
    void huoqujifenshijian();

    BFSThread bsfthread0;
    BFSThread1 bsfthread1;
    SaveThread savethread;
    ProcessThread processthread;
    quint16 TooFastNum;

private slots:
    void imageshow1(QByteArray);
    void imageshow2(QByteArray);
    void process123();
    void process456();
    void on_pushButton_sendcommand_clicked();
    void ServerNewConnection();
    void ServerInit();
    void ServerReadData();
    void ServerWriteData(quint8,quint32);
    void ServerDisConnection();

    double get_disk_space();

    QString GetFileList(QString path);

    bool DelDir(QString path);

    void space_DelDir(double restspace);

    void on_checkBox_Continuous_clicked(bool checked);

    void ReadData7001();
    void on_pushButton_1_clicked();
    void ReadData7002();
    void on_pushButton_2_clicked();

    void on_checkBox_onecamera_clicked(bool checked);
    void portprocess(const QByteArray );

    void on_pushButton_portInit_clicked();

    void on_pushButton_imagecommand_clicked();

    void on_pushButton_denggonglv1_clicked();

    void on_pushButton_denggonglv2_clicked();

    void on_pushButton_yuzhi_clicked();

    void showBFSimage00();
    void showBFSimage01();
    void showBFSimage10();
    void showBFSimage11();

    void on_pushButton_BFS1time_clicked();

    void on_pushButton_BFS1gain_clicked();

    void on_pushButton_BFS2time_clicked();

    void on_pushButton_BFS2gain_clicked();

    void showBFSmessage();

    void jiancejieguo(PyObject *MReturn4);

signals:

};

#endif // MAINWINDOW_H
