#ifndef THREAD2_H
#define THREAD2_H


#include <QThread>
#include <QtNetwork>
#include <QByteArray>
#include <QDebug>
#include "globalvalue.h"



class Thread2 : public QThread
{
    Q_OBJECT
public:
    Thread2();
    ~Thread2();
    void run();
    QUdpSocket *udpSocket_rx;
    QHostAddress groupAddress;
    quint16 zhen1;
    quint16 zhen2;
    quint16 imagenum;


private:
    //保存数据文件
    //QByteArray imageData1;
    //QByteArray imageData2;
    //QByteArray imageData3;



signals:
    void sendstring2(QByteArray);
    void receiveover456();


private slots:
    void processPendingDatagrams();

};

#endif // THREAD2_H
