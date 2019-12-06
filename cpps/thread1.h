#ifndef THREAD1_H
#define THREAD1_H

#include <QThread>
#include <QtNetwork>
#include <QByteArray>
#include <QDebug>

#include "globalvalue.h"



class Thread1 : public QThread
{
    Q_OBJECT
public:
    Thread1();
    ~Thread1();
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
    void sendstring1(QByteArray);
    void receiveover123();



private slots:
    void processPendingDatagrams();
};

#endif // THREAD1_H
