#include "sendthread.h"

SendThread::SendThread()
{
    zhenhao=1;

}

SendThread::~SendThread()
{

}

void SendThread::run()
{
    qDebug()<<"run";
    QUdpSocket *udpSocket_send= new QUdpSocket(this);
    QByteArray data_send("");
    data_send.resize(984);

    data_send[0]=0x18;
    data_send[1]=0x00;
    data_send[2]=0x00;
    data_send[3]=0x00;

    data_send[4]=0xd8;
    data_send[5]=0x03;
    data_send[6]=0x00;
    data_send[7]=0x00;

    data_send[8]=0x00;
    data_send[9]=0xd2;
    data_send[10]=0x0f;
    data_send[11]=0x00;

    data_send[12]=0x38;
    data_send[13]=0x04;
    data_send[14]=0x00;
    data_send[15]=0x00;


    data_send[20]=GlobalValue::shibiema&0x000000ff;
    data_send[21]=(GlobalValue::shibiema&0x0000ff00)>>8;
    data_send[22]=(GlobalValue::shibiema&0x00ff0000)>>16;
    data_send[23]=(GlobalValue::shibiema&0xff000000)>>24;

    for (quint16 i=0;i<540;i++)
    {
        zhenhao=2*i+1;
        data_send[16]=zhenhao&0x000000ff;
        data_send[17]=(zhenhao&0x0000ff00)>>8;
        data_send[18]=(zhenhao&0x00ff0000)>>16;
        data_send[19]=(zhenhao&0xff000000)>>24;
        memcpy(data_send.data()+24, GlobalValue::imageDataSend1.data()+i*960, 960);
        udpSocket_send->writeDatagram(data_send, QHostAddress("192.168.0.124"), 6000);
        //udpSocket_send->writeDatagram(data_send, QHostAddress::Broadcast, 6000);

        usleep(100);
        //qDebug()<<zhenhao<<data_send.size();
        zhenhao=2*i+2;
        data_send[16]=zhenhao&0x000000ff;
        data_send[17]=(zhenhao&0x0000ff00)>>8;
        data_send[18]=(zhenhao&0x00ff0000)>>16;
        data_send[19]=(zhenhao&0xff000000)>>24;
        memcpy(data_send.data()+24, GlobalValue::imageDataSend2.data()+i*960, 960);
        udpSocket_send->writeDatagram(data_send, QHostAddress("192.168.0.124"), 6000);
        //udpSocket_send->writeDatagram(data_send, QHostAddress::Broadcast, 6000);
        //qDebug()<<zhenhao<<data_send.size();
        //qDebug()<<zhenhao;
        usleep(100);


    }
    //qDebug()<<"第6段程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";
    udpSocket_send->close();
    GlobalValue::imageDataSend1.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageDataSend2.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);

}
