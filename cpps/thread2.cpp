#include "thread2.h"

Thread2::Thread2()
{
    udpSocket_rx= new QUdpSocket(this);
    udpSocket_rx->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption,20*1024*1024);
    udpSocket_rx->bind(QHostAddress::Any, 6002);
    connect(udpSocket_rx, SIGNAL(readyRead()), this, SLOT(processPendingDatagrams()));
    zhen1=0;
    zhen2=0;
    imagenum=0;
    GlobalValue::imageData4.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData4.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData5.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData5.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData6.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData6.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData44.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData44.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData55.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData55.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData66.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData66.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);

}

Thread2::~Thread2()
{
    udpSocket_rx->close();
}

void Thread2::run()
{
    bool a = udpSocket_rx->isOpen();
    //qDebug()<<"udpSocket_rx->isOpen()"<<a;

}

void Thread2::processPendingDatagrams()
{
    while (udpSocket_rx->hasPendingDatagrams())
    {
        QByteArray datagram_rx("");
        datagram_rx.resize(udpSocket_rx->pendingDatagramSize());
        udpSocket_rx->readDatagram(datagram_rx.data(), datagram_rx.size());
        //qDebug()<<datagram_rx.size();
        //qDebug()<<"接收到的数据"<<datagram_rx.toHex();
        //GlobalValue::timeList.append(datagram_rx);

        /////////////////////数据处理///////////////////////
        QByteArray datatest("");
        datatest.resize(4);
        datatest[0]=datagram_rx[19];
        datatest[1]=datagram_rx[18];
        datatest[2]=datagram_rx[17];
        datatest[3]=datagram_rx[16];
        bool ok;
        quint16 zhen2=datatest.toHex().toInt(&ok,16);
        //qDebug()<<packageHead->uDataPackageCurrIndex;
        QByteArray imagebao=datagram_rx.right(1080);
        if (zhen2 == 1)
        {
            imagenum++;
        }
        if ((zhen2 == 1)&&(imagenum==1))
        {
            //cout<<"image44444444444444444 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;
        }
        switch (imagenum) {
        case 1:
        {memcpy(GlobalValue::imageData4.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData44.data()+(zhen2-1)*1080,imagebao,1080);
        //qDebug()<<"zhen444444num="<<zhen2;
            break;}
        case 2:
        {memcpy(GlobalValue::imageData5.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData55.data()+(zhen2-1)*1080,imagebao,1080);
        //qDebug()<<"zhen555555num="<<zhen2;
            break;}
        case 3:
        {memcpy(GlobalValue::imageData6.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData66.data()+(zhen2-1)*1080,imagebao,1080);
        //qDebug()<<"zhen666666num="<<zhen2;
            if (zhen2==480)
            {
                imagenum=0;
                udpSocket_rx->close();
                qDebug()<<"第2段程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";
                emit receiveover456();

            }

            break;}
        default:
            break;
        }

    }

}

