#include "thread1.h"

Thread1::Thread1()
{
    udpSocket_rx= new QUdpSocket(this);
    udpSocket_rx->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption,20*1024*1024);
    bool bindok = udpSocket_rx->bind(QHostAddress::Any, 6001);
    //qDebug()<<bindok;
    connect(udpSocket_rx, SIGNAL(readyRead()), this, SLOT(processPendingDatagrams()));
    zhen1=0;
    zhen2=0;
    imagenum=0;
    GlobalValue::imageData1.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData1.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData2.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData2.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData3.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData3.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);

    GlobalValue::imageData11.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData11.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData22.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData22.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData33.resize(VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData33.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);

}

Thread1::~Thread1()
{

    udpSocket_rx->close();
}

void Thread1::run()
{
    bool a = udpSocket_rx->isOpen();
    //qDebug()<<"udpSocket_rx->isOpen()"<<a;

}

void Thread1::processPendingDatagrams()
{
    while (udpSocket_rx->hasPendingDatagrams())
    {
        QByteArray datagram_rx("");
        datagram_rx.resize(udpSocket_rx->pendingDatagramSize());
        udpSocket_rx->readDatagram(datagram_rx.data(), datagram_rx.size());
        //if(datagram_rx.size()!=1104)
            //qDebug()<<"datagram_rx.size()="<<datagram_rx.size();
        //qDebug()<<datagram_rx.size();
        //qDebug()<<"接收到数据";
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
        QByteArray imagebao=datagram_rx.right(1080);
        if (zhen2 == 1)
        {
            imagenum++;
            //qDebug()<<"-------------------------------image++------------------------------------";
        }
        if ((zhen2 == 1)&&(imagenum==1))
        {
            //cout<<"image111111111111111111111 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;
        }
        switch (imagenum) {
        case 1:
        {memcpy(GlobalValue::imageData1.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData11.data()+(zhen2-1)*1080,imagebao,1080);
        //qDebug()<<"zhen111111num="<<zhen2;
        GlobalValue::baoshu1++;
            break;}
        case 2:
        {memcpy(GlobalValue::imageData2.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData22.data()+(zhen2-1)*1080,imagebao,1080);
        //qDebug()<<"zhen222222num="<<zhen2;
        GlobalValue::baoshu2++;
            break;}
        case 3:
        {memcpy(GlobalValue::imageData3.data()+(zhen2-1)*1080,imagebao,1080);
        memcpy(GlobalValue::imageData33.data()+(zhen2-1)*1080,imagebao,1080);
        GlobalValue::baoshu3++;
        //qDebug()<<"zhen333333num="<<zhen2;
            if (zhen2==480)
            {
                imagenum=0;
                udpSocket_rx->close();
                qDebug()<<"第1段程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";
                emit receiveover123();
                //qDebug()<<"image1baozhu="<<GlobalValue::baoshu1;
                //qDebug()<<"image2baozhu="<<GlobalValue::baoshu2;
                //qDebug()<<"image3baozhu="<<GlobalValue::baoshu3;

            }

            break;}
        default:
            break;
        }

    }

}


