#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "globalvalue.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(&thread1,SIGNAL(receiveover123()),this,SLOT(process123()));
    connect(&thread2,SIGNAL(receiveover456()),this,SLOT(process456()));
    timer_ContinuousShooting = new QTimer(this);
    connect(timer_ContinuousShooting, SIGNAL(timeout()), this, SLOT(on_pushButton_imagecommand_clicked()));
    connect(&readthread,SIGNAL(sendstring(const QByteArray )),this,SLOT(portprocess(const QByteArray )));


    ServerInit();
    SocketInit7001();
    SocketInit7002();
    QString dir_str = "/home/user/Data1";
    // 检查目录是否存在，若不存在则新建
    QDir dir;
    if (!dir.exists(dir_str))
    {
        bool res = dir.mkpath(dir_str);
        qDebug() << "新建目录是否成功" << res;
    }
    FLAG_process123=false;
    FLAG_process456=false;
    Flag_ContinuousShooting=false;
    Flag_OneCamera=false;
    Flag_TcpServerConnect=false;
    Flag_label=false;

    //ThreadsModule = PyImport_ImportModule("seat_merge");
    space_DelDir(50);


    //////////////////////////////////////////////////////////////
    jifenshijianzhenshu=1500;
    yanshixiangyingshijian=250;
    jifenshijianyuzhi=15000;
    QString qfilenameNum=QString("/home/user/canshu.txt");
    std::string filenameNum=qfilenameNum.toStdString();
    const char* filenameNumm=filenameNum.c_str();
    if ((fpcanshu=fopen(filenameNumm,"rb"))==NULL)
    {
        qDebug()<<"cannot open this file\n";
    }
    else
    {
        fscanf(fpcanshu,"%u %u", &jifenshijianzhenshu,&yanshixiangyingshijian);
        fclose(fpcanshu);
    }

    qfilenameNum=QString("/home/user/yuzhi.txt");
    filenameNum=qfilenameNum.toStdString();
    filenameNumm=filenameNum.c_str();
    if ((fpyuzhi=fopen(filenameNumm,"rb"))==NULL)
    {
        qDebug()<<"cannot open this file\n";
    }
    else
    {
        fscanf(fpyuzhi,"%u",&jifenshijianyuzhi);
        fclose(fpyuzhi);
    }

    bsfthread0.init();
    bsfthread0.start();//BFScamera0
    connect(&bsfthread0,SIGNAL(sendBFSimage00()),this,SLOT(showBFSimage00()));
    connect(&bsfthread0,SIGNAL(sendBFSimage01()),this,SLOT(showBFSimage01()));

    bsfthread1.init();
    bsfthread1.start();//BFScamera1
    connect(&bsfthread1,SIGNAL(sendBFSimage10()),this,SLOT(showBFSimage10()));
    connect(&bsfthread1,SIGNAL(sendBFSimage11()),this,SLOT(showBFSimage11()));
    //GlobalValue::BFS1_Flag=true;
    //GlobalValue::BFS1_Flag1=true;
    //connect(&bsfthread0,SIGNAL(sendyanshi()),this,SLOT(showBFSmessage()));
    //connect(&bsfthread1,SIGNAL(sendyanshi()),this,SLOT(showBFSmessage()));

    connect(&processthread,SIGNAL(jieguo(PyObject *)),this,SLOT(jiancejieguo(PyObject *)));

    TooFastNum=0;
    GlobalValue::timeserial.start();
    qDebug() << "新建";
}

MainWindow::~MainWindow()
{
    bsfthread0.terminate();
    bsfthread1.terminate();    
    delete ui;
}

void MainWindow::process123()
{
    FLAG_process123=true;
    process123456();
}

void MainWindow::process456()
{
    FLAG_process456=true;
    process123456();
}

void MainWindow::process123456()
{
    if (Flag_OneCamera)
    {
        FLAG_process123=false;
        FLAG_process456=false;
        process();
    }
    else
    {
        bool Flag=FLAG_process123&&FLAG_process456;
        if (Flag)
       {
            FLAG_process123=false;
            FLAG_process456=false;
            process();
       }
    }

}




void MainWindow::process()
{

    if (!Flag_ContinuousShooting)
    {
        GlobalValue::imageDataSend1=GlobalValue::imageData1;
        GlobalValue::imageDataSend2=GlobalValue::imageData4;        
        savethread.start();

        imageshow1(GlobalValue::imageData1);
        imageshow1(GlobalValue::imageData2);
        imageshow1(GlobalValue::imageData3);
        imageshow2(GlobalValue::imageData4);
        imageshow2(GlobalValue::imageData5);
        imageshow2(GlobalValue::imageData6);

        processthread.start();
    }
///////////////////////////////////////////////////////////////////////////////////
    space_DelDir(50);

}

void MainWindow::jiancejieguo(PyObject *MReturn4)
{
    qDebug()<<"jiancejieguoshuchu";
    int M_list_size4 = PyList_Size(MReturn4);  //使用PyList_Size方法得到返回值长度。
    qDebug()<<M_list_size4;
    quint8 total=M_list_size4-2;
    QString seat_number=QString("");

    //////////////////////////////////////////////////////////////////////
    QString qfilenameNum=GlobalValue::dir_time+QString("/Num.txt");
    //QString qfilenameNum=GlobalValue::dir+QString("/vediodata.h264");
    std::string filenameNum=qfilenameNum.toStdString();
    const char* filenameNumm=filenameNum.c_str();
    if ((fpNum=fopen(filenameNumm,"wb"))==NULL)
    {
        qDebug()<<"cannot open this file\n";
    }

    /// //////////////////////////////////////////////////////////////////
    if (total==0)
    {int seat_num =0;
        fprintf(fpNum,"%lu\n", seat_num);}
    else {
        for(int i=0; i<total; i=i+1)
        {
            cout<<"Seats taken merged are:"<<PyLong_AsLong(PyList_GetItem(MReturn4, i))<<endl;
            int seat_num = PyLong_AsLong(PyList_GetItem(MReturn4, i));
            seat_number=seat_number+QString::number(seat_num)+QString(" ");
            fprintf(fpNum,"%lu\n", seat_num);

        }
    }
    int zuohou_num = PyLong_AsLong(PyList_GetItem(MReturn4, total));
    fprintf(fpNum,"%s", "zuohou_num:");
    fprintf(fpNum,"%lu\n", zuohou_num);
    int youhou_num = PyLong_AsLong(PyList_GetItem(MReturn4, total+1));
    fprintf(fpNum,"%s", "youhou_num:");
    fprintf(fpNum,"%lu\n", youhou_num);


    fclose(fpNum);
    GlobalValue::Flag_receiveimagefinish=true;
    ui->lineEdit_total->setText(QString::number(total));
    ui->lineEdit_seatnum->setText(seat_number);

    ////////////////////////////////////////////////////////////////////////////

    if (Flag_TcpServerConnect)
        ServerWriteData(total,GlobalValue::shibiema);
    sendthread.start();
    GlobalValue::shibiema++;
    qDebug()<<"第3段程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";
}

void MainWindow::imageshow1(QByteArray imageData)
{

    //////////////////////////
    image = new QImage((uchar *)imageData.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);

    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->showImageLabel_1->width(), ui->showImageLabel_1->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    if (ui->checkBox->checkState())
    {
        quint16 pixmapwidth=ui->showImageLabel_2->width();
        quint16 pixmapheight=ui->showImageLabel_2->height();
        QPainter llpainter(&fitpixmap);
        llpainter.setPen(QPen(Qt::yellow, 1, Qt::DotLine));
        for (int i=1;i<12;i++)
        {
            llpainter.drawLine(640*i/12, 0, 640*i/12, 360);
        }
        for (int j=1;j<6;j++)
        {
            llpainter.drawLine(0, 60*j, 639,60*j);
        }
    llpainter.end();
    }

    ui->showImageLabel_1->setPixmap(fitpixmap);
}


void MainWindow::imageshow2(QByteArray imageData)
{

    //////////////////////////
    image = new QImage((uchar *)imageData.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);

    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->showImageLabel_2->width(), ui->showImageLabel_2->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    if (ui->checkBox->checkState())
    {
        quint16 pixmapwidth=ui->showImageLabel_2->width();
        quint16 pixmapheight=ui->showImageLabel_2->height();
        QPainter llpainter(&fitpixmap);
        llpainter.setPen(QPen(Qt::yellow, 1, Qt::DotLine));
        for (int i=1;i<12;i++)
        {
            llpainter.drawLine(640*i/12, 0, 640*i/12, 360);
        }
        for (int j=1;j<6;j++)
        {
            llpainter.drawLine(0, 60*j, 639,60*j);
        }
    llpainter.end();
    }

    ui->showImageLabel_2->setPixmap(fitpixmap);
}

void MainWindow::showBFSimage00()
{
    QDateTime datetime;
    QString timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    image = new QImage((uchar *)GlobalValue::BFSimageData00.data(), 640, 512, QImage::Format_Indexed8);
    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->BFSImageLabel_1->width(), ui->BFSImageLabel_1->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    ui->BFSImageLabel_1->setPixmap(fitpixmap);
}

void MainWindow::showBFSimage01()
{
    QDateTime datetime;
    QString timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    image = new QImage((uchar *)GlobalValue::BFSimageData01.data(), 640, 512, QImage::Format_Indexed8);
    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->BFSImageLabel_1->width(), ui->BFSImageLabel_1->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    ui->BFSImageLabel_1->setPixmap(fitpixmap);
}

void MainWindow::showBFSimage10()
{
    QDateTime datetime;
    QString timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    image = new QImage((uchar *)GlobalValue::BFSimageData10.data(), 640, 512, QImage::Format_Indexed8);
    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->BFSImageLabel_2->width(), ui->BFSImageLabel_2->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    ui->BFSImageLabel_2->setPixmap(fitpixmap);
}

void MainWindow::showBFSimage11()
{
    QDateTime datetime;
    QString timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    image = new QImage((uchar *)GlobalValue::BFSimageData11.data(), 640, 512, QImage::Format_Indexed8);
    QPixmap pixmap=QPixmap::fromImage(*image);
    QPixmap fitpixmap = pixmap.scaled(ui->BFSImageLabel_2->width(), ui->BFSImageLabel_2->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    ui->BFSImageLabel_2->setPixmap(fitpixmap);
}

void MainWindow::on_pushButton_sendcommand_clicked()
{
    cout<<"timeserial ----------------------------------------------------------------------------------" << GlobalValue::timeserial.elapsed() / 1000.0 << "s"<<endl;
    qDebug()<<"GlobalValue::Flag_receiveimagefinish============="<<GlobalValue::Flag_receiveimagefinish;
    if (GlobalValue::Flag_receiveimagefinish)
    {
        GlobalValue::Flag_receiveimagefinish=false;
        //qDebug()<<jifenshijianyuzhi<<jifenshijianzhenshu<<yanshixiangyingshijian;
        GlobalValue::timebfs.start();
        GlobalValue::timebfs1.start();
        GlobalValue::BFS1_imagenum0=0;
        GlobalValue::BFS1_imagenum1=0;
        GlobalValue::time.start();
        GlobalValue::baoshu1=0;
        GlobalValue::baoshu2=0;
        GlobalValue::baoshu3=0;
        GlobalValue::baoshu4=0;
        GlobalValue::baoshu5=0;
        GlobalValue::baoshu6=0;
        WriteData7001();
        //cout<<"sendcommand1111111111 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;
        WriteData7002();
        //cout<<"sendcommand2222222222 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;

        /////////////////////判断现在是否在收图,如果notrunning,开启收图线程,数据处理//////////////////////////

        QDateTime current_date_time = QDateTime::currentDateTime();
        GlobalValue::dir_time = QString("/home/user/Data1/")+current_date_time.toString("yyyy-MM-dd hh-mm-ss");
        QDir dir;
        bool res = dir.mkpath(GlobalValue::dir_time);
        //qDebug() << "新建目录是否成功" << res;

        current_date_time = QDateTime::currentDateTime();
        qDebug()<<"新建目录================================================================"<<current_date_time.toString("hh-mm-ss-zzz");
        bool bindok = thread1.udpSocket_rx->bind(QHostAddress::Any, 6001);
        qDebug()<<"6001bind="<<bindok;
        thread1.start();
        bool bindok1 = thread2.udpSocket_rx->bind(QHostAddress::Any, 6002);
        qDebug()<<"6002bind="<<bindok1;
        thread2.start();
    }
        else {
            TooFastNum++;
        }
        qDebug()<<"TooFastNum========"<<TooFastNum;


}

const unsigned short MainWindow::CRC16Tab[256]= {
    0x0000,0x1021,0x2042,0x3063,0x4084,0x50A5,0x60C6,0x70E7,
    0x8108,0x9129,0xA14A,0xB16B,0xC18C,0xD1AD,0xE1CE,0xF1EF,
    0x1231,0x0210,0x3273,0x2252,0x52B5,0x4294,0x72F7,0x62D6,
    0x9339,0x8318,0xB37B,0xA35A,0xD3BD,0xC39C,0xF3FF,0xE3DE,

    0x2462,0x3443,0x0420,0x1401,0x64E6,0x74C7,0x44A4,0x5485,

    0xA56A,0xB54B,0x8528,0x9509,0xE5EE,0xF5CF,0xC5AC,0xD58D,

    0x3653,0x2672,0x1611,0x0630,0x76D7,0x66F6,0x5695,0x46B4,

    0xB75B,0xA77A,0x9719,0x8738,0xF7DF,0xE7FE,0xD79D,0xC7BC,

    0x48C4,0x58E5,0x6886,0x78A7,0x0840,0x1861,0x2802,0x3823,

    0xC9CC,0xD9ED,0xE98E,0xF9AF,0x8948,0x9969,0xA90A,0xB92B,

    0x5AF5,0x4AD4,0x7AB7,0x6A96,0x1A71,0x0A50,0x3A33,0x2A12,

    0xDBFD,0xCBDC,0xFBBF,0xEB9E,0x9B79,0x8B58,0xBB3B,0xAB1A,

    0x6CA6,0x7C87,0x4CE4,0x5CC5,0x2C22,0x3C03,0x0C60,0x1C41,

    0xEDAE,0xFD8F,0xCDEC,0xDDCD,0xAD2A,0xBD0B,0x8D68,0x9D49,

    0x7E97,0x6EB6,0x5ED5,0x4EF4,0x3E13,0x2E32,0x1E51,0x0E70,

    0xFF9F,0xEFBE,0xDFDD,0xCFFC,0xBF1B,0xAF3A,0x9F59,0x8F78,

    0x9188,0x81A9,0xB1CA,0xA1EB,0xD10C,0xC12D,0xF14E,0xE16F,

    0x1080,0x00A1,0x30C2,0x20E3,0x5004,0x4025,0x7046,0x6067,

    0x83B9,0x9398,0xA3FB,0xB3DA,0xC33D,0xD31C,0xE37F,0xF35E,

    0x02B1,0x1290,0x22F3,0x32D2,0x4235,0x5214,0x6277,0x7256,

    0xB5EA,0xA5CB,0x95A8,0x8589,0xF56E,0xE54F,0xD52C,0xC50D,

    0x34E2,0x24C3,0x14A0,0x0481,0x7466,0x6447,0x5424,0x4405,

    0xA7DB,0xB7FA,0x8799,0x97B8,0xE75F,0xF77E,0xC71D,0xD73C,

    0x26D3,0x36F2,0x0691,0x16B0,0x6657,0x7676,0x4615,0x5634,

    0xD94C,0xC96D,0xF90E,0xE92F,0x99C8,0x89E9,0xB98A,0xA9AB,

    0x5844,0x4865,0x7806,0x6827,0x18C0,0x08E1,0x3882,0x28A3,

    0xCB7D,0xDB5C,0xEB3F,0xFB1E,0x8BF9,0x9BD8,0xABBB,0xBB9A,

    0x4A75,0x5A54,0x6A37,0x7A16,0x0AF1,0x1AD0,0x2AB3,0x3A92,

    0xFD2E,0xED0F,0xDD6C,0xCD4D,0xBDAA,0xAD8B,0x9DE8,0x8DC9,

    0x7C26,0x6C07,0x5C64,0x4C45,0x3CA2,0x2C83,0x1CE0,0x0CC1,

    0xEF1F,0xFF3E,0xCF5D,0xDF7C,0xAF9B,0xBFBA,0x8FD9,0x9FF8,

    0x6E17,0x7E36,0x4E55,0x5E74,0x2E93,0x3EB2,0x0ED1,0x1EF0};

unsigned short MainWindow::GetCRC16_CCITT(const unsigned char *buf, int nLen)
{

    if (!buf) return 0;

    unsigned short wCRC = 0;

    for( int i = 0; i < nLen; i++ )  // 从长度开始计算CRC

        wCRC = (wCRC<<8) ^ CRC16Tab[((wCRC>>8) ^ *(unsigned char *)buf++)&0x00FF];

    return wCRC;
}

void MainWindow::SocketInit7001()
{
    TCPSocket7001 = new QTcpSocket();
    TCPSocket7001->connectToHost("192.168.1.101",7001,QTcpSocket::ReadWrite);
    connect(TCPSocket7001, SIGNAL(readyRead()), this, SLOT(ReadData7001()));
}
void MainWindow::ReadData7001()
{
    QByteArray readbuffer7001= TCPSocket7001->readAll();
    //qDebug()<<"接收到TCP数据程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";
    //qDebug()<<readbuffer7001.length();
    //qDebug()<<"readbuffer7001.toHex()="<<readbuffer7001.toHex();
    quint8 readbuffer0=readbuffer7001[0];
    quint8 readbuffer1=readbuffer7001[1];
    bool a=readbuffer0==0x55;
    bool b=readbuffer1==0xaa;
    if (a&&b)
    {
        quint8 zhilingzi=readbuffer7001[3];
        switch (zhilingzi)
        {
        case 4:
            {
                bool ok;
                QByteArray  readbuffer4567=QByteArray();
                readbuffer4567.resize(4);
                readbuffer4567[0]=readbuffer7001[7];
                readbuffer4567[1]=readbuffer7001[6];
                readbuffer4567[2]=readbuffer7001[5];
                readbuffer4567[3]=readbuffer7001[4];
                quint32 intetime=readbuffer4567.toHex().toULong(&ok,16);                
                ui->label_intetime1->setText(QString::number(intetime));


                unsigned int suanfajifenshijian=0;
                suanfajifenshijian=(intetime>jifenshijianyuzhi)?25:15;
                QString qfilenameNum=QString("daylight");
                std::string filenameNum=qfilenameNum.toStdString();
                const char* filenameNumm=filenameNum.c_str();
                if ((fpsuanfa=fopen(filenameNumm,"wb"))==NULL)
                {
                    qDebug()<<"cannot open this file\n";
                }
                fprintf(fpsuanfa,"%u\n", suanfajifenshijian);
                fclose(fpsuanfa);

                break;
            }
        case 1:
            {    //qDebug()<<"command1";
                 quint16 camerastatelength=readbuffer7001[4];
                 QByteArray  camerastate = readbuffer7001.right(camerastatelength);
                 //qDebug()<<"camerastate7001="<<camerastate.toHex();
                 if (ui->comboBox_PortNO->currentText()!=NULL)
                    {
                     readthread.startSlave(ui->comboBox_PortNO->currentText(),20,camerastate);
                     qDebug()<<"发送到串口数据程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";

                 }

            }
            default:
                break;
        }

    }

}

void MainWindow::WriteData7001()
{
    QByteArray sendbuffer7001 = QByteArray();
    sendbuffer7001.resize(4);
    sendbuffer7001[0]=0x55;
    sendbuffer7001[1]=0xaa;
    sendbuffer7001[2]=0x1c;
    sendbuffer7001[3]=0x03;

    TCPSocket7001->write(sendbuffer7001);


}

void MainWindow::on_pushButton_1_clicked()
{
    /*QString jifenshijian7001=ui->lineEdit_1->text();
    QByteArray sendbuffer7001 = QByteArray();
    if (jifenshijian7001.length()==0)
    {ui->lineEdit_3->setText("输入不能为空!");}
    else
    {
        quint8 len=4+3+2+jifenshijian7001.length();
        sendbuffer7001.resize(len);
        sendbuffer7001[0]=0x55;
        sendbuffer7001[1]=0xaa;
        sendbuffer7001[2]=0x1c;
        sendbuffer7001[3]=0x01;
        sendbuffer7001[4]=len-4;
        sendbuffer7001[5]=0x50;
        sendbuffer7001[6]=0x45;
        sendbuffer7001[7]=0x3d;
        for (int i=0;i<len-10;i++)
        {
            sendbuffer7001[8+i] = jifenshijian7001[i].toLatin1();
        }
        sendbuffer7001[len-2]=0x0d;
        sendbuffer7001[len-1]=0x0a;

        TCPSocket7001->write(sendbuffer7001);
        qDebug()<<sendbuffer7001.toHex();
    }*/

    QString integralframenum7001=ui->lineEdit_integralframenum1->text();
    QString delaycorrtime7001=ui->lineEdit_delaycorrtime1->text();
    if ((delaycorrtime7001.length()==0)||(delaycorrtime7001==0))
    {}
    else
    {
        QByteArray sendbuffer7001 = QByteArray();
        sendbuffer7001.resize(12);
        sendbuffer7001[0]=0x55;
        sendbuffer7001[1]=0xaa;
        sendbuffer7001[2]=0x1c;
        sendbuffer7001[3]=0x04;
        bool ok;
        quint32 a=integralframenum7001.toULong(&ok,10);
        quint32 b=delaycorrtime7001.toULong(&ok,10);
        sendbuffer7001[4]=a&0x000000ff;
        sendbuffer7001[5]=(a&0x0000ff00)>>8;
        sendbuffer7001[6]=(a&0x00ff0000)>>16;
        sendbuffer7001[7]=(a&0xff000000)>>24;
        sendbuffer7001[8]=b&0x000000ff;
        sendbuffer7001[9]=(b&0x0000ff00)>>8;
        sendbuffer7001[10]=(b&0x00ff0000)>>16;
        sendbuffer7001[11]=(b&0xff000000)>>24;
        ui->label_intetime1->clear();
        Flag_label=false;
        TCPSocket7001->write(sendbuffer7001);
        qDebug()<<sendbuffer7001.toHex()<<integralframenum7001<<a;

        QString qfilenameNum=QString("/home/user/canshu.txt");
        std::string filenameNum=qfilenameNum.toStdString();
        const char* filenameNumm=filenameNum.c_str();
        if ((fpcanshu=fopen(filenameNumm,"wb"))==NULL)
        {
            qDebug()<<"cannot open this file\n";
        }
        fprintf(fpcanshu,"%u %u\n", a,b);
        fclose(fpcanshu);

    }

}

void MainWindow::huoqujifenshijian()
{
        QByteArray sendbuffer7001 = QByteArray();
        sendbuffer7001.resize(12);
        sendbuffer7001[0]=0x55;
        sendbuffer7001[1]=0xaa;
        sendbuffer7001[2]=0x1c;
        sendbuffer7001[3]=0x04;
        sendbuffer7001[4]=jifenshijianzhenshu&0x000000ff;
        sendbuffer7001[5]=(jifenshijianzhenshu&0x0000ff00)>>8;
        sendbuffer7001[6]=(jifenshijianzhenshu&0x00ff0000)>>16;
        sendbuffer7001[7]=(jifenshijianzhenshu&0xff000000)>>24;
        sendbuffer7001[8]=yanshixiangyingshijian&0x000000ff;
        sendbuffer7001[9]=(yanshixiangyingshijian&0x0000ff00)>>8;
        sendbuffer7001[10]=(yanshixiangyingshijian&0x00ff0000)>>16;
        sendbuffer7001[11]=(yanshixiangyingshijian&0xff000000)>>24;
        TCPSocket7001->write(sendbuffer7001);

}



///////////////////////////////////////////////
void MainWindow::SocketInit7002()
{
    TCPSocket7002 = new QTcpSocket();
    TCPSocket7002->connectToHost("192.168.1.102",7002,QTcpSocket::ReadWrite);
    connect(TCPSocket7002, SIGNAL(readyRead()), this, SLOT(ReadData7002()));
}
void MainWindow::ReadData7002()
{
    QByteArray readbuffer7002= TCPSocket7002->readAll();
    qDebug()<<readbuffer7002;
    quint8 readbuffer0=readbuffer7002[0];
    quint8 readbuffer1=readbuffer7002[1];
    bool a=readbuffer0==0x55;
    bool b=readbuffer1==0xaa;
    if (a&&b)
    {
        quint8 zhilingzi=readbuffer7002[3];
        switch (zhilingzi)
        {
            case 4:
            {
            bool ok;
            QByteArray  readbuffer4567=QByteArray();
            readbuffer4567.resize(4);
            readbuffer4567[0]=readbuffer7002[7];
            readbuffer4567[1]=readbuffer7002[6];
            readbuffer4567[2]=readbuffer7002[5];
            readbuffer4567[3]=readbuffer7002[4];
            quint32 intetime=readbuffer4567.toHex().toULong(&ok,16);
            if (Flag_label)
                 ui->label_intetime2->setText(QString::number(intetime));
            else
                ui->label_intetime1->setText(QString::number(intetime));
            break;
            }
        case 1:
            {
                 quint8 camerastatelength=readbuffer7002[4];
                 QByteArray  camerastate = readbuffer7002.right(camerastatelength);
                  //qDebug()<<"camerastate7002="<<camerastate.toHex();
                 if (ui->comboBox_PortNO->currentText()!=NULL)
                    readthread.startSlave(ui->comboBox_PortNO->currentText(),20,camerastate);
            }
            default:
                break;
        }

    }

}

void MainWindow::WriteData7002()
{
    QByteArray sendbuffer7002 = QByteArray();
    sendbuffer7002.resize(4);
    sendbuffer7002[0]=0x55;
    sendbuffer7002[1]=0xaa;
    sendbuffer7002[2]=0x1c;
    sendbuffer7002[3]=0x03;

    TCPSocket7002->write(sendbuffer7002);


}

////////////////////////////////////////////////
void MainWindow::on_pushButton_2_clicked()
{
    QString integralframenum7002=ui->lineEdit_integralframenum2->text();
    QString delaycorrtime7002=ui->lineEdit_delaycorrtime2->text();
    if ((delaycorrtime7002.length()==0)||(delaycorrtime7002==0))
    {}
    else
    {
        QByteArray sendbuffer7002 = QByteArray();
        sendbuffer7002.resize(12);
        sendbuffer7002[0]=0x55;
        sendbuffer7002[1]=0xaa;
        sendbuffer7002[2]=0x1c;
        sendbuffer7002[3]=0x04;
        bool ok;
        quint32 a=integralframenum7002.toULong(&ok,10);
        quint32 b=delaycorrtime7002.toULong(&ok,10);
        sendbuffer7002[4]=a&0x000000ff;
        sendbuffer7002[5]=(a&0x0000ff00)>>8;
        sendbuffer7002[6]=(a&0x00ff0000)>>16;
        sendbuffer7002[7]=(a&0xff000000)>>24;
        sendbuffer7002[8]=b&0x000000ff;
        sendbuffer7002[9]=(b&0x0000ff00)>>8;
        sendbuffer7002[10]=(b&0x00ff0000)>>16;
        sendbuffer7002[11]=(b&0xff000000)>>24;
        ui->label_intetime2->clear();
        Flag_label=true;
        TCPSocket7002->write(sendbuffer7002);
        qDebug()<<sendbuffer7002.toHex()<<integralframenum7002<<a;

    }
}



void MainWindow::ServerInit()
{
    readbuffer.fill(0x00,32) ;
    sendbuffer.fill(0x00,32);
    CRCflag=false;
    mp_TCPServer = new QTcpServer();
    if(!mp_TCPServer->listen(QHostAddress::Any,5550))
    {
        //QMessageBox::information(this, "QT网络通信", "服务器端监听失败");
        qDebug()<<"QT网络通信:服务器端监听失败";
        return;
    }
    else
    {
        //QMessageBox::information(this, "QT网络通信", "服务器端监听成功");
        qDebug()<<"QT网络通信:服务器端监听成功";
    }
    connect(mp_TCPServer, SIGNAL(newConnection()), this, SLOT(ServerNewConnection()));
}


void MainWindow::ServerNewConnection()
{qDebug()<<"33333333333333";
    mp_TCPSocket = mp_TCPServer->nextPendingConnection();
    if(!mp_TCPSocket)
    {
        //QMessageBox::information(this, "QT网络通信", "未正确获取客户端连接");
        qDebug()<<"QT网络通信:未正确获取客户端连接";
        return;
    }
    else
    {
        //QMessageBox::information(this, "QT网络通信", "成功接受客户端的连接");

        qDebug()<<"QT网络通信:成功接受客户端的连接";
        Flag_TcpServerConnect=true;

        connect(mp_TCPSocket, SIGNAL(readyRead()), this, SLOT(ServerReadData()));
        connect(mp_TCPSocket, SIGNAL(disconnected()), this, SLOT(ServerDisConnection()));

    }
}


void MainWindow::ServerReadData()
{
     readbuffer= mp_TCPSocket->readAll();

    //////////////CRC校验/////////////小端模式（高字节保存在高地址，低字节保存在低地址）////////////////////
    unsigned char zhframe[32];
    for(int i = 0; i <32; i++)
        zhframe[i]=readbuffer[i];
    quint8 jiaoyanhe = 0;
    for (int i=2;i<30;i++)
    {
        jiaoyanhe=jiaoyanhe+zhframe[i];//累加和
    }
    //CRC1=GetCRC16_CCITT(&zhframe[2], 28);
    //bool w1=(CRC1&0xff)==((quint8)readbuffer[30]);
    //bool w2=((CRC1>>8)&0xff)==((quint8)readbuffer[31]);
    if (jiaoyanhe==zhframe[31])//如果校验成功
    {
    CRCflag=true;qDebug()<<"right";
    on_pushButton_sendcommand_clicked();
    }
    else
    {
        CRCflag=false;
     }

}


void MainWindow::ServerWriteData(quint8 personnum,quint32 imagenum)
{
    sendbuffer[0]=0x55;
    sendbuffer[1]=0xaa;
    sendbuffer[2]=0x1c;
    sendbuffer[3]=0x01;
    sendbuffer[4]=readbuffer[4];//序号
    sendbuffer[5]=readbuffer[5];//序号


    QDateTime current_date_time = QDateTime::currentDateTime();
    ///////////////////////////////////////////////////

    QChar y1=current_date_time.toString("yy").at(0);
    QChar y2=current_date_time.toString("yy").at(1);
    sendbuffer[6]=((y1.unicode()-0x30)<<4)+(y2.unicode()-0x30);

    QChar M1=current_date_time.toString("MM").at(0);
    QChar M2=current_date_time.toString("MM").at(1);
    sendbuffer[7]=((M1.unicode()-0x30)<<4)+(M2.unicode()-0x30);

    QChar d1=current_date_time.toString("dd").at(0);
    QChar d2=current_date_time.toString("dd").at(1);
    sendbuffer[8]=((d1.unicode()-0x30)<<4)+(d2.unicode()-0x30);

    QChar H1=current_date_time.toString("HH").at(0);
    QChar H2=current_date_time.toString("HH").at(1);
    sendbuffer[9]=((H1.unicode()-0x30)<<4)+(H2.unicode()-0x30);

    QChar m1=current_date_time.toString("mm").at(0);
    QChar m2=current_date_time.toString("mm").at(1);
    sendbuffer[10]=((m1.unicode()-0x30)<<4)+(m2.unicode()-0x30);

    QChar s1=current_date_time.toString("ss").at(0);
    QChar s2=current_date_time.toString("ss").at(1);
    sendbuffer[11]=((s1.unicode()-0x30)<<4)+(s2.unicode()-0x30);
    /*sendbuffer[6]=current_date_time.toString("yy").toInt();//年
    sendbuffer[7]=current_date_time.toString("M").toInt();//月
    sendbuffer[8]=current_date_time.toString("d").toInt();//日
    sendbuffer[9]=current_date_time.toString("h").toInt();//时
    sendbuffer[10]=current_date_time.toString("m").toInt();//分
    sendbuffer[11]=current_date_time.toString("s").toInt();//秒*/

    sendbuffer[12]=personnum;//识别到的车内人数
    sendbuffer[13]=0x01;
    //tupian zhenxuhao
    sendbuffer[14]=imagenum&0x000000ff;
    sendbuffer[15]=(imagenum&0x0000ff00)>>8;
    sendbuffer[16]=(imagenum&0x00ff0000)>>16;
    sendbuffer[17]=(imagenum&0xff000000)>>24;

    //////////////CRC校验/////////////小端模式（高字节保存在高地址，低字节保存在低地址）/////////////////
    unsigned char zhframe[32];
    for(int i = 0; i <32; i++)
        zhframe[i]=sendbuffer[i];
    //unsigned short CRC2 = 0;
    //CRC2=GetCRC16_CCITT(&zhframe[2], 28);
    quint8 jiaoyanhe=0;
    for (int i=2;i<30;i++)
    {
        jiaoyanhe=jiaoyanhe+zhframe[i];//累加和
    }
    sendbuffer[30]=0x00;//CRC
    sendbuffer[31]=jiaoyanhe;//CRC

    mp_TCPSocket->write(sendbuffer);


}

void MainWindow::ServerDisConnection()
{
    //QMessageBox::information(this, "QT网络通信:与客户端的连接断开");
    qDebug()<<"QT网络通信:与客户端的连接断开";
    Flag_TcpServerConnect=false;

}


/////////////////////////计算剩余磁盘空间////////////////////////////
double MainWindow::get_disk_space()
{
    QProcess process;
    process.start("df -m /home/user/Data1");
    process.waitForFinished();
    process.readLine();
    QString str = process.readLine();

    str.replace("\n","");
    str.replace(QRegExp("( ){1,}")," ");
    auto lst = str.split(" ");
    if(lst.size() > 5)
    qDebug("挂载点:%s 已用:%.0lfGB 可用:%.0lfGB",lst[5].toStdString().c_str(),lst[2].toDouble()/1024.0,lst[3].toDouble()/1024.0);
    double sapce = lst[3].toDouble()/1024.0;
    qDebug()<<sapce;
    return sapce;
}

///////////////////////获取文件夹列表///////////////////////////////
QString MainWindow::GetFileList(QString path)
{
    QDir dir(path);

    //列出dir(path)目录下所有子文件夹
    QFileInfoList folder_list = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot,QDir::Time);
    QString name = folder_list.last().absoluteFilePath();//最旧文件夹
    qDebug()<<name;
    return name;

}

///////////////////////删除文件夹///////////////////////////////
bool MainWindow::DelDir(QString path)
{
    if (path.isEmpty()){
        return false;
    }
    QDir dir(path);
    if(!dir.exists()){
        return true;
    }
    dir.setFilter(QDir::AllEntries | QDir::NoDotAndDotDot); //设置过滤
    QFileInfoList fileList = dir.entryInfoList(); // 获取所有的文件信息
    foreach (QFileInfo file, fileList){ //遍历文件信息
        if (file.isFile()){ // 是文件，删除
            file.dir().remove(file.fileName());
        }else{ // 递归删除
            DelDir(file.absoluteFilePath());
        }
    }
    return dir.rmpath(dir.absolutePath()); // 删除文件夹
}

/////////////////////////控制剩余磁盘空间变量，满了循环删除/////////////////
void MainWindow::space_DelDir(double restspace)
{
    double space = get_disk_space();

    while (space<restspace)
    {
       QString name = GetFileList(QString("/home/user/Data1"));
       bool ok = DelDir(name);
       space = get_disk_space();
       qDebug()<<QString("shanchu:")<<name;
       qDebug()<<space;

    }
}


void MainWindow::on_checkBox_Continuous_clicked(bool checked)
{
    if (checked)
    {
        Flag_ContinuousShooting=true;
        timer_ContinuousShooting->start(500);
    }
    else
    {
        Flag_ContinuousShooting=false;
        timer_ContinuousShooting->stop();
    }
}



void MainWindow::on_checkBox_onecamera_clicked(bool checked)
{

    if (checked)
    {
        Flag_OneCamera=true;
    }
    else
    {
        Flag_OneCamera=false;
    }
}

void MainWindow::PortInfo()//查询可用的串口资源号
{
    ui->comboBox_PortNO->clear();

    foreach (const QSerialPortInfo &info, QSerialPortInfo::availablePorts()) {
        QStringList list;
        list << info.portName()
             << info.description()
             << info.manufacturer()
             << info.systemLocation()
             << (info.vendorIdentifier() ? QString::number(info.vendorIdentifier(), 16) : QString())
             << (info.productIdentifier() ? QString::number(info.productIdentifier(), 16) : QString());

        ui->comboBox_PortNO->addItem(list.first(), list);
    }
}

void MainWindow::portprocess(const QByteArray cameracommand )
{
    if (ui->checkBox_SerialPortThrough1->isChecked())
    {sendcamera7001(cameracommand);}
    if (ui->checkBox_SerialPortThrough2->isChecked())
    {sendcamera7002(cameracommand);}
}
void MainWindow::sendcamera7001(const QByteArray cameracommand7001)
{
    QByteArray sendbuffer7001 = QByteArray();
    quint8 len=5+cameracommand7001.length();
    sendbuffer7001.resize(len);
    sendbuffer7001[0]=0x55;
    sendbuffer7001[1]=0xaa;
    sendbuffer7001[2]=0x1c;
    sendbuffer7001[3]=0x01;
    sendbuffer7001[4]=cameracommand7001.length();
    for(int i=0;i<cameracommand7001.length();i++)
    {
        sendbuffer7001[5+i]=cameracommand7001[i];
    }
    TCPSocket7001->write(sendbuffer7001);
    //qDebug()<<"sendbuffer7001:"<<sendbuffer7001.toHex();
    GlobalValue::time.start();
}
void MainWindow::sendcamera7002(const QByteArray cameracommand7002)
{
    QByteArray sendbuffer7002 = QByteArray();
    quint8 len=5+cameracommand7002.length();
    sendbuffer7002.resize(len);
    sendbuffer7002[0]=0x55;
    sendbuffer7002[1]=0xaa;
    sendbuffer7002[2]=0x1c;
    sendbuffer7002[3]=0x01;
    sendbuffer7002[4]=cameracommand7002.length();
    for(int i=0;i<cameracommand7002.length();i++)
    {
        sendbuffer7002[5+i]=cameracommand7002[i];
    }
    TCPSocket7002->write(sendbuffer7002);
     //qDebug()<<"sendbuffer7002:"<<sendbuffer7002.toHex();
     //qDebug()<<sendbuffer7002.toHex();
}

void MainWindow::on_pushButton_portInit_clicked()
{
    PortInfo();
    QByteArray port("PortHello");
    readthread.startSlave(ui->comboBox_PortNO->currentText(),20,port);
}

void MainWindow::on_pushButton_imagecommand_clicked()
{
    WriteData70015();
    WriteData70025();
    /////////////////////判断现在是否在收图,如果notrunning,开启收图线程,数据处理//////////////////////////
    //if()

    QDateTime current_date_time = QDateTime::currentDateTime();
    GlobalValue::dir_time = QString("/home/user/Data1/")+current_date_time.toString("yyyy-MM-dd hh-mm-ss");
    QDir dir;
    bool res = dir.mkpath(GlobalValue::dir_time);
    qDebug() << "新建目录是否成功" << res;

    if(!thread1.isRunning())
    {
        bool bindok = thread1.udpSocket_rx->bind(QHostAddress::Any, 6001);
        qDebug()<<"6001bind="<<bindok;
        thread1.start();
    }

    if(!thread2.isRunning())
    {
        bool bindok1 = thread2.udpSocket_rx->bind(QHostAddress::Any, 6002);
        qDebug()<<"6002bind="<<bindok1;
        thread2.start();
    }
    GlobalValue::time.start();
}

void MainWindow::WriteData70015()
{
    QByteArray sendbuffer7001 = QByteArray();
    sendbuffer7001.resize(4);
    sendbuffer7001[0]=0x55;
    sendbuffer7001[1]=0xaa;
    sendbuffer7001[2]=0x1c;
    sendbuffer7001[3]=0x05;

    TCPSocket7001->write(sendbuffer7001);
}

void MainWindow::WriteData70025()
{
    QByteArray sendbuffer7002 = QByteArray();
    sendbuffer7002.resize(4);
    sendbuffer7002[0]=0x55;
    sendbuffer7002[1]=0xaa;
    sendbuffer7002[2]=0x1c;
    sendbuffer7002[3]=0x05;

    TCPSocket7002->write(sendbuffer7002);
}

void MainWindow::on_pushButton_denggonglv1_clicked()
{
    QByteArray sendbuffer7001 = QByteArray();
    sendbuffer7001.resize(5);
    sendbuffer7001[0]=0x55;
    sendbuffer7001[1]=0xaa;
    sendbuffer7001[2]=0x1c;
    sendbuffer7001[3]=0x06;
    bool ok;
    qint16 a=ui->comboBox_denggonglv1->currentText().toInt(&ok,10);
    sendbuffer7001[4]=quint8(a);
    TCPSocket7001->write(sendbuffer7001);
}

void MainWindow::on_pushButton_denggonglv2_clicked()
{
    QByteArray sendbuffer7002 = QByteArray();
    sendbuffer7002.resize(5);
    sendbuffer7002[0]=0x55;
    sendbuffer7002[1]=0xaa;
    sendbuffer7002[2]=0x1c;
    sendbuffer7002[3]=0x06;
    bool ok;
    qint16 a=ui->comboBox_denggonglv1->currentText().toInt(&ok,10);
    sendbuffer7002[4]=quint8(a);
    TCPSocket7002->write(sendbuffer7002);
}

void MainWindow::on_pushButton_yuzhi_clicked()//积分时间阈值设置
{
    bool ok;
    unsigned int yuzhi=ui->lineEdit_yuzhi->text().toUInt(&ok,10);
    QString qfilenameNum=QString("/home/user/yuzhi.txt");
    std::string filenameNum=qfilenameNum.toStdString();
    const char* filenameNumm=filenameNum.c_str();
    if ((fpyuzhi=fopen(filenameNumm,"wb"))==NULL)
    {
        qDebug()<<"cannot open this file\n";
    }
    fprintf(fpyuzhi,"%u\n", yuzhi);
    fclose(fpyuzhi);
}

void MainWindow::on_pushButton_BFS1time_clicked()
{
    bool ok;
    unsigned int ExposureTime/* = 94*/;
    ExposureTime=ui->lineEdit_BFSExposureime1->text().toUInt(&ok,10);
    bsfthread0.ConfigureExposurenodeMap(ExposureTime);
}

void MainWindow::on_pushButton_BFS1gain_clicked()
{
    bool ok;
    unsigned int Gain=ui->lineEdit_BFSGain1->text().toUInt(&ok,10);
    bsfthread0.ConfigureGainnodeMap(Gain);
}

void MainWindow::on_pushButton_BFS2time_clicked()
{
    bool ok;
    unsigned int ExposureTime=ui->lineEdit_BFSExposureime2->text().toUInt(&ok,10);
    bsfthread1.ConfigureExposurenodeMap(ExposureTime);
}

void MainWindow::on_pushButton_BFS2gain_clicked()
{
    bool ok;
    unsigned int Gain=ui->lineEdit_BFSGain2->text().toUInt(&ok,10);
    bsfthread1.ConfigureGainnodeMap(Gain);
}

void MainWindow::showBFSmessage()
{
    //QMessageBox msgBox(QMessageBox::Warning, tr("QMessageBox::warning()"),
                            //MESSAGE, 0, this);
    QDateTime current_date_time = QDateTime::currentDateTime();
    QMessageBox::information(this, tr("Empty Field"),
                  current_date_time.toString("hh-mm-ss-zzz"));
     //qDebug()<<"11111111111111111111111111111111111111111111111";
}
