#include "savethread.h"

SaveThread::SaveThread()
{

}

SaveThread::~SaveThread()
{

}

void SaveThread::run()
{
    image = new QImage((uchar *)GlobalValue::imageData11.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    QDateTime datetime;
    QString timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    QString tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::imageData22.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::imageData33.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::imageData44.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::imageData55.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::imageData66.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+ timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::BFSimageData00.data(), 640, 512, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+QString("BACK-")+timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::BFSimageData01.data(), 640, 512, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+QString("BACK-")+timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::BFSimageData10.data(), 640, 512, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+QString("BACK-")+timestr + ".png";
    image->save(tempImagePath, "PNG");

    image = new QImage((uchar *)GlobalValue::BFSimageData11.data(), 640, 512, QImage::Format_Indexed8);
    timestr=datetime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz");
    tempImagePath=GlobalValue::dir_time+QString("/")+QString("BACK-")+timestr + ".png";
    image->save(tempImagePath, "PNG");

    qDebug()<<tempImagePath;
    qDebug()<<"第5段程序耗时："<<GlobalValue::time.elapsed()/1000.0<<"s";

    GlobalValue::imageData1.fill(0x00,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData2.fill(0x01,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData3.fill(0x02,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData4.fill(0x03,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData5.fill(0x04,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::imageData6.fill(0x05,VIDEO_WIDTH*VIDEO_HEIGHT);
    GlobalValue::BFSimageData00.fill(0x00,640*512);
    GlobalValue::BFSimageData01.fill(0x00,640*512);
    GlobalValue::BFSimageData10.fill(0x00,640*512);
    GlobalValue::BFSimageData11.fill(0x00,640*512);

}
