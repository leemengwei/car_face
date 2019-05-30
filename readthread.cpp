#include "readthread.h"

ReadThread::ReadThread(QObject *parent)
    : QThread(parent), waitTimeout(0), quit(false)
{
    FlagWriteData=false;
}
ReadThread::~ReadThread()
{
    mutex.lock();
    quit = true;
    mutex.unlock();
    wait();
}

void ReadThread::startSlave(const QString &portName, int waitTimeout,QByteArray &writeData)
{
    QMutexLocker locker(&mutex);
    if (portName!=NULL)
    {
    this->portName = portName;
    this->waitTimeout = waitTimeout;
    this->writeData = writeData;
    FlagWriteData=true;
    //qDebug()<<writeData.toHex();
    if (!isRunning())
        start();
    }
}

void ReadThread::run()
{
    bool currentPortNameChanged = false;


    mutex.lock();
    QString currentPortName;
    if (currentPortName != portName) {
        currentPortName = portName;
        currentPortNameChanged = true;
    }
    QByteArray currentWriteData;
    currentWriteData = writeData;

    int currentWaitTimeout = waitTimeout;
    mutex.unlock();
    QSerialPort serial;

    while (!quit) {

        //qDebug()<<"POrt";
       if (currentPortNameChanged) {
            serial.close();
            serial.setPortName(currentPortName);

            if (!serial.open(QIODevice::ReadWrite)) {
                qDebug()<<"can't open serialport";
                return;
            }
            else {
                emit sentstate(currentPortName);
                qDebug()<<" open serialport";
            }

            if (!serial.setBaudRate(QSerialPort::Baud115200)) {
                return;
            }

            if (!serial.setDataBits(QSerialPort::Data8)) {
                return;
            }

            if (!serial.setParity(QSerialPort::NoParity)) {
                return;
            }

            if (!serial.setStopBits(QSerialPort::OneStop)) {
                return;
            }

            if (!serial.setFlowControl(QSerialPort::NoFlowControl)) {
                return;
            }
        }
        if (FlagWriteData)
        {
            if (currentWaitTimeout)
            {
                serial.write(currentWriteData);
                FlagWriteData=false;
            }
        }

        if (serial.waitForReadyRead(currentWaitTimeout))
        {
            // read request
            QByteArray requestData = serial.readAll();
            if(serial.waitForReadyRead(20))
            requestData += serial.readAll();
            //qDebug()<<"requestData.length="<<requestData.length();
            qDebug()<<requestData.toHex();
            emit sendstring(requestData);
        }
        mutex.lock();
        if (currentPortName != portName) {
            currentPortName = portName;
            currentPortNameChanged = true;
        } else {
            currentPortNameChanged = false;
        }
        currentWriteData = writeData;
        currentWaitTimeout = waitTimeout;
        mutex.unlock();


    }
    qDebug()<<"read thread quit";
}

