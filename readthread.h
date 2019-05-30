#ifndef READTHREAD_H
#define READTHREAD_H


#include <QThread>
#include <QString>
#include <QtSerialPort/QtSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QMutex>
#include <QWaitCondition>
#include <QTime>


class ReadThread : public QThread
{
    Q_OBJECT
public:
    ReadThread(QObject *parent = 0);
    ~ReadThread();
    void startSlave(const QString &portName, int waitTimeout,QByteArray &writeData);
    void run();

signals:
    void sendstring(const QByteArray );
    void sentstate(const QString);


private:
    QString portName;
    int waitTimeout;
    QByteArray writeData;
    QMutex mutex;
    bool quit;
    bool FlagWriteData;



};


#endif // READTHREAD_H
