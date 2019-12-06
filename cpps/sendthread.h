#ifndef SENDTHREAD_H
#define SENDTHREAD_H

#include <QThread>
#include <QtNetwork>
#include <QByteArray>
#include <QDebug>
#include <cstring>

#include "globalvalue.h"


class SendThread : public QThread
{
    Q_OBJECT
public:
    SendThread();
    ~SendThread();
    void run();

    quint16 zhenhao;
};


#endif // SENDTHREAD_H
