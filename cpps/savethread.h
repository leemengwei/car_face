#ifndef SAVETHREAD_H
#define SAVETHREAD_H
#include <QThread>
#include "globalvalue.h"

class SaveThread : public QThread
{
    Q_OBJECT
    public:
        SaveThread();
        ~SaveThread();
        void run();
        QImage *image;


};

#endif // SAVETHREAD_H

