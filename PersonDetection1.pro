#-------------------------------------------------
#
# Project created by QtCreator 2019-03-21T10:45:42
#
#-------------------------------------------------

QT       += core gui
QT       += network
QT       += serialport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PersonDetection1
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

LIBS += /usr/lib/x86_64-linux-gnu/libstdc++.so.6

LIBS += /usr/local/lib/libpython3.6m.so

LIBS += /usr/local/lib/libopencv_calib3d.so \
            /usr/local/lib/libopencv_core.so \
            /usr/local/lib/libopencv_dnn.so \
            /usr/local/lib/libopencv_features2d.so \
            /usr/local/lib/libopencv_flann.so \
            /usr/local/lib/libopencv_highgui.so \
            /usr/local/lib/libopencv_imgcodecs.so \
            /usr/local/lib/libopencv_imgproc.so \
            /usr/local/lib/libopencv_ml.so \
            /usr/local/lib/libopencv_objdetect.so \
            /usr/local/lib/libopencv_photo.so \
            /usr/local/lib/libopencv_shape.so \
            /usr/local/lib/libopencv_stitching.so \
            /usr/local/lib/libopencv_superres.so \
            /usr/local/lib/libopencv_videoio.so \
            /usr/local/lib/libopencv_video.so \
            /usr/local/lib/libopencv_videostab.so

#LIBS += /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_core.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_features2d.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_flann.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_ml.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_photo.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_stitching.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_superres.so \
#          /usr/lib/x86_64-linux-gnu/libopencv_video.so \
#            /usr/lib/x86_64-linux-gnu/libopencv_videostab.so

#INCLUDEPATH += /usr/include/gstreamer-1.0/      \
#                            /usr/include/glib-2.0       \
#                            /usr/lib/x86_64-linux-gnu/glib-2.0/include  \
#                            /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include \
#                            /usr/include

INCLUDEPATH += /usr/local/include/python3.6m

INCLUDEPATH += /usr/local/lib/python3.6/site-packages/numpy/core/include/numpy/

INCLUDEPATH += /usr/local/include \
    /usr/local/include/opencv \
    /usr/local/include/opencv2

SOURCES += main.cpp\
        mainwindow.cpp \
    thread1.cpp \
    globalvalue.cpp \
    thread2.cpp \
    sendthread.cpp \
    readthread.cpp

HEADERS  += mainwindow.h \
    thread1.h \
    globalvalue.h \
    thread2.h \
    sendthread.h \
    readthread.h

FORMS    += mainwindow.ui
