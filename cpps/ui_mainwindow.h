/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_9;
    QPushButton *pushButton_sendcommand;
    QPushButton *pushButton_imagecommand;
    QSplitter *splitter;
    QLabel *label;
    QLineEdit *lineEdit_seatnum;
    QLabel *label_2;
    QLineEdit *lineEdit_total;
    QVBoxLayout *verticalLayout_4;
    QCheckBox *checkBox_onecamera;
    QCheckBox *checkBox_Continuous;
    QCheckBox *checkBox;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *verticalLayout_3;
    QPushButton *pushButton_portInit;
    QComboBox *comboBox_PortNO;
    QHBoxLayout *horizontalLayout_12;
    QPushButton *pushButton_yuzhi;
    QLineEdit *lineEdit_yuzhi;
    QSpacerItem *verticalSpacer_3;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_8;
    QVBoxLayout *verticalLayout;
    QLabel *label_3;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_5;
    QLineEdit *lineEdit_integralframenum1;
    QHBoxLayout *horizontalLayout;
    QLabel *label_7;
    QLineEdit *lineEdit_delaycorrtime1;
    QPushButton *pushButton_1;
    QCheckBox *checkBox_SerialPortThrough1;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_10;
    QLabel *label_intetime1;
    QHBoxLayout *horizontalLayout_10;
    QComboBox *comboBox_denggonglv1;
    QPushButton *pushButton_denggonglv1;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout_14;
    QLineEdit *lineEdit_BFSExposureime1;
    QPushButton *pushButton_BFS1time;
    QHBoxLayout *horizontalLayout_13;
    QLineEdit *lineEdit_BFSGain1;
    QPushButton *pushButton_BFS1gain;
    QLabel *showImageLabel_1;
    QLabel *BFSImageLabel_1;
    QGroupBox *groupBox1;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_4;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_9;
    QLineEdit *lineEdit_integralframenum2;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_8;
    QLineEdit *lineEdit_delaycorrtime2;
    QPushButton *pushButton_2;
    QCheckBox *checkBox_SerialPortThrough2;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_13;
    QLabel *label_intetime2;
    QHBoxLayout *horizontalLayout_11;
    QComboBox *comboBox_denggonglv2;
    QPushButton *pushButton_denggonglv2;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout_16;
    QLineEdit *lineEdit_BFSExposureime2;
    QPushButton *pushButton_BFS2time;
    QHBoxLayout *horizontalLayout_15;
    QLineEdit *lineEdit_BFSGain2;
    QPushButton *pushButton_BFS2gain;
    QLabel *showImageLabel_2;
    QLabel *BFSImageLabel_2;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1424, 1020);
        MainWindow->setStyleSheet(QString::fromUtf8(""));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        pushButton_sendcommand = new QPushButton(centralWidget);
        pushButton_sendcommand->setObjectName(QString::fromUtf8("pushButton_sendcommand"));
        pushButton_sendcommand->setMinimumSize(QSize(0, 80));

        horizontalLayout_9->addWidget(pushButton_sendcommand);

        pushButton_imagecommand = new QPushButton(centralWidget);
        pushButton_imagecommand->setObjectName(QString::fromUtf8("pushButton_imagecommand"));
        pushButton_imagecommand->setMinimumSize(QSize(0, 80));

        horizontalLayout_9->addWidget(pushButton_imagecommand);


        verticalLayout_5->addLayout(horizontalLayout_9);

        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        label = new QLabel(splitter);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);
        label->setMinimumSize(QSize(60, 0));
        label->setMaximumSize(QSize(60, 16777215));
        label->setSizeIncrement(QSize(300, 0));
        label->setBaseSize(QSize(300, 0));
        splitter->addWidget(label);
        lineEdit_seatnum = new QLineEdit(splitter);
        lineEdit_seatnum->setObjectName(QString::fromUtf8("lineEdit_seatnum"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(lineEdit_seatnum->sizePolicy().hasHeightForWidth());
        lineEdit_seatnum->setSizePolicy(sizePolicy1);
        splitter->addWidget(lineEdit_seatnum);
        label_2 = new QLabel(splitter);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        sizePolicy.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy);
        label_2->setMinimumSize(QSize(60, 0));
        label_2->setMaximumSize(QSize(60, 16777215));
        label_2->setBaseSize(QSize(300, 0));
        splitter->addWidget(label_2);
        lineEdit_total = new QLineEdit(splitter);
        lineEdit_total->setObjectName(QString::fromUtf8("lineEdit_total"));
        sizePolicy1.setHeightForWidth(lineEdit_total->sizePolicy().hasHeightForWidth());
        lineEdit_total->setSizePolicy(sizePolicy1);
        splitter->addWidget(lineEdit_total);

        verticalLayout_5->addWidget(splitter);


        gridLayout->addLayout(verticalLayout_5, 0, 0, 1, 1);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        checkBox_onecamera = new QCheckBox(centralWidget);
        checkBox_onecamera->setObjectName(QString::fromUtf8("checkBox_onecamera"));

        verticalLayout_4->addWidget(checkBox_onecamera);

        checkBox_Continuous = new QCheckBox(centralWidget);
        checkBox_Continuous->setObjectName(QString::fromUtf8("checkBox_Continuous"));

        verticalLayout_4->addWidget(checkBox_Continuous);

        checkBox = new QCheckBox(centralWidget);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));

        verticalLayout_4->addWidget(checkBox);


        gridLayout->addLayout(verticalLayout_4, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 2, 1, 1);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        pushButton_portInit = new QPushButton(centralWidget);
        pushButton_portInit->setObjectName(QString::fromUtf8("pushButton_portInit"));

        verticalLayout_3->addWidget(pushButton_portInit);

        comboBox_PortNO = new QComboBox(centralWidget);
        comboBox_PortNO->setObjectName(QString::fromUtf8("comboBox_PortNO"));

        verticalLayout_3->addWidget(comboBox_PortNO);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setSpacing(6);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        pushButton_yuzhi = new QPushButton(centralWidget);
        pushButton_yuzhi->setObjectName(QString::fromUtf8("pushButton_yuzhi"));

        horizontalLayout_12->addWidget(pushButton_yuzhi);

        lineEdit_yuzhi = new QLineEdit(centralWidget);
        lineEdit_yuzhi->setObjectName(QString::fromUtf8("lineEdit_yuzhi"));

        horizontalLayout_12->addWidget(lineEdit_yuzhi);


        verticalLayout_3->addLayout(horizontalLayout_12);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer_3);


        gridLayout->addLayout(verticalLayout_3, 0, 3, 1, 1);

        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_8 = new QHBoxLayout(groupBox);
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        QFont font;
        font.setPointSize(20);
        label_3->setFont(font);

        verticalLayout->addWidget(label_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_2->addWidget(label_5);

        lineEdit_integralframenum1 = new QLineEdit(groupBox);
        lineEdit_integralframenum1->setObjectName(QString::fromUtf8("lineEdit_integralframenum1"));

        horizontalLayout_2->addWidget(lineEdit_integralframenum1);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout->addWidget(label_7);

        lineEdit_delaycorrtime1 = new QLineEdit(groupBox);
        lineEdit_delaycorrtime1->setObjectName(QString::fromUtf8("lineEdit_delaycorrtime1"));

        horizontalLayout->addWidget(lineEdit_delaycorrtime1);


        verticalLayout->addLayout(horizontalLayout);

        pushButton_1 = new QPushButton(groupBox);
        pushButton_1->setObjectName(QString::fromUtf8("pushButton_1"));

        verticalLayout->addWidget(pushButton_1);

        checkBox_SerialPortThrough1 = new QCheckBox(groupBox);
        checkBox_SerialPortThrough1->setObjectName(QString::fromUtf8("checkBox_SerialPortThrough1"));

        verticalLayout->addWidget(checkBox_SerialPortThrough1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(label_10->sizePolicy().hasHeightForWidth());
        label_10->setSizePolicy(sizePolicy2);

        horizontalLayout_5->addWidget(label_10);

        label_intetime1 = new QLabel(groupBox);
        label_intetime1->setObjectName(QString::fromUtf8("label_intetime1"));
        sizePolicy1.setHeightForWidth(label_intetime1->sizePolicy().hasHeightForWidth());
        label_intetime1->setSizePolicy(sizePolicy1);
        label_intetime1->setStyleSheet(QString::fromUtf8("border: none;\n"
"        border-bottom: 1px solid rgb(145, 145, 145);"));

        horizontalLayout_5->addWidget(label_intetime1);


        verticalLayout->addLayout(horizontalLayout_5);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        comboBox_denggonglv1 = new QComboBox(groupBox);
        comboBox_denggonglv1->addItem(QString());
        comboBox_denggonglv1->addItem(QString());
        comboBox_denggonglv1->addItem(QString());
        comboBox_denggonglv1->addItem(QString());
        comboBox_denggonglv1->setObjectName(QString::fromUtf8("comboBox_denggonglv1"));

        horizontalLayout_10->addWidget(comboBox_denggonglv1);

        pushButton_denggonglv1 = new QPushButton(groupBox);
        pushButton_denggonglv1->setObjectName(QString::fromUtf8("pushButton_denggonglv1"));
        pushButton_denggonglv1->setMinimumSize(QSize(100, 0));
        pushButton_denggonglv1->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_10->addWidget(pushButton_denggonglv1);


        verticalLayout->addLayout(horizontalLayout_10);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(6);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        lineEdit_BFSExposureime1 = new QLineEdit(groupBox);
        lineEdit_BFSExposureime1->setObjectName(QString::fromUtf8("lineEdit_BFSExposureime1"));

        horizontalLayout_14->addWidget(lineEdit_BFSExposureime1);

        pushButton_BFS1time = new QPushButton(groupBox);
        pushButton_BFS1time->setObjectName(QString::fromUtf8("pushButton_BFS1time"));
        pushButton_BFS1time->setMinimumSize(QSize(100, 0));
        pushButton_BFS1time->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_14->addWidget(pushButton_BFS1time);


        verticalLayout->addLayout(horizontalLayout_14);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setSpacing(6);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        lineEdit_BFSGain1 = new QLineEdit(groupBox);
        lineEdit_BFSGain1->setObjectName(QString::fromUtf8("lineEdit_BFSGain1"));

        horizontalLayout_13->addWidget(lineEdit_BFSGain1);

        pushButton_BFS1gain = new QPushButton(groupBox);
        pushButton_BFS1gain->setObjectName(QString::fromUtf8("pushButton_BFS1gain"));
        pushButton_BFS1gain->setMinimumSize(QSize(100, 0));
        pushButton_BFS1gain->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_13->addWidget(pushButton_BFS1gain);


        verticalLayout->addLayout(horizontalLayout_13);


        horizontalLayout_8->addLayout(verticalLayout);

        showImageLabel_1 = new QLabel(groupBox);
        showImageLabel_1->setObjectName(QString::fromUtf8("showImageLabel_1"));
        showImageLabel_1->setMinimumSize(QSize(640, 360));
        showImageLabel_1->setMaximumSize(QSize(640, 360));
        showImageLabel_1->setStyleSheet(QString::fromUtf8("border: 1px solid rgb(145,145, 145);"));

        horizontalLayout_8->addWidget(showImageLabel_1);

        BFSImageLabel_1 = new QLabel(groupBox);
        BFSImageLabel_1->setObjectName(QString::fromUtf8("BFSImageLabel_1"));
        BFSImageLabel_1->setMinimumSize(QSize(450, 360));
        BFSImageLabel_1->setMaximumSize(QSize(450, 360));
        BFSImageLabel_1->setStyleSheet(QString::fromUtf8("border: 1px solid rgb(145,145, 145);"));

        horizontalLayout_8->addWidget(BFSImageLabel_1);


        gridLayout->addWidget(groupBox, 1, 0, 1, 4);

        groupBox1 = new QGroupBox(centralWidget);
        groupBox1->setObjectName(QString::fromUtf8("groupBox1"));
        horizontalLayout_7 = new QHBoxLayout(groupBox1);
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label_4 = new QLabel(groupBox1);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);

        verticalLayout_2->addWidget(label_4);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_9 = new QLabel(groupBox1);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout_3->addWidget(label_9);

        lineEdit_integralframenum2 = new QLineEdit(groupBox1);
        lineEdit_integralframenum2->setObjectName(QString::fromUtf8("lineEdit_integralframenum2"));

        horizontalLayout_3->addWidget(lineEdit_integralframenum2);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_8 = new QLabel(groupBox1);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_4->addWidget(label_8);

        lineEdit_delaycorrtime2 = new QLineEdit(groupBox1);
        lineEdit_delaycorrtime2->setObjectName(QString::fromUtf8("lineEdit_delaycorrtime2"));

        horizontalLayout_4->addWidget(lineEdit_delaycorrtime2);


        verticalLayout_2->addLayout(horizontalLayout_4);

        pushButton_2 = new QPushButton(groupBox1);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        verticalLayout_2->addWidget(pushButton_2);

        checkBox_SerialPortThrough2 = new QCheckBox(groupBox1);
        checkBox_SerialPortThrough2->setObjectName(QString::fromUtf8("checkBox_SerialPortThrough2"));

        verticalLayout_2->addWidget(checkBox_SerialPortThrough2);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_13 = new QLabel(groupBox1);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_6->addWidget(label_13);

        label_intetime2 = new QLabel(groupBox1);
        label_intetime2->setObjectName(QString::fromUtf8("label_intetime2"));
        label_intetime2->setStyleSheet(QString::fromUtf8("border: none;\n"
"        border-bottom: 1px solid rgb(145, 145, 145);"));

        horizontalLayout_6->addWidget(label_intetime2);


        verticalLayout_2->addLayout(horizontalLayout_6);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        comboBox_denggonglv2 = new QComboBox(groupBox1);
        comboBox_denggonglv2->addItem(QString());
        comboBox_denggonglv2->addItem(QString());
        comboBox_denggonglv2->addItem(QString());
        comboBox_denggonglv2->addItem(QString());
        comboBox_denggonglv2->setObjectName(QString::fromUtf8("comboBox_denggonglv2"));

        horizontalLayout_11->addWidget(comboBox_denggonglv2);

        pushButton_denggonglv2 = new QPushButton(groupBox1);
        pushButton_denggonglv2->setObjectName(QString::fromUtf8("pushButton_denggonglv2"));
        pushButton_denggonglv2->setMinimumSize(QSize(100, 0));
        pushButton_denggonglv2->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_11->addWidget(pushButton_denggonglv2);


        verticalLayout_2->addLayout(horizontalLayout_11);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setSpacing(6);
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        lineEdit_BFSExposureime2 = new QLineEdit(groupBox1);
        lineEdit_BFSExposureime2->setObjectName(QString::fromUtf8("lineEdit_BFSExposureime2"));

        horizontalLayout_16->addWidget(lineEdit_BFSExposureime2);

        pushButton_BFS2time = new QPushButton(groupBox1);
        pushButton_BFS2time->setObjectName(QString::fromUtf8("pushButton_BFS2time"));
        pushButton_BFS2time->setMinimumSize(QSize(100, 0));
        pushButton_BFS2time->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_16->addWidget(pushButton_BFS2time);


        verticalLayout_2->addLayout(horizontalLayout_16);

        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setSpacing(6);
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        lineEdit_BFSGain2 = new QLineEdit(groupBox1);
        lineEdit_BFSGain2->setObjectName(QString::fromUtf8("lineEdit_BFSGain2"));

        horizontalLayout_15->addWidget(lineEdit_BFSGain2);

        pushButton_BFS2gain = new QPushButton(groupBox1);
        pushButton_BFS2gain->setObjectName(QString::fromUtf8("pushButton_BFS2gain"));
        pushButton_BFS2gain->setMinimumSize(QSize(100, 0));
        pushButton_BFS2gain->setMaximumSize(QSize(100, 16777215));

        horizontalLayout_15->addWidget(pushButton_BFS2gain);


        verticalLayout_2->addLayout(horizontalLayout_15);


        horizontalLayout_7->addLayout(verticalLayout_2);

        showImageLabel_2 = new QLabel(groupBox1);
        showImageLabel_2->setObjectName(QString::fromUtf8("showImageLabel_2"));
        showImageLabel_2->setMinimumSize(QSize(640, 360));
        showImageLabel_2->setMaximumSize(QSize(640, 360));
        showImageLabel_2->setStyleSheet(QString::fromUtf8("border: 1px solid rgb(145, 145, 145);"));

        horizontalLayout_7->addWidget(showImageLabel_2);

        BFSImageLabel_2 = new QLabel(groupBox1);
        BFSImageLabel_2->setObjectName(QString::fromUtf8("BFSImageLabel_2"));
        BFSImageLabel_2->setMinimumSize(QSize(450, 360));
        BFSImageLabel_2->setMaximumSize(QSize(450, 360));
        BFSImageLabel_2->setStyleSheet(QString::fromUtf8("border: 1px solid rgb(145,145, 145);"));

        horizontalLayout_7->addWidget(BFSImageLabel_2);


        gridLayout->addWidget(groupBox1, 2, 0, 1, 4);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1424, 22));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Two Camera", nullptr));
        pushButton_sendcommand->setText(QApplication::translate("MainWindow", "\346\222\236\347\272\277\344\277\241\345\217\267\346\250\241\346\213\237\346\214\207\344\273\244\345\217\221\351\200\201--", nullptr));
        pushButton_imagecommand->setText(QApplication::translate("MainWindow", "\345\233\276\345\203\217\346\265\213\350\257\225\346\214\207\344\273\244\345\217\221\351\200\201", nullptr));
        label->setText(QApplication::translate("MainWindow", "\345\272\247\344\275\215\345\217\267", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\346\200\273\344\272\272\346\225\260", nullptr));
        checkBox_onecamera->setText(QApplication::translate("MainWindow", "\345\215\225\347\233\270\346\234\272\346\265\213\350\257\225", nullptr));
        checkBox_Continuous->setText(QApplication::translate("MainWindow", "\350\277\236\347\273\255\346\213\215\347\205\247", nullptr));
        checkBox->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\347\275\221\346\240\274", nullptr));
        pushButton_portInit->setText(QApplication::translate("MainWindow", "\344\270\262\345\217\243\345\210\235\345\247\213\345\214\226", nullptr));
        pushButton_yuzhi->setText(QApplication::translate("MainWindow", "\347\247\257\345\210\206\346\227\266\351\227\264\351\230\210\345\200\274\350\256\276\347\275\256", nullptr));
        lineEdit_yuzhi->setText(QString());
        label_3->setText(QApplication::translate("MainWindow", "\351\251\276\351\251\266\344\275\215", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\347\247\257\345\210\206\346\227\266\351\227\264\345\270\247\346\225\260", nullptr));
        lineEdit_integralframenum1->setText(QApplication::translate("MainWindow", "1500", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\345\273\266\346\227\266\345\223\215\345\272\224\346\227\266\351\227\264", nullptr));
        lineEdit_delaycorrtime1->setText(QApplication::translate("MainWindow", "250", nullptr));
        pushButton_1->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256", nullptr));
        checkBox_SerialPortThrough1->setText(QApplication::translate("MainWindow", "\344\270\262\345\217\243\347\233\264\351\200\232", nullptr));
        label_10->setText(QApplication::translate("MainWindow", "\345\271\263\345\235\207\347\247\257\345\210\206\346\227\266\351\227\264", nullptr));
        label_intetime1->setText(QString());
        comboBox_denggonglv1->setItemText(0, QApplication::translate("MainWindow", "7", nullptr));
        comboBox_denggonglv1->setItemText(1, QApplication::translate("MainWindow", "10", nullptr));
        comboBox_denggonglv1->setItemText(2, QApplication::translate("MainWindow", "15", nullptr));
        comboBox_denggonglv1->setItemText(3, QApplication::translate("MainWindow", "20", nullptr));

        pushButton_denggonglv1->setText(QApplication::translate("MainWindow", "\347\201\257\345\212\237\347\216\207\350\256\276\347\275\256", nullptr));
        lineEdit_BFSExposureime1->setText(QApplication::translate("MainWindow", "94", nullptr));
        pushButton_BFS1time->setText(QApplication::translate("MainWindow", "\347\247\257\345\210\206\346\227\266\351\227\264\350\256\276\347\275\256", nullptr));
        lineEdit_BFSGain1->setText(QApplication::translate("MainWindow", "0", nullptr));
        pushButton_BFS1gain->setText(QApplication::translate("MainWindow", "\345\242\236\347\233\212\350\256\276\347\275\256", nullptr));
        showImageLabel_1->setText(QString());
        BFSImageLabel_1->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\345\211\257\351\251\276\351\251\266\344\275\215", nullptr));
        label_9->setText(QApplication::translate("MainWindow", "\347\247\257\345\210\206\346\227\266\351\227\264\345\270\247\346\225\260", nullptr));
        lineEdit_integralframenum2->setText(QApplication::translate("MainWindow", "1500", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "\345\273\266\346\227\266\345\223\215\345\272\224\346\227\266\351\227\264", nullptr));
        lineEdit_delaycorrtime2->setText(QApplication::translate("MainWindow", "250", nullptr));
        pushButton_2->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256", nullptr));
        checkBox_SerialPortThrough2->setText(QApplication::translate("MainWindow", "\344\270\262\345\217\243\347\233\264\351\200\232", nullptr));
        label_13->setText(QApplication::translate("MainWindow", "\345\271\263\345\235\207\347\247\257\345\210\206\346\227\266\351\227\264", nullptr));
        label_intetime2->setText(QString());
        comboBox_denggonglv2->setItemText(0, QApplication::translate("MainWindow", "7", nullptr));
        comboBox_denggonglv2->setItemText(1, QApplication::translate("MainWindow", "10", nullptr));
        comboBox_denggonglv2->setItemText(2, QApplication::translate("MainWindow", "15", nullptr));
        comboBox_denggonglv2->setItemText(3, QApplication::translate("MainWindow", "20", nullptr));

        pushButton_denggonglv2->setText(QApplication::translate("MainWindow", "\347\201\257\345\212\237\347\216\207\350\256\276\347\275\256", nullptr));
        lineEdit_BFSExposureime2->setText(QApplication::translate("MainWindow", "94", nullptr));
        pushButton_BFS2time->setText(QApplication::translate("MainWindow", "\347\247\257\345\210\206\346\227\266\351\227\264\350\256\276\347\275\256", nullptr));
        lineEdit_BFSGain2->setText(QApplication::translate("MainWindow", "0", nullptr));
        pushButton_BFS2gain->setText(QApplication::translate("MainWindow", "\345\242\236\347\233\212\350\256\276\347\275\256", nullptr));
        showImageLabel_2->setText(QString());
        BFSImageLabel_2->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
