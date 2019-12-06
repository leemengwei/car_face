#include "processthread.h"

ProcessThread::ProcessThread()
{
    SuanfaInit();
}
ProcessThread::~ProcessThread()
{Py_Finalize();}
Mat ProcessThread::QImageToMat(QImage* image)
{
    cv::Mat mat;
    switch (image->format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image->height(), image->width(), CV_8UC4, (void*)image->constBits(), image->bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image->height(), image->width(), CV_8UC3, (void*)image->constBits(), image->bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:

        mat = cv::Mat(image->height(), image->width(), CV_8UC1, (void*)image->constBits(), image->bytesPerLine());
        break;
    }
    return mat;
}

Mat ProcessThread::convertTo3Channels(const Mat& binImg)
{     Mat three_channel = Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
      vector<Mat> channels;
        for (int i=0;i<3;i++)
        {         channels.push_back(binImg);     }
          merge(channels,three_channel);
            return three_channel;
}

bool ProcessThread::SuanfaInit()
{
    Py_Initialize();//初始化Python
    import_array();
    //先准备python环境路径:
    PyRun_SimpleString("import sys,os");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('./front_position_algorithm')");
    PyRun_SimpleString("sys.path.append('./object_detection_network')");
    PyRun_SimpleString("sys.path.append('./spatial_in_seat_network')");
    PyRun_SimpleString("sys.path.append('./side_position_algorithm')");
    cout<<"hello from C+"<<endl;
    PyRun_SimpleString("print('hello from python')");
    PyRun_SimpleString("print(os.getcwd())");
    PyRun_SimpleString("print(sys.path)");
    PyRun_SimpleString("sys.stdout.flush()");

    PyObject *A_root_dir = PyBytes_FromString("./");
    PyObject *A_ArgDir = PyTuple_New(1);    //准备空元组包
    PyTuple_SetItem(A_ArgDir, 0, A_root_dir);   //填值

    //调用Python算法总共包含一下4个步骤，其中包含了4个API，约定A表示司机侧，B表示另一侧。
    //STEP1（这个不是API, 是必要步骤）:
    //先import python模块
    //PyObject* simple_function = PyImport_ImportModule("simple");
    PyObject * helloModule = PyImport_ImportModule("helo");
    ThreadsModule = PyImport_ImportModule("threads_start");
    //PyObject* AModule = PyImport_ImportModule("A");
    if (!ThreadsModule)
    {
        PyErr_Print();
        cout << "[ERROR] Python get module failed." << endl;
        return 0;
    }
    cout << "[INFO] Python get module succeed." << endl;
    //再从模块import得到类或者函数
    PyObject* workers_cluster_class = PyObject_GetAttrString(ThreadsModule, "workers_cluster");

    if (!workers_cluster_class)
    {
        PyErr_Print();
        cout << "[ERROR] Can't find class " << endl;
        return 0;
    }
    cout << "[INFO] Get class succeed." << endl;

    //STEP2（API1加载模型）:
    //实例化, 此时完成初始化并加载模型，需要几秒种:
    workers_instance = PyObject_CallObject(workers_cluster_class, NULL);
    if (!workers_instance)
    {
        PyErr_Print();
        printf("Can't create instance./n");
        return -1;
    }

}

void ProcessThread::run()
{
    qDebug()<<"======================================process prepare============================================";
        QImage* image1 = new QImage((uchar *)GlobalValue::imageData1.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image11 = QImageToMat(image1);
        Mat A1_image = convertTo3Channels(image11);
        npy_intp A1_Dims[3] = { A1_image.rows, A1_image.cols, A1_image.channels() };
        PyObject *A1_PyArray = PyArray_SimpleNewFromData(3, A1_Dims, NPY_UINT8, A1_image.data);


        QImage* image2 = new QImage((uchar *)GlobalValue::imageData2.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image22 = QImageToMat(image2);
        Mat A2_image = convertTo3Channels(image22);
        npy_intp A2_Dims[3] = { A2_image.rows, A2_image.cols, A2_image.channels() };
        PyObject *A2_PyArray = PyArray_SimpleNewFromData(3, A2_Dims, NPY_UINT8, A2_image.data);


        QImage* image3 = new QImage((uchar *)GlobalValue::imageData3.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image33 = QImageToMat(image3);
        Mat A3_image = convertTo3Channels(image33);
        npy_intp A3_Dims[3] = { A3_image.rows, A3_image.cols, A3_image.channels() };
        PyObject *A3_PyArray = PyArray_SimpleNewFromData(3, A3_Dims, NPY_UINT8, A3_image.data);


        QImage* image4 = new QImage((uchar *)GlobalValue::imageData4.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image44 = QImageToMat(image4);
        Mat A4_image = convertTo3Channels(image44);
        npy_intp A4_Dims[3] = { A4_image.rows, A4_image.cols, A4_image.channels() };
        PyObject *B1_PyArray = PyArray_SimpleNewFromData(3, A4_Dims, NPY_UINT8, A4_image.data);


        QImage* image5 = new QImage((uchar *)GlobalValue::imageData5.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image55 = QImageToMat(image5);
        Mat A5_image = convertTo3Channels(image55);
        npy_intp A5_Dims[3] = { A5_image.rows, A5_image.cols, A5_image.channels() };
        PyObject *B2_PyArray = PyArray_SimpleNewFromData(3, A5_Dims, NPY_UINT8, A5_image.data);


        QImage* image6 = new QImage((uchar *)GlobalValue::imageData6.data(), VIDEO_WIDTH, VIDEO_HEIGHT, QImage::Format_Indexed8);
        Mat image66 = QImageToMat(image6);
        Mat A6_image = convertTo3Channels(image66);
        npy_intp A6_Dims[3] = { A6_image.rows, A6_image.cols, A6_image.channels() };
        PyObject *B3_PyArray = PyArray_SimpleNewFromData(3, A6_Dims, NPY_UINT8, A6_image.data);

        QImage* image7 = new QImage((uchar *)GlobalValue::BFSimageData00.data(), 640, 512, QImage::Format_Indexed8);
        Mat image77 = QImageToMat(image7);
        Mat A7_image = convertTo3Channels(image77);
        npy_intp A7_Dims[3] = { A7_image.rows, A7_image.cols, A7_image.channels() };
        PyObject *BFS1_PyArray = PyArray_SimpleNewFromData(3, A7_Dims, NPY_UINT8, A7_image.data);

        QImage* image8 = new QImage((uchar *)GlobalValue::BFSimageData10.data(), 640, 512, QImage::Format_Indexed8);
        Mat image88 = QImageToMat(image8);
        Mat A8_image = convertTo3Channels(image88);
        npy_intp A8_Dims[3] = { A8_image.rows, A8_image.cols, A8_image.channels() };
        PyObject *BFS2_PyArray = PyArray_SimpleNewFromData(3, A8_Dims, NPY_UINT8, A8_image.data);

        PyObject *A123B123BFS_ArgImg = PyTuple_New(9);    //准备空元组包
        PyTuple_SetItem(A123B123BFS_ArgImg, 0, workers_instance);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 1, A1_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 2, A2_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 3, A3_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 4, B1_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 5, B2_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 6, B3_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 7, BFS1_PyArray);   //填值
        PyTuple_SetItem(A123B123BFS_ArgImg, 8, BFS2_PyArray);   //填值
        PyObject *MReturn4;
qDebug()<<"======================================process start============================================";
        MReturn4 = PyObject_CallMethod(ThreadsModule, "algorithm_detection_and_merge", "O", A123B123BFS_ArgImg);
qDebug()<<"======================================process output============================================";
        jieguo(MReturn4);

}

