#include "bfsthread1.h"
#include "globalvalue.h"


BFSThread1::BFSThread1()
{
    //system = System::GetInstance();

    // Retrieve list of cameras from the system
    //camList = system->GetCameras();

    unsigned int numCameras = GlobalValue::camList.GetSize();

    // Finish if there are no cameras
    /*if (numCameras == 0)
    {
        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();
    }*/

    //const triggerType chosenTrigger = SOFTWARE;
    //chosenTrigger = HARDWARE;     //这里选择了硬件触发
    m_number = 3;                               //多帧采集时，一个信号采集图像的个数
    exposureTimeToSet = 94.0;           //设置的曝光值，单位us
    gainValueToSet = 1.5;           //设置的曝光值，单位us

    GlobalValue::BFSimageData10.resize(640*512);
    GlobalValue::BFSimageData10.fill(0x99,640*512);
    GlobalValue::BFSimageData11.resize(640*512);
    GlobalValue::BFSimageData11.fill(0x99,640*512);

}

void BFSThread1::init()
{
    if(GlobalValue::BFSFlag_change)
    {
        pCam = GlobalValue::camList.GetByIndex(1);
    }
    else{
        pCam = GlobalValue::camList.GetByIndex(0);
    }
    pCam->Init();
    ConfigureUserSet(pCam);
}
BFSThread1::~BFSThread1()
{
    // End acquisition
    pCam->EndAcquisition();

    // Reset trigger
    //ResetTrigger(nodeMap);

    // Deinitialize camera
    pCam->DeInit();

    //camList.Clear();
    // Release system

    //system->ReleaseInstance();
}

void BFSThread1::run()
{

    // Retrieve GenICam nodemap
    INodeMap & nodeMap = pCam->GetNodeMap();

    // Configure trigger 配置触发模式
    //ConfigureTrigger(nodeMap);
    //ConfigureAcquisitionFrameRate(nodeMap,10);

    // Set acquisition mode to continuous
    CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
    if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
    {
        cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
    }

    CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
    if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
    {
        cout << "Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting..." << endl << endl;
    }

    int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

    cout << "Acquisition mode set to continuous..." << endl;

    // Begin acquiring images
    pCam->BeginAcquisition();

    cout << "Acquiring images..." << endl;

    // Retrieve, convert, and save images
    //const int unsigned k_numImages = 100;

    //for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)



    /*CFloatPtr ptrExposureTime = nodeMap.GetNode("TriggerDelay");
    if (!IsAvailable(ptrExposureTime) || !IsWritable(ptrExposureTime))
    {
        cout << "Unable to set exposure time. Aborting..." << endl << endl;
    }

    ptrExposureTime->SetValue(0);*/

    unsigned int imageCnt = 0;
    while(1)
    {
        try
        {
            // Retrieve the next image from the trigger
            //result = result | GrabNextImageByTrigger(nodeMap, pCam);

            // Retrieve the next received image
           ImagePtr pResultImage = pCam->GetNextImage();

           imageCnt++;

             if (pResultImage->IsIncomplete())
            {
                sendyanshi();
                cout<<"BFSimage1 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;

                cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl << endl;
            }
            else
            {
                //cout<<"BFSimage1 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s"<<endl;


                // Print image information
                cout << "Grabbed image " << imageCnt << ", width = " << pResultImage->GetWidth() << ", height = " << pResultImage->GetHeight() << endl;

                // Convert image to mono 8
                ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);

                // Create a unique filename
                ostringstream filename;

                filename << "Trigger-";
                /*(if (deviceSerialNumber != "")
                {
                    filename << deviceSerialNumber.c_str() << "-";
                }*/
                filename << imageCnt << ".jpg";

                // Save image
//                convertedImage->Save(filename.str().c_str());

                //cout << "Image saved at " << filename.str() << endl;

                switch(GlobalValue::BFS1_imagenum1)
                {
                    case 0:
                    {GlobalValue::BFS1_imagenum1++;
                    qDebug()<<"BFSimage --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s";
                    if(GlobalValue::timebfs.elapsed()>83)
                    {
                        //sendyanshi();
                    }
                    memcpy(GlobalValue::BFSimageData10.data(),(uchar *)convertedImage->GetData(),640*512);
                    sendBFSimage10();
                    break;
                    }
                    case 1:
                    {
                    GlobalValue::BFS1_imagenum1++;
                    //qDebug()<<"BFSimage1 --------------------------------------------" << GlobalValue::timebfs.elapsed() / 1000.0 << "s";
                    memcpy(GlobalValue::BFSimageData11.data(),(uchar *)convertedImage->GetData(),640*512);
                    sendBFSimage11();
                    break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }

            // Release image
            pResultImage->Release();

            cout << endl;
        }
        catch (Spinnaker::Exception &e)
        {
            cout << "Error: " << e.what() << endl;
        }
    }


}

int BFSThread1::ResetTrigger(INodeMap & nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn trigger mode back off
        //
        // *** NOTES ***
        // Once all images have been captured, turn trigger mode back off to
        // restore the camera to a clean state.
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Non-fatal error..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Non-fatal error..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << "Trigger mode disabled..." << endl << endl;
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

/*int BFSThread1::ConfigureTrigger(INodeMap & nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING TRIGGER ***" << endl << endl;

    if (chosenTrigger == SOFTWARE)
    {
        cout << "Software trigger chosen..." << endl;
    }
    else if (chosenTrigger == HARDWARE)
    {
        cout << "Hardware trigger chosen..." << endl;
    }

    try
    {
        ////在配置触发的时候确定触发模式是off
        // Ensure trigger mode off
        //
        // *** NOTES ***
        // The trigger must be disabled in order to configure whether the source
        // is software or hardware.


        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");// 1,选择设置节点********************

        if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");//2，选择设置值**************

        if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());//3，设置生效*****************************


        cout << "Trigger mode disabled..." << endl;




        ////选择触发源
        // Select trigger source

        // *** NOTES ***
        // The trigger source must be set to hardware or software while trigger
        // mode is off.
        //
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");  // 1, 选择设置节点********************

        if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        if (chosenTrigger == SOFTWARE)                                                    //全局变量chosenTrigger选择为软触发时*******************
        {
            // Set trigger mode to software
            CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");// 2, 选择设置值********************
            if (!IsAvailable(ptrTriggerSourceSoftware) || !IsReadable(ptrTriggerSourceSoftware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());//3，设置生效*****************************

            cout << "Trigger source set to software..." << endl;
        }
        else if (chosenTrigger == HARDWARE)                                            //全局变量chosenTrigger选择为硬件触发时*******************
        {
            // Set trigger mode to hardware ('Line0')
            CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line0"); // 2, 选择设置值********************
            if (!IsAvailable(ptrTriggerSourceHardware) || !IsReadable(ptrTriggerSourceHardware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue()); //3，设置生效*****************************

            cout << "Trigger source set to hardware..." << endl;





            //设置触发采集模式

            CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");  //  1, 选择设置节点********************

            if (!IsAvailable(ptrTriggerSelector) || !IsWritable(ptrTriggerSelector))
            {
                cout << "Unable to set trigger selector (node retrieval). Aborting..." << endl;
                return -1;
            }

            CEnumEntryPtr ptrTriggerSelectorFrameBurstStart = ptrTriggerSelector->GetEntryByName("FrameStart");//  2, 选择设置值********************
            if (!IsAvailable(ptrTriggerSelectorFrameBurstStart) || !IsReadable(ptrTriggerSelectorFrameBurstStart))
            {
                cout << "Unable to set Trigger Selector FrameBurstStart (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSelector->SetIntValue(ptrTriggerSelectorFrameBurstStart->GetValue());//  3, 设置生效********************

            cout << "Trigger Selector FrameStart..." << endl;





            //设置一次触发采集多少张图

            CIntegerPtr ptrAcquisitionBurstFrameCount = nodeMap.GetNode("AcquisitionBurstFrameCount"); //1, 选择设置节点********************
            if (!IsAvailable(ptrAcquisitionBurstFrameCount) || !IsWritable(ptrAcquisitionBurstFrameCount))
            {
                return -1;
            }

            ptrAcquisitionBurstFrameCount->SetValue(m_number);//2，设置生效**********************************************

        }


        //设置好相关参数后，开启触发模式
        // Turn trigger mode on
        //
        // *** LATER ***
        // Once the appropriate trigger source has been set, turn trigger mode
        // on in order to retrieve images using the trigger.
        //

        CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");//  2, 选择设置值********************
        if (!IsAvailable(ptrTriggerModeOn) || !IsReadable(ptrTriggerModeOn))
        {
            cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());//3，设置生效**********************************************

        // TODO: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on

        cout << "Trigger mode turned back on..." << endl << endl;
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}*/

int BFSThread1::ConfigureExposure(INodeMap& nodeMap,double exposureTimeToSet)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING EXPOSURE ***" << endl << endl;

    try
    {
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (!IsAvailable(ptrExposureAuto) || !IsWritable(ptrExposureAuto))
        {
            cout << "Unable to disable automatic exposure (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
        if (!IsAvailable(ptrExposureAutoOff) || !IsReadable(ptrExposureAutoOff))
        {
            cout << "Unable to disable automatic exposure (enum entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());

        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsAvailable(ptrExposureTime) || !IsWritable(ptrExposureTime))
        {
            cout << "Unable to set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double exposureTimeMax = ptrExposureTime->GetMax();

        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }

        ptrExposureTime->SetValue(exposureTimeToSet);

        cout << std::fixed << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int BFSThread1::ConfigureExposurenodeMap(double exposureTimeToSet)
{

    // Retrieve GenICam nodemap
    INodeMap & nodeMap = pCam->GetNodeMap();

    // Configure trigger 配置触发模式
    int result = ConfigureExposure(nodeMap,exposureTimeToSet);

    return result;
}

int BFSThread1::ConfigureGain(INodeMap& nodeMap,double GainToSet)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING GAIN ***" << endl << endl;

    try
    {

        CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
        if (!IsAvailable(ptrGainAuto) || !IsWritable(ptrGainAuto))
        {
            cout << "Unable to disable automatic exposure (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrGainAutoOff = ptrGainAuto->GetEntryByName("Off");
        if (!IsAvailable(ptrGainAutoOff) || !IsReadable(ptrGainAutoOff))
        {
            cout << "Unable to disable automatic exposure (enum entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrGainAuto->SetIntValue(ptrGainAutoOff->GetValue());

        CFloatPtr ptrGain = nodeMap.GetNode("Gain");
        if (!IsAvailable(ptrGain) || !IsWritable(ptrGain))
        {
            cout << "Unable to set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double GainMax = ptrGain->GetMax();
        //double GainToSet = 10.0;

        if (GainToSet > GainMax)
        {
            GainToSet = GainMax;
        }

        ptrGain->SetValue(GainToSet);

    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int BFSThread1::ConfigureGainnodeMap(double GainToSet)
{

    // Retrieve GenICam nodemap
    INodeMap & nodeMap = pCam->GetNodeMap();

    // Configure trigger 配置触发模式
    int result = ConfigureGain(nodeMap,GainToSet);

    return result;
}

int BFSThread1::ConfigureUserSet(CameraPtr pCam)
{
    int result = 0;
    int err = 0;
    try
    {


        // Retrieve TL device nodemap and print device information
        INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        //result = PrintDeviceInfo(nodeMapTLDevice);

        // Retrieve GenICam nodemap
        INodeMap & nodeMap = pCam->GetNodeMap();


        CEnumerationPtr ptrUserSetSelector = nodeMap.GetNode("UserSetSelector");// 1,选择设置节点********************
        CEnumEntryPtr UserSet1 = ptrUserSetSelector->GetEntryByName("UserSet1");//2，选择设置值**************
        ptrUserSetSelector->SetIntValue(UserSet1->GetValue());//3，设置生效*****************************


        CCommandPtr ptrUserSetLoadCommand = nodeMap.GetNode("UserSetLoad");
        ptrUserSetLoadCommand->Execute();

    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
    return result;
}

int BFSThread1::ConfigureAcquisitionFrameRate(INodeMap& nodeMap,double AcquisitionFrameRateToSet)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING AcquisitionFrameRate ***" << endl << endl;

    try
    {

        /*CEnumerationPtr ptrAcquisitionFrameRateEnabled = nodeMap.GetNode("AcquisitionFrameRateEnabled");
        if (!IsAvailable(ptrAcquisitionFrameRateEnabled) || !IsWritable(ptrAcquisitionFrameRateEnabled))
        {
            cout << "Unable to disable automatic AcquisitionFrameRateEnabled (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrAcquisitionFrameRateEnabledOn = ptrAcquisitionFrameRateEnabled->GetEntryByName("On");
        if (!IsAvailable(ptrAcquisitionFrameRateEnabledOn) || !IsReadable(ptrAcquisitionFrameRateEnabledOn))
        {
            cout << "Unable to disable automatic exposure (enum entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrAcquisitionFrameRateEnabled->SetIntValue(ptrAcquisitionFrameRateEnabledOn->GetValue());*/



        CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
        if (!IsAvailable(ptrAcquisitionFrameRate) || !IsWritable(ptrAcquisitionFrameRate))
        {
            cout << "Unable to set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double AcquisitionFrameRateMax = ptrAcquisitionFrameRate->GetMax();
        //double GainToSet = 10.0;

        if (AcquisitionFrameRateToSet > AcquisitionFrameRateMax)
        {
            AcquisitionFrameRateToSet = AcquisitionFrameRateMax;
        }

        ptrAcquisitionFrameRate->SetValue(AcquisitionFrameRateToSet);

    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}
