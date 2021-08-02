#include <stdio.h>
#include <cmath>
#include <fstream>//ifstream读文件，ofstream写文件，fstream读写文件
#include <algorithm>
#include <vector>
#include <sstream>    //引用stringstream的头文件
#include <string>//文本对象，储存读取的内容
#include <iostream>//屏幕输出cout
#include <cstdlib>//调用system("pause");
#include <chrono>
#include <functional>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace std::chrono_literals;

#define PI 3.14159265
vector<Vec4f> lines;
vector<vector<Point2f>>LinePoints;

class MinimalPublisher : public rclcpp::Node{
  public:
    MinimalPublisher(float i): Node("minimal_publisher"), count_(i)
    {
      publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
      timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }
  private:
    void timer_callback()
    {
      auto message = std_msgs::msg::String();
      message.data = "The Image Need Rotate  " + std::to_string(count_);
      publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    float count_;
};

//Fun splitKeychar2Int  
void splitKeychar2Int(const std::string &s, char delimiter, std::vector<std::string> &v)
{
  std::string::size_type i = 0;
  std::string::size_type j = s.find(delimiter);
  while (j != std::string::npos)
  {
      v.push_back(s.substr(i, j - i));
      i = ++j;
      j = s.find(delimiter, i);
  }
  if (j == std::string::npos)
      v.push_back(s.substr(i, s.length()));
} 

//Fun vector<int>array->cv::Mat
cv::Mat array2mat(vector<int>array, int width, int height) 
{
	cv::Mat src = cv::Mat::zeros(height,width,CV_8U);
	for(int i = 0; i < height; ++i) 
  {
        for(int j = 0; j < width; ++j) 
        {
          src.at<uchar>(height-1-i, j) = 255-array[width*i+j];
        }
  }
  return src;
}

//fld->detect(InputMat, lines);
// Because of some CPU's power strategy, it seems that the first running of
// an algorithm takes much longer. So here we run the algorithm 10 times
// to see the algorithm's processing time with sufficiently warmed-up CPU performance.
cv::Mat FindLineDet(cv::Mat InputMat)
{
  int length_threshold = 10;
  float distance_threshold = 1.41421356f;
  double canny_th1 = 50.0;
  double canny_th2 = 50.0;
  int canny_aperture_size = 3;
  bool do_merge = true;

  Ptr<FastLineDetector> fld = createFastLineDetector(length_threshold,
            distance_threshold, canny_th1, canny_th2, canny_aperture_size,
            do_merge);
  lines.clear();
  for (int run_count = 0; run_count < 5; run_count++) 
  {
    double freq = getTickFrequency();
    lines.clear();
    int64 start = getTickCount();
    fld->detect(InputMat, lines);
    double duration_ms = double(getTickCount() - start) * 1000 / freq;
  }
  Mat line_image_fld(InputMat);
  fld->drawSegments(line_image_fld, lines);
  return line_image_fld;
}

// param 0  Distance   The distance of the two points
// param 1  SlopeK     The Slope of the two points
// param 2-5  Points   The  two points
vector<vector<float>> DistanceCalculate()
{
  vector<vector<float>> LineIncludeDis;
  for(int i=0;i<lines.size();i++)
  {
    cv::Vec4f lws = lines[i];
    float Distance = sqrt( (lws[3]-lws[1])*(lws[3]-lws[1]) +(lws[2]-lws[0])*(lws[2]-lws[0]) );  //caculate the distance based on the two points
    float SlopeK = (lws[3]-lws[1])/(lws[2]-lws[0]); //caculate the Slope based on the two points
    vector<float> SigleLineInf;
    SigleLineInf.push_back(Distance);
    SigleLineInf.push_back(SlopeK);
    LineIncludeDis.push_back(SigleLineInf);
  }
  sort(LineIncludeDis.begin(),LineIncludeDis.end());
  cout<<"The Number of Detected Line is: "<<LineIncludeDis.size()<<endl;
  vector<float> LineIncludeSlope;
  for(int i = LineIncludeDis.size()-1;i>int(LineIncludeDis.size()*0.8);i--)
  {
    LineIncludeSlope.push_back(LineIncludeDis[i][1]); //Save The slope 
  }
  sort(LineIncludeSlope.begin(),LineIncludeSlope.end());
  cout<<"The Number of Main Slpoe is: "<<LineIncludeSlope.size()<<endl;

  float NumOfCount = 0,MinSlope = LineIncludeSlope[0];
  vector<vector<float>> Histogram;
  vector<float> SigleHis;
  for(int i=0;i<LineIncludeSlope.size();i++)
  {
    if(LineIncludeSlope[i]<(0.1+MinSlope))
    {
      NumOfCount++;
    }
    else
    {
      SigleHis.clear();
      SigleHis.push_back(NumOfCount);     //Histogram number
      SigleHis.push_back(0.05+MinSlope);  //midle Slope
      Histogram.push_back(SigleHis);
      cout<<" The Slope  "<<0.05+MinSlope<<" +_0.1 ,The Number of Statistiacl: "<<NumOfCount<<endl;
      NumOfCount = 1;
      MinSlope = LineIncludeSlope[i];
    }
  }
  cout<<"-------The nub of Histogram is ----- "<<Histogram.size()<<endl;

  sort(Histogram.begin(),Histogram.end());
  cout<<"  Slope:  "<<Histogram[Histogram.size()-1][1]<<" Number:  "<<Histogram[Histogram.size()-1][0]<<endl;
  return Histogram;
}

//Fun Rotate the image based on X-axis
void Rotate(const Mat &srcImage, Mat &destImage, double angle)
{
	Point2f center(srcImage.cols / 2, srcImage.rows / 2);//中心
	Mat M = getRotationMatrix2D(center, angle, 1);//计算旋转的仿射变换矩阵 
	warpAffine(srcImage, destImage, M, Size(srcImage.cols, srcImage.rows));//仿射变换  
	circle(destImage, center, 2, Scalar(255, 0, 0));
}


//Main fun 
//Read The txt data Get Image hight width and datearray
//Line detect and slope caculate
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher = rclcpp::Node::create_publisher<std_msgs::msg::String>("topic", 10);

  int Count=0;
  int height,Width,TransDate;
  vector<string> SingelNumberVect;
  vector<int>DataArray;

  ifstream in("/home/mi/Lws-Ros2_pkg/Test2_ws/data/map_data4.txt");
  string line;

  if(!in) // No such File
    cout <<"no such file" << endl;

  else
  {
    while (getline (in, line))
    {
      Count++;
      // Read Image height 
      if(Count==2)
      {
        SingelNumberVect.clear();
        splitKeychar2Int(line, ':', SingelNumberVect); 
        stringstream ss;
	      ss << SingelNumberVect[1];
	      ss >> height;
        cout << height << endl;
      }
      // Read Image Width 
      if(Count==3)
      {
        SingelNumberVect.clear();
        splitKeychar2Int(line, ':', SingelNumberVect); 
        stringstream ss;
	      ss << SingelNumberVect[1];
	      ss >> Width;
        cout << Width << endl;
      }
        // Read Image Data Array 
      if(Count==11)
      {
        SingelNumberVect.clear();
        splitKeychar2Int(line, ':', SingelNumberVect); 
        stringstream ss;
        ss.str(SingelNumberVect[1]);
        DataArray.clear();
        while(1)
        {
          ss>>TransDate;
          DataArray.push_back(TransDate);
          if(ss.fail())
          {
            break;
          }
        }
      } 
    }
  }
  cv::Mat lwsImg = array2mat(DataArray, Width, height);       //Get the Image based on Image Data Array
  //Rotate(lwsImg, lwsImg, 15);                                //Test 
  cv::Mat MatAfterLineDec = FindLineDet(lwsImg);              //Line Detect Fun
  vector<vector<float>>ResultArray = DistanceCalculate();     //Get the main slope 

  Mat destImage;
	double angle = atan (ResultArray[ResultArray.size()-1][1]) * 180 / PI;          //get the angle based on main slope
  cout<<"angle is "<<angle<<endl;
	Rotate(MatAfterLineDec, destImage, angle);

  rclcpp::spin(std::make_shared<MinimalPublisher>(angle));
  imshow("Initial Image", lwsImg);
  imshow("FLD result", MatAfterLineDec);
  imshow("Image After Rotate", destImage);
  cv::waitKey(0);

  rclcpp::shutdown();
  return 0;
  }