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

void ReadTxtFlie(int &height,int &Width, vector<int> &DataArray);
void splitKeychar2Int(const std::string &s, char delimiter, std::vector<std::string> &v);
vector<vector<float>> DistanceCalculate(vector<Vec4f> &UsefulLine);
cv::Mat array2mat(vector<int>array, int width, int height);

#define PI 3.14159265
vector<Vec4f> lines;

void ReadTxtFlie(int &height,int &Width, vector<int> &DataArray)
{
  ifstream in("/home/mi/Lws-Ros2_pkg/Test2_ws/data/map_data4.txt");
  string line;
  int Count = 0,TransDate =0;
  vector<string> SingelNumberVect;
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
      }
      // Read Image Width 
      if(Count==3)
      {
        SingelNumberVect.clear();
        splitKeychar2Int(line, ':', SingelNumberVect); 
        stringstream ss;
	      ss << SingelNumberVect[1];
	      ss >> Width;
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
}
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

void FindLineDet(cv::Mat &InitalImage,
                cv::Mat &MatAfterLineDecAll,
                cv::Mat &MatAfterLineDecfilter,
                vector<Vec4f> &lines,
                vector<Vec4f> &UsefulLine)
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
    lines.clear();
    fld->detect(InitalImage, lines);
    // cout<<"lines  "<<lines.size()<<endl;
  }

  MatAfterLineDecAll = InitalImage;
  MatAfterLineDecfilter = InitalImage;
  vector<vector<float>>DistanceArray;
  for(size_t i=0;i<lines.size();i++)
  {
    vector<float>Cur_dis;
    cv::Vec4f Cur_line =lines[i];
    float Distance = sqrt( (Cur_line[3]-Cur_line[1])*(Cur_line[3]-Cur_line[1]) +(Cur_line[2]-Cur_line[0])*(Cur_line[2]-Cur_line[0]) );
    Cur_dis.push_back(Distance);
    Cur_dis.push_back(i);
    DistanceArray.push_back(Cur_dis);
  }
  sort(DistanceArray.begin(),DistanceArray.end());
  for(size_t i = DistanceArray.size()-1;i>int(DistanceArray.size()*0.8);i--)
  {
        cv::Vec4f Cur_line =lines[DistanceArray[i][1]];
        UsefulLine.push_back(Cur_line);
  }
  cout<<"lines  "<<lines.size()<<"  UsefulLine "<<UsefulLine.size()<<endl;
  fld->drawSegments(MatAfterLineDecAll, lines);
  fld->drawSegments(MatAfterLineDecfilter, UsefulLine);
}

vector<vector<float>> DistanceCalculate(vector<Vec4f> &lines)
{
  vector<vector<float>> LineIncludeDis;
  for(size_t i=0;i<lines.size();i++)
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
  for(size_t i=0;i < LineIncludeSlope.size();i++)
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
  float New_clos = srcImage.cols/cos(angle*PI/180);
  float New_rows = srcImage.rows*sin(angle*PI/180);
	Point2f center(New_clos/2, New_rows/2);//中心
	Mat M = getRotationMatrix2D(center, angle, 1);//计算旋转的仿射变换矩阵 
	warpAffine(srcImage, destImage, M, Size(srcImage.cols*1.2, srcImage.rows*1.2));//仿射变换  
	circle(destImage, center, 2, Scalar(255, 0, 0));
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("talker");
  auto chatter_pub = node->create_publisher<std_msgs::msg::String>("chatter", 10);

  int height = 0, Width = 0;
  vector<int>DataArray;
  ReadTxtFlie(height,Width,DataArray);
  cout<<"height  "<<height<<"  Width  "<<Width<<" DataArray "<<DataArray.size()<<endl;

  

  cv::Mat InitalImage = array2mat(DataArray, Width, height);       //Get the Image based on Image Data Array
  // Rotate(lwsImg, lwsImg, 45);                                //Test 

  vector<Vec4f> lines,UsefulLine;
  cv::Mat MatAfterLineDecAll,MatAfterLineDecfilter;
  FindLineDet(InitalImage,MatAfterLineDecAll,MatAfterLineDecfilter,lines,UsefulLine);              //Line Detect Fun

  vector<vector<float>>ResultArray = DistanceCalculate(lines);     //Get the main slope 
  Mat ImageAfterRotate;
	double angle = atan (ResultArray[ResultArray.size()-1][1]) * 180 / PI;          //get the angle based on main slope
  cout<<"angle is "<<angle<<endl;
	Rotate(MatAfterLineDecfilter, ImageAfterRotate, angle);

  imshow("Initial Image", InitalImage);
  imshow("FLD result with all line", MatAfterLineDecAll);
  imshow("FLD result with filter line", MatAfterLineDecfilter);
  imshow("Image After Rotate", ImageAfterRotate);
  cv::waitKey(0);

  rclcpp::WallRate loop_rate(2);
  auto message = std_msgs::msg::String();
  while (rclcpp::ok()) 
  {
    message.data = "The Image Need Rotate  " + std::to_string(angle);
    chatter_pub->publish(message);
    rclcpp::spin_some(node);
    loop_rate.sleep();
  }
  return 0;
}