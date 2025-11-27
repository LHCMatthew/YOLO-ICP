#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <json/json.h>
#include <fstream>
#include <filesystem>

#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h> // For explicit point-to-plane
#include <pcl/registration/gicp.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "jly_goicp.h"
#include "ConfigMap.hpp"

using namespace std::chrono_literals;

#define GOICP false
#define VISUALIZE false
#define initial_angle_guess false
#define USING_METHOD_1 false

std::vector<std::string> classNames;


auto timeStart = std::chrono::high_resolution_clock::now();

int sec()
{
	auto done = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::seconds>(done - timeStart).count();
}


void visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string cloud_name)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, cloud_name);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cloud_name);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
}

void visualizeICPCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar, std::string cloud_name_tar, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sou, std::string cloud_name_sou)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud_tar, *cloud_tar_rgb);
  for (auto& point : cloud_tar_rgb->points)
  {
    point.r = 0;
    point.g = 255;
    point.b = 0;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sou_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud_sou, *cloud_sou_rgb);
  for (auto& point : cloud_sou_rgb->points)
  {
    point.r = 255;
    point.g = 0;
    point.b = 0;
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rg1(cloud_tar_rgb);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_tar_rgb, rg1, cloud_name_tar);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name_tar);

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rg2(cloud_sou_rgb);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sou_rgb, rg2, cloud_name_sou);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name_sou);
      
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr depth_to_point_cloud(cv::Mat& depth_image)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < depth_image.rows; ++i)
  {
      for (int j = 0; j < depth_image.cols; ++j)
      {
          double z = depth_image.at<ushort>(i, j) * 0.1;
          if (z > 0)
          {
              double x = (j - 325.2611082792282) * z / 572.4114;
              double y = (i - 242.04899594187737) * z / 573.57043;
              cloud->points.emplace_back(x, y, z);
              // std::cout << "Point: (" << x << ", " << y << ", " << z << ")\n";
          }
      }
  }
  return cloud;
}

void init_className()
{
  classNames.push_back("ape");
  classNames.push_back("obj 2");
  classNames.push_back("obj 3");
  classNames.push_back("obj 4");
  classNames.push_back("watering can");
  classNames.push_back("cat toy");
  classNames.push_back("obj 7");
  classNames.push_back("driller");
  classNames.push_back("duck toy");
  classNames.push_back("egg box");
  classNames.push_back("glue");
  classNames.push_back("holepuncher");
  classNames.push_back("obj 13");
  classNames.push_back("obj 14");
  classNames.push_back("obj 15");
}

Eigen::Vector4f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    Eigen::Vector4f centroid(0.0, 0.0, 0.0, 0.0);
    for (const auto& point : cloud->points)
    {
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    centroid /= static_cast<float>(cloud->points.size());
    return centroid;
}

struct ScaleParams
{
    double minX = 0, maxX = 0;
    double minY = 0, maxY = 0;
    double minZ = 0, maxZ = 0;
};

ScaleParams loadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr data_cloud, POINT3D ** pModel, POINT3D ** pData, int & Nm, int & Nd)
{
  Nm = model_cloud->points.size();
  Nd = data_cloud->points.size();
  double minX = std::numeric_limits<double>::max();
  double minY = std::numeric_limits<double>::max();
  double minZ = std::numeric_limits<double>::max();
  double maxX = -std::numeric_limits<double>::max();
  double maxY = -std::numeric_limits<double>::max();
  double maxZ = -std::numeric_limits<double>::max();

  for(int i = 0; i < Nm; i++)
  {
    if (model_cloud->points[i].x < minX) minX = model_cloud->points[i].x;
    if (model_cloud->points[i].y < minY) minY = model_cloud->points[i].y;
    if (model_cloud->points[i].z < minZ) minZ = model_cloud->points[i].z;
    if (model_cloud->points[i].x > maxX) maxX = model_cloud->points[i].x;
    if (model_cloud->points[i].y > maxY) maxY = model_cloud->points[i].y;
    if (model_cloud->points[i].z > maxZ) maxZ = model_cloud->points[i].z;
  }

  for (int i = 0; i < Nd; i++)
  {
    if (data_cloud->points[i].x < minX) minX = data_cloud->points[i].x;
    if (data_cloud->points[i].y < minY) minY = data_cloud->points[i].y;
    if (data_cloud->points[i].z < minZ) minZ = data_cloud->points[i].z;
    if (data_cloud->points[i].x > maxX) maxX = data_cloud->points[i].x;
    if (data_cloud->points[i].y > maxY) maxY = data_cloud->points[i].y;
    if (data_cloud->points[i].z > maxZ) maxZ = data_cloud->points[i].z;
  }

	*pModel = (POINT3D *)malloc(sizeof(POINT3D) * Nm);
	for(int i = 0; i < Nm; i++)
	{
    if (minX < 0 && maxX > 0)
    {
      if (model_cloud->points[i].x < 0)
        (*pModel)[i].x = model_cloud->points[i].x / -minX;
      else
        (*pModel)[i].x = model_cloud->points[i].x / maxX;
    }
    else if (minX >=0 && maxX >= 0)
    {
      (*pModel)[i].x = model_cloud->points[i].x / maxX;
    }
    else
    {
      (*pModel)[i].x = model_cloud->points[i].x / -minX;
    }

    if (minY < 0 && maxY > 0)
    {
      if (model_cloud->points[i].y < 0)
        (*pModel)[i].y = model_cloud->points[i].y / -minY;
      else
        (*pModel)[i].y = model_cloud->points[i].y / maxY;
    }
    else if (minY >=0 && maxY >= 0)
    {
      (*pModel)[i].y = model_cloud->points[i].y / maxY;
    }
    else
    {
      (*pModel)[i].y = model_cloud->points[i].y / -minY;
    }

    if (minZ < 0 && maxZ > 0)
    {
      if (model_cloud->points[i].z < 0)
        (*pModel)[i].z = model_cloud->points[i].z / -minZ;
      else
        (*pModel)[i].z = model_cloud->points[i].z / maxZ;
    }
    else if (minZ >=0 && maxZ >= 0)
    {
      (*pModel)[i].z = model_cloud->points[i].z / maxZ;
    }
    else
    {
      (*pModel)[i].z = model_cloud->points[i].z / -minZ;
    }

    // std::cout << "Point " << i << ": (" << (*pModel)[i].x << ", " << (*pModel)[i].y << ", " << (*pModel)[i].z << ")\n";
	}

  *pData = (POINT3D *)malloc(sizeof(POINT3D) * Nd);
  for(int i = 0; i < Nd; i++)
  {
    if (minX < 0 && maxX > 0)
    {
      if (data_cloud->points[i].x < 0)
        (*pData)[i].x = data_cloud->points[i].x / -minX;
      else
        (*pData)[i].x = data_cloud->points[i].x / maxX;
    }
    else if (minX >=0 && maxX >= 0)
    {
      (*pData)[i].x = data_cloud->points[i].x / maxX;
    }
    else
    {
      (*pData)[i].x = data_cloud->points[i].x / -minX;
    }

    if (minY < 0 && maxY > 0)
    {
      if (data_cloud->points[i].y < 0)
        (*pData)[i].y = data_cloud->points[i].y / -minY;
      else
        (*pData)[i].y = data_cloud->points[i].y / maxY;
    }
    else if (minY >=0 && maxY >= 0)
    {
      (*pData)[i].y = data_cloud->points[i].y / maxY;
    }
    else
    {
      (*pData)[i].y = data_cloud->points[i].y / -minY;
    }

    if (minZ < 0 && maxZ > 0)
    {
      if (data_cloud->points[i].z < 0)
        (*pData)[i].z = data_cloud->points[i].z / -minZ;
      else
        (*pData)[i].z = data_cloud->points[i].z / maxZ;
    }
    else if (minZ >=0 && maxZ >= 0)
    {
      (*pData)[i].z = data_cloud->points[i].z / maxZ;
    }
    else
    {
      (*pData)[i].z = data_cloud->points[i].z / -minZ;
    }

    // std::cout << "Point " << i << ": (" << (*pData)[i].x << ", " << (*pData)[i].y << ", " << (*pData)[i].z << ")\n";
  }

  ScaleParams scaleParams = {minX, maxX, minY, maxY, minZ, maxZ};
  return scaleParams;
}

void readConfig(string FName, GoICP & goicp)
{
	// Open and parse the associated config file
	ConfigMap config(FName.c_str());

	goicp.MSEThresh = config.getF("MSEThresh");
	goicp.initNodeRot.a = config.getF("rotMinX");
	goicp.initNodeRot.b = config.getF("rotMinY");
	goicp.initNodeRot.c = config.getF("rotMinZ");
	goicp.initNodeRot.w = config.getF("rotWidth");
	goicp.initNodeTrans.x = config.getF("transMinX");
	goicp.initNodeTrans.y = config.getF("transMinY");
	goicp.initNodeTrans.z = config.getF("transMinZ");
	goicp.initNodeTrans.w = config.getF("transWidth");
	goicp.trimFraction = config.getF("trimFraction");
	// If < 0.1% trimming specified, do no trimming
	if(goicp.trimFraction < 0.001)
	{
		goicp.doTrim = false;
	}
	goicp.dt.SIZE = config.getI("distTransSize");
	goicp.dt.expandFactor = config.getF("distTransExpandFactor");

	cout << "CONFIG:" << endl;
	config.print();
	//cout << "(doTrim)->(" << goicp.doTrim << ")" << endl;
	cout << endl;
}

std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> load_cad_model(std::string cad_model_path)
{
  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> results;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000001.pcd", *cloud1) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000001.pcd\n");
    exit(-1);
  }
  results[1] = cloud1;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud5 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000005.pcd", *cloud5) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000005.pcd\n");
    exit(-1);
  }
  results[5] = cloud5;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud6 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000006.pcd", *cloud6) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000006.pcd\n");
    exit(-1);
  }
  results[6] = cloud6;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud8 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000008.pcd", *cloud8) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000008.pcd\n");
    exit(-1);
  }
  results[8] = cloud8;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud9 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000009.pcd", *cloud9) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000009.pcd\n");
    exit(-1);
  }
  results[9] = cloud9;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud10 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000010.pcd", *cloud10) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000010.pcd\n");
    exit(-1);
  }
  results[10] = cloud10;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud11 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000011.pcd", *cloud11) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000011.pcd\n");
    exit(-1);
  }
  results[11] = cloud11;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud12 (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cad_model_path + "/obj_000012.pcd", *cloud12) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file obj_000012.pcd\n");
    exit(-1);
  }
  results[12] = cloud12;

  return results;
}

void load_net(cv::dnn::Net& net, const std::string& modelPath, bool is_cuda)
{
  // if has the following issue:
  // [ERROR:0@0.066] global ./modules/dnn/src/onnx/onnx_importer.cpp (1018) handleNode DNN/ONNX: ERROR during processing node with 3 inputs and 1 outputs: [Conv]:(onnx_node!node_conv2d) from domain='ai.onnx'
  // terminate called after throwing an instance of 'cv::Exception'
  // what():  OpenCV(4.6.0) ./modules/dnn/src/onnx/onnx_importer.cpp:1040: error: (-2:Unspecified error) in function 'handleNode'
  // > Node [Conv@ai.onnx]:(onnx_node!node_conv2d) parse error: OpenCV(4.6.0) ./modules/dnn/src/layers/layers_common.cpp:106: error: (-5:Bad argument) kernel_size (or kernel_h and kernel_w) not specified in function 'getKernelSize'

  // export the model using: yolo export model=best.pt format=onnx opset=12 simplify=True dynamic=False imgsz=640
  auto result = cv::dnn::readNetFromONNX(modelPath);
  if (is_cuda)
  {
    std::cout << "Attempting to use CUDA" << std::endl;
    result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
  }
  else
  {
    std::cout << "Using CPU" << std::endl;
    result.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
  net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

// You can change this parameters to obtain better results
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
    cv::Mat mask;
};

// yolov8 format
cv::Mat format_yolov8(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void object_detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;
    std::cout << "className size: " << className.size() << std::endl;
    // Format the input image to fit the model input requirements
    auto input_image = format_yolov8(image);
    
    // Convert the image into a blob and set it as input to the network
    cv::dnn::blobFromImage(input_image, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    // if have issue such as:
    // terminate called after throwing an instance of 'cv::Exception'
    // what(): OpenCV(4.6.0) ./modules/dnn/include/opencv2/dnn/shape_utils.hpp:170: error: (-215:Assertion failed) start <= (int)shape.size() && end <= (int)shape.size() && start <= end in function 'total'
    // Aborted (core dumped)
    // use opencv version 4.10.0 or 4.8.0
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Scaling factors to map the bounding boxes back to original image size
    int dimensions = outputs[0].size[2];
    int rows = outputs[0].size[1];
    std::cout << "Rows: " << rows << ", Dimensions: " << dimensions << std::endl;

    if (dimensions > rows) {
      rows = outputs[0].size[2];
      dimensions = outputs[0].size[1];

      outputs[0] = outputs[0].reshape(1, dimensions);
      cv::transpose(outputs[0], outputs[0]);
    }

    float *data = (float *)outputs[0].data;

    float x_factor = input_image.cols * 1.0f / INPUT_WIDTH;
    float y_factor = input_image.rows * 1.0f / INPUT_HEIGHT;
    
    std::vector<int> class_ids; // Stores class IDs of detections
    std::vector<float> confidences; // Stores confidence scores of detections
    std::vector<cv::Rect> boxes;   // Stores bounding boxes

    // Loop through all the rows to process predictions
    for (int i = 0; i < rows; ++i) {

        float * classes_scores = data + 4;
        cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > SCORE_THRESHOLD) {

            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            std::cout << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h << std::endl;
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            std::cout << "left: " << left << ", top: " << top << ", width: " << width << ", height: " << height << std::endl;
            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    // Draw the NMS filtered boxes and push results to output
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];

        // Only push the filtered detections
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);

        // Draw the final NMS bounding box and label
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 0), 3);
        std::string label = className[class_ids[idx]];
        cv::putText(image, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
    }
}

// Detection function
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    // Format the input image to fit the model input requirements
    auto input_image = format_yolov8(image);
    
    // Convert the image into a blob and set it as input to the network
    cv::dnn::blobFromImage(input_image, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    // if have issue such as:
    // terminate called after throwing an instance of 'cv::Exception'
    // what(): OpenCV(4.6.0) ./modules/dnn/include/opencv2/dnn/shape_utils.hpp:170: error: (-215:Assertion failed) start <= (int)shape.size() && end <= (int)shape.size() && start <= end in function 'total'
    // Aborted (core dumped)
    // use opencv version 4.10.0 or 4.8.0 (don't use 4.6.0)
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat detection_output = outputs[0];
    cv::Mat proto_output;
    if (outputs.size() > 1) {
        // Heuristic: Proto is usually 4D [1, 32, 160, 160], Detection is 3D [1, 4+nc+32, 8400]
        if (outputs[1].dims == 4) {
            proto_output = outputs[1];
            detection_output = outputs[0];
        } else {
            proto_output = outputs[0];
            detection_output = outputs[1];
        }
    }

    // Scaling factors to map the bounding boxes back to original image size
    int dimensions = detection_output.size[2];
    int rows = detection_output.size[1];

    if (dimensions > rows) {
      rows = detection_output.size[2];
      dimensions = detection_output.size[1];

      detection_output = detection_output.reshape(1, dimensions);
      cv::transpose(detection_output, detection_output);
    }

    float *data = (float *)detection_output.data;

    float x_factor = input_image.cols * 1.0f / INPUT_WIDTH;
    float y_factor = input_image.rows * 1.0f / INPUT_HEIGHT;
    
    std::vector<int> class_ids; // Stores class IDs of detections
    std::vector<float> confidences; // Stores confidence scores of detections
    std::vector<cv::Rect> boxes;   // Stores bounding boxes
    std::vector<std::vector<float>> masks_coeffs;

    // Loop through all the rows to process predictions
    for (int i = 0; i < rows; ++i) {

        float * classes_scores = data + 4;
        cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > SCORE_THRESHOLD) {

            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            // std::cout << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h << std::endl;
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            // std::cout << "left: " << left << ", top: " << top << ", width: " << width << ", height: " << height << std::endl;
            boxes.push_back(cv::Rect(left, top, width, height));

            if (!proto_output.empty()) {
                float* mask_ptr = data + 4 + className.size();
                std::vector<float> mask_coeff(mask_ptr, mask_ptr + 32);
                masks_coeffs.push_back(mask_coeff);
            }
        }

        data += dimensions;
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    cv::Mat proto;
    if (!proto_output.empty()) {
        // Reshape proto to [32, 160*160]
        // proto_output is [1, 32, 160, 160]
        proto = proto_output.reshape(1, 32);
    }

    // Draw the NMS filtered boxes and push results to output
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];

        // Only push the filtered detections
        Detection result;
        result.class_id = class_ids[idx] + 1;
        result.confidence = confidences[idx];
        result.box = boxes[idx];

        if (!proto.empty()) {
            cv::Mat coeff(1, 32, CV_32F, masks_coeffs[idx].data());
            cv::Mat mask_logits = coeff * proto; // [1, 25600]
            mask_logits = mask_logits.reshape(1, 160); // [160, 160]
            
            cv::Mat mask;
            cv::exp(-mask_logits, mask);
            mask = 1.0 / (1.0 + mask); // Sigmoid
            
            // Resize to input size (640x640)
            cv::resize(mask, mask, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            
            // Crop to bounding box (in input space)
            int x = boxes[idx].x / x_factor;
            int y = boxes[idx].y / y_factor;
            int w = boxes[idx].width / x_factor;
            int h = boxes[idx].height / y_factor;
            cv::Rect box_input(x, y, w, h);
            box_input = box_input & cv::Rect(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
            
            if (box_input.width > 0 && box_input.height > 0) {
                cv::Mat mask_cropped = mask(box_input);
                cv::resize(mask_cropped, result.mask, boxes[idx].size());
                result.mask = result.mask > 0.5; // Binarize
            } else {
                result.mask = cv::Mat::zeros(boxes[idx].size(), CV_8U);
            }
        }

        output.push_back(result);

        // Draw the final NMS bounding box and label
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 0), 3);
        std::string label = className[class_ids[idx]];
        cv::putText(image, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);

        if (!result.mask.empty()) {
           cv::Mat colored_mask = cv::Mat::zeros(image.size(), CV_8UC3);
           cv::Mat full_mask = cv::Mat::zeros(image.size(), CV_8U);
           
           // Ensure mask fits in the image (clipping)
           cv::Rect roi = boxes[idx] & cv::Rect(0, 0, image.cols, image.rows);
           if (roi.width > 0 && roi.height > 0) {
               // If box was clipped, we need to clip the mask too
               cv::Rect mask_roi(roi.x - boxes[idx].x, roi.y - boxes[idx].y, roi.width, roi.height);
               result.mask(mask_roi).copyTo(full_mask(roi));
               
               cv::Scalar color = colors[result.class_id % colors.size()];
               colored_mask.setTo(color, full_mask);
               cv::addWeighted(image, 1.0, colored_mask, 0.5, 0.0, image);
           }
        }
    }
}

int parse_scene_gt_json(std::string folder_name, std::map<int, std::map<int, Eigen::Matrix4d>>& all_img_obj_poses)
{
    std::string json_gt_path = folder_name + "/scene_gt.json";
    std::ifstream json_gt_file(json_gt_path);
    if (!json_gt_file.is_open())
    {
        std::cerr << "Failed to open file: " << json_gt_path << '\n';
        return -1;
    }
    
    Json::Value root;
    json_gt_file >> root;
    for (const auto& key : root.getMemberNames())
    {
        Json::Value instance = root[key];
        std::map<int, Eigen::Matrix4d> obj_poses;
        for (const auto& instance : root[key])
        {
            std::map<int, Eigen::Matrix4d> obj_pos;
            int obj_id = instance["obj_id"].asInt();
            auto translation = instance["cam_t_m2c"];
            auto rotation = instance["cam_R_m2c"];
            // std::cout << "Object ID: " << obj_id << " " << translation << '\n';
            Eigen::Vector3d position(translation[0].asDouble(), translation[1].asDouble(), translation[2].asDouble());
            Eigen::Matrix3d rotation_matrix;
            rotation_matrix << rotation[0].asDouble(), rotation[1].asDouble(), rotation[2].asDouble(),
                                rotation[3].asDouble(), rotation[4].asDouble(), rotation[5].asDouble(),
                                rotation[6].asDouble(), rotation[7].asDouble(), rotation[8].asDouble();

            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            transform.block<3,3>(0,0) = rotation_matrix;
            transform.block<3,1>(0,3) = position;
            obj_poses[obj_id] = transform;
        }
        all_img_obj_poses[std::stoi(key)] = obj_poses;
        // std::cout << "Processed frame " << key << " with " << obj_poses.size() << " objects." << '\n';
    }

    return 0;
}

void parse_model_info_json(std::string model_info_path, std::map<int, Eigen::Matrix4d>& model_diameter)
{
    std::ifstream json_model_file(model_info_path);
    if (!json_model_file.is_open())
    {
        std::cerr << "Failed to open file: " << model_info_path << '\n';
        return;
    }
    
    Json::Value root;
    json_model_file >> root;
    for (const auto& key : root.getMemberNames())
    {
        Json::Value instance = root[key];
        int obj_id = std::stoi(key);
        auto symmetries_discrete = instance["symmetries_discrete"];
        Eigen::Matrix4d sym_matrix = Eigen::Matrix4d::Identity();
        if (symmetries_discrete.size() == 0)
        {
            // std::cerr << "No symmetries_discrete for object ID " << obj_id << '\n';
            continue;
        }
        sym_matrix << symmetries_discrete[0][0].asDouble(), symmetries_discrete[0][1].asDouble(), symmetries_discrete[0][2].asDouble(), symmetries_discrete[0][3].asDouble(),
                      symmetries_discrete[0][4].asDouble(), symmetries_discrete[0][5].asDouble(), symmetries_discrete[0][6].asDouble(), symmetries_discrete[0][7].asDouble(),
                      symmetries_discrete[0][8].asDouble(), symmetries_discrete[0][9].asDouble(), symmetries_discrete[0][10].asDouble(), symmetries_discrete[0][11].asDouble(),
                      symmetries_discrete[0][12].asDouble(), symmetries_discrete[0][13].asDouble(), symmetries_discrete[0][14].asDouble(), symmetries_discrete[0][15].asDouble();
        model_diameter[obj_id] = sym_matrix;
        // std::cout << "Object ID: " << obj_id << " Symmetry Matrix:\n" << sym_matrix << '\n';
    }
}

std::string get_string_num(int i)
{
    std::string fold_num = "/";
    if (i < 10)
    {
        fold_num += "00000";
        fold_num += std::to_string(i);
    }
    else if (i < 100)
    {
        fold_num += "0000";
        fold_num += std::to_string(i);
    }
    else
    {
        fold_num += "000";
        fold_num += std::to_string(i);
    }
    return fold_num;
}

double mssd(Eigen::Matrix4d& gt, Eigen::Matrix4d& pred, pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud, std::map<int, Eigen::Matrix4d>& models_st)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*model_cloud, *transformed_model, pred);

    std::vector<double> es;

    Eigen::Matrix3d R = gt.block<3,3>(0,0);
    Eigen::Vector3d t = gt.block<3,1>(0,3);

    for (const auto model_st : models_st)
    {
        Eigen::Matrix3d R_sym = model_st.second.block<3,3>(0,0);
        Eigen::Vector3d t_sym = model_st.second.block<3,1>(0,3);

        Eigen::Matrix3d R_gt_sym = R * R_sym;
        Eigen::Vector3d t_gt_sym = R * t_sym + t;

        Eigen::Matrix4d gt_transform = Eigen::Matrix4d::Identity();
        gt_transform.block<3,3>(0,0) = R_gt_sym;
        gt_transform.block<3,1>(0,3) = t_gt_sym;

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_sym (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*model_cloud, *transformed_model_sym, gt_transform);

        double max_distance = 0.0;
        for (size_t i = 0; i < transformed_model->points.size(); ++i)
        {
            double distance = std::sqrt(std::pow(transformed_model->points[i].x - transformed_model_sym->points[i].x, 2) +
                                        std::pow(transformed_model->points[i].y - transformed_model_sym->points[i].y, 2) +
                                        std::pow(transformed_model->points[i].z - transformed_model_sym->points[i].z, 2));
            if (distance > max_distance)
            {
                max_distance = distance;
            }
        }
        es.push_back(max_distance);
    }
    double min_es = *std::min_element(es.begin(), es.end());
    return min_es;
}

int main()
{
  // Set verbosity to a lower level (e.g., only show errors)
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR); 

  std::string model_info_path = "../../../lmo_models/models/models_info.json";
  std::map<int, Eigen::Matrix4d> model_st;
  parse_model_info_json(model_info_path, model_st);

  // load the model point cloud (pcl_ply2pcd)
  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> model_clouds = load_cad_model("../../../lmo_models/models/pcd");

  // init YOLOV8
  init_className();
  cv::dnn::Net net;
  load_net(net, "../../../models/yolo_icp_correct_label2/weights/best.onnx",  false);

  std::string train_pbr_path = "../../../train_pbr";
  for (int i = 0; i < 50; i++)
  {
    std::string dataset = get_string_num(i);

    // parse the ground truth json file
    std::string scene_folder = train_pbr_path + dataset;
    std::map<int, std::map<int, Eigen::Matrix4d>> all_img_obj_poses;
    parse_scene_gt_json(scene_folder, all_img_obj_poses);

    // loop through all the image in the test folder
    std::filesystem::path test_imgs_path = train_pbr_path + dataset + "/images/test";
    if (!std::filesystem::exists(test_imgs_path) || !std::filesystem::is_directory(test_imgs_path)) {
      std::cerr << "Error: Directory not found or not a valid directory." << std::endl;
      continue;
    }

    std::map<int, double> mssd_errors;
    std::map<int, int> mssd_error_counts; 

    std::map<int, double> trans_errors;
    std::map<int, int> trans_error_counts; 

    std::map<int, double> rot_errors;
    std::map<int, int> rot_error_counts;

    int time_spents = 0;
    int time_counts = 0;

    int count = 0;
    for (const auto& img : std::filesystem::directory_iterator(test_imgs_path)) 
    {
      if (!std::filesystem::is_regular_file(img)) 
      {
        continue; // Skip non-regular files
      }

      // if (count > 20) break;
      count++;

      std::string img_filename = img.path().filename().string();
      std::cout << "Processing image: " << img_filename << std::endl;

      // read the image
      std::string image_path = test_imgs_path.string() + "/" + img_filename;
      cv::Mat frame = cv::imread(image_path);

      // ********************************** part 1 detection and masking using yolov8 *******************************
      std::vector<Detection> output;
      detect(frame, net, output, classNames);

      // save the frame
      // cv::imwrite("detection_output.jpg", frame);

      // generate mask image
      std::map<int, cv::Mat> masks;
      for (const auto& detection : output)
      {
        cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        if (!detection.mask.empty()) 
        {
          cv::Rect roi = detection.box & cv::Rect(0, 0, frame.cols, frame.rows);
          if (roi.width > 0 && roi.height > 0) 
          {
            cv::Rect mask_roi(roi.x - detection.box.x, roi.y - detection.box.y, roi.width, roi.height);
            detection.mask(mask_roi).copyTo(mask(roi));
          }
        } 
        else 
        {
          cv::rectangle(mask, detection.box, cv::Scalar(255), cv::FILLED);
        }
        masks[detection.class_id] = mask;
      }
      // ***************************************** end of part 1 ***************************************** 




      // depth image
      std::string depth_image_path = train_pbr_path + dataset + "/depth/" + img_filename.substr(0, img_filename.find_last_of('.')) + ".png";
      cv::Mat depth_image = cv::imread(depth_image_path, cv::IMREAD_UNCHANGED);

      // generate masked depth point clouds and pre-process the point clouds
      std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> depth_point_clouds;
      for (const auto& mask : masks)
      {
        cv::Mat masked;
        cv::bitwise_and(depth_image, depth_image, masked, mask.second);

        pcl::PointCloud<pcl::PointXYZ>::Ptr depth_pc = depth_to_point_cloud(masked);

        pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud (depth_pc);
        vg.setLeafSize (2, 2, 2);
        vg.filter (*depth_cloud_filtered);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (depth_cloud_filtered);
        sor.setMeanK (50);
        sor.setStddevMulThresh (0.5);
        sor.filter (*depth_cloud_filtered);
        depth_point_clouds[mask.first] = depth_cloud_filtered;

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(depth_cloud_filtered, 0, 255, 0);
        std::cout << "Masked depth point cloud for class " << mask.first << " has " << depth_cloud_filtered->points.size() << " points. (origin: " << depth_pc->points.size() << ")\n";
      }


      // ****************************************** part 2 registration using ICP ******************************************
      std::map<int, Eigen::Matrix4f> final_transformation;
      int start_time = sec();
#if GOICP
      for (const auto& depth_pc : depth_point_clouds)
      {
        if (depth_pc.first != 1 && depth_pc.first != 5 && depth_pc.first != 6 && depth_pc.first != 8 && depth_pc.first != 9 && depth_pc.first != 10 && depth_pc.first != 11 && depth_pc.first != 12)
          continue;

        std::cout << "Registering for class " << depth_pc.first << '\n';
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4f centroid = computeCentroid(depth_pc.second);
        Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
        translation(0, 3) = centroid[0];
        translation(1, 3) = centroid[1];
        translation(2, 3) = centroid[2];
        pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, translation);

        int Nm = 0, Nd = 0, NdDownsampled;
        clock_t  clockBegin, clockEnd;
        string modelFName, dataFName, configFName, outputFname;
        POINT3D * pModel, * pData;
        GoICP goicp;

        readConfig("../config_example.txt", goicp);

        // Load model and data point clouds
        ScaleParams scaleParams = loadPointCloud(transformed_model_cloud, depth_pc.second, &pModel, &pData, Nm, Nd);
      
        goicp.pModel = pModel;
        goicp.Nm = Nm;
        goicp.pData = pData;
        goicp.Nd = Nd;

        // Build Distance Transform
        std::cout << "Building Distance Transform..." << std::flush;
        clockBegin = clock();
        goicp.BuildDT();
        clockEnd = clock();
        std::cout << (double)(clockEnd - clockBegin)/CLOCKS_PER_SEC << "s (CPU)" << std::endl;
        // Run GO-ICP
        // if(NdDownsampled > 0)
        // {
        //   goicp.Nd = NdDownsampled; // Only use first NdDownsampled data points (assumes data points are randomly ordered)
        // }
        // cout << "Model ID: " << modelFName << " (" << goicp.Nm << "), Data ID: " << dataFName << " (" << goicp.Nd << ")" << std::endl;
        std::cout << "Registering..." << std::endl;
        clockBegin = clock();
        goicp.Register();
        clockEnd = clock();
        double time = (double)(clockEnd - clockBegin)/CLOCKS_PER_SEC;
        std::cout << "Optimal Rotation Matrix:" << std::endl;
        std::cout << goicp.optR << std::endl;
        std::cout << "Optimal Translation Vector:" << std::endl;
        std::cout << goicp.optT << std::endl;
        std::cout << "Finished in " << time << std::endl;

        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        transformation(0, 0) = goicp.optR.val[0][0];
        transformation(0, 1) = goicp.optR.val[0][1];
        transformation(0, 2) = goicp.optR.val[0][2];
        transformation(1, 0) = goicp.optR.val[1][0];
        transformation(1, 1) = goicp.optR.val[1][1];
        transformation(1, 2) = goicp.optR.val[1][2];
        transformation(2, 0) = goicp.optR.val[2][0];
        transformation(2, 1) = goicp.optR.val[2][1];
        transformation(2, 2) = goicp.optR.val[2][2];
        transformation(0, 3) = goicp.optT.val[0][0];
        transformation(1, 3) = goicp.optT.val[1][0];
        transformation(2, 3) = goicp.optT.val[2][0];
        std::cout << "Before scaling translation: \n" << transformation << '\n';

        if (scaleParams.minX < 0 && scaleParams.maxX > 0)
        {
          transformation(0, 3) = goicp.optT.val[0][0] * ((goicp.optT.val[0][0] < 0) ? -scaleParams.minX : scaleParams.maxX);
        }
        else if (scaleParams.minX >=0 && scaleParams.maxX >= 0)
        {
          transformation(0, 3) = goicp.optT.val[0][0] * scaleParams.maxX;
        }
        else
        {
          transformation(0, 3) = goicp.optT.val[0][0] * -scaleParams.minX;
        }

        if (scaleParams.minY < 0 && scaleParams.maxY > 0)
        {
          transformation(1, 3) = goicp.optT.val[1][0] * ((goicp.optT.val[1][0] < 0) ? -scaleParams.minY : scaleParams.maxY);
        }
        else if (scaleParams.minY >=0 && scaleParams.maxY >= 0)
        {
          transformation(1, 3) = goicp.optT.val[1][0] * scaleParams.maxY;
        }
        else
        {
          transformation(1, 3) = goicp.optT.val[1][0] * -scaleParams.minY;
        }

        if (scaleParams.minZ < 0 && scaleParams.maxZ > 0)
        {
          transformation(2, 3) = goicp.optT.val[2][0] * ((goicp.optT.val[2][0] < 0) ? -scaleParams.minZ : scaleParams.maxZ);
        }
        else if (scaleParams.minZ >=0 && scaleParams.maxZ >= 0)
        {
          transformation(2, 3) = goicp.optT.val[2][0] * scaleParams.maxZ;
        }
        else
        {
          transformation(2, 3) = goicp.optT.val[2][0] * -scaleParams.minZ;
        }
        std::cout << "After scaling translation: \n" << transformation << '\n';

        final_transformation[depth_pc.first] = transformation * translation;

        delete(pModel);
        delete(pData);
      }

#else

      for (const auto& depth_pc : depth_point_clouds)
      {
        if (depth_pc.first != 1 && depth_pc.first != 5 && depth_pc.first != 6 && depth_pc.first != 8 && depth_pc.first != 9 && depth_pc.first != 10 && depth_pc.first != 11 && depth_pc.first != 12)
          continue;

        std::cout << "Registering for class " << depth_pc.first << '\n';
#if initial_angle_guess

        double best_fitness = std::numeric_limits<double>::max();
        Eigen::Vector4f centroid = computeCentroid(depth_pc.second);

        double cy = 0, sy = 0; // yoll (z)
        double cp = 0, sp = 0; // pitch (y)
        double cr = 0, sr = 0; // roll (x)

        int step = 90;

#if USING_METHOD_1

        for (int r = 0; r < 360; r += step)
        {
          double angle_x = r * M_PI / 180.0;
          sr = sin(angle_x);
          cr = cos(angle_x);

          for (int p = 0; p < 360; p += step)
          {
            double angle_y = p * M_PI / 180.0;
            sp = sin(angle_y);
            cp = cos(angle_y);

            for (int y = 0; y < 360; y += step)
            {
              // std::cout << "Testing initial guess rotation (r,p,y): (" << r << ", " << p << ", " << y << "), fitness: ";

              double angle_z = y * M_PI / 180.0;
              sy = sin(angle_z);
              cy = cos(angle_z);

              Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
              initial_guess(0,0) = cp * cy;
              initial_guess(0,1) = cy * sp * sr - sy * cr;
              initial_guess(0,2) = cy * sp * cr + sy * sr;
              initial_guess(1,0) = cp * sy;
              initial_guess(1,1) = sy * sp * sr + cy * cr;
              initial_guess(1,2) = sy * sp * cr - cy * sr;
              initial_guess(2,0) = -sp;
              initial_guess(2,1) = cp * sr;
              initial_guess(2,2) = cp * cr;

              pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
              initial_guess(0, 3) = centroid[0];
              initial_guess(1, 3) = centroid[1];
              initial_guess(2, 3) = centroid[2];
              pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, initial_guess);
              
              pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
              icp.setMaximumIterations (200);
              icp.setInputSource(depth_pc.second);
              icp.setInputTarget(transformed_model_cloud);
              icp.setMaxCorrespondenceDistance(100); // Example value
              pcl::PointCloud<pcl::PointXYZ> unused_result;
              icp.align(unused_result);

              if (icp.getFitnessScore() < best_fitness)
              {
                best_fitness = icp.getFitnessScore();
                final_transformation[depth_pc.first] = icp.getFinalTransformation() * initial_guess;
              }
              // std::cout << icp.getFitnessScore() << '\n';
            }

          }
        }

#else
        // x rotation (0, 90, 180, 270)
        cy = 0, sy = 0; 
        cp = 0, sp = 0; 
        cr = 0, sr = 0;
        for (int r = 0; r < 360; r += step)
        {
          std::cout << "Testing initial guess rotation (r,p,y): (" << r << ", 0, 0), fitness: ";

          double angle_x = r * M_PI / 180.0;
          sr = sin(angle_x);
          cr = cos(angle_x);
          sp = sin(0);
          cp = cos(0);
          sy = sin(0);
          cy = cos(0);

          Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
          initial_guess(0,0) = cp * cy;
          initial_guess(0,1) = cy * sp * sr - sy * cr;
          initial_guess(0,2) = cy * sp * cr + sy * sr;
          initial_guess(1,0) = cp * sy;
          initial_guess(1,1) = sy * sp * sr + cy * cr;
          initial_guess(1,2) = sy * sp * cr - cy * sr;
          initial_guess(2,0) = -sp;
          initial_guess(2,1) = cp * sr;
          initial_guess(2,2) = cp * cr;

          pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
          initial_guess(0, 3) = centroid[0];
          initial_guess(1, 3) = centroid[1];
          initial_guess(2, 3) = centroid[2];
          pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, initial_guess);
          
          pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;  // GICP 泛化的ICP，或者叫Plane to Plane
          icp.setTransformationEpsilon(10.0);
          // icp.setMaxCorrespondenceDistance(150.0);
          icp.setMaximumIterations(200);
          icp.setRANSACIterations(20);
          icp.setInputTarget(depth_pc.second);
          icp.setInputSource(transformed_model_cloud);
          pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZ>);
          icp.align(*unused_result);

          if (icp.getFitnessScore() < best_fitness)
          {
            best_fitness = icp.getFitnessScore();
            final_transformation[depth_pc.first] = icp.getFinalTransformation() * initial_guess;
          }
          std::cout << icp.getFitnessScore() << '\n';
        }

        // y rotation (0, 90, 180, 270)
        cy = 0, sy = 0; 
        cp = 0, sp = 0; 
        cr = 0, sr = 0;
        for (int p = step; p < 360; p += step)
        {
          std::cout << "Testing initial guess rotation (r,p,y): (0, " << p << ", 0), fitness: ";

          double angle_y = p * M_PI / 180.0;
          sr = sin(0);
          cr = cos(0);
          sp = sin(angle_y);
          cp = cos(angle_y);
          sy = sin(0);
          cy = cos(0);

          Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
          initial_guess(0,0) = cp * cy;
          initial_guess(0,1) = cy * sp * sr - sy * cr;
          initial_guess(0,2) = cy * sp * cr + sy * sr;
          initial_guess(1,0) = cp * sy;
          initial_guess(1,1) = sy * sp * sr + cy * cr;
          initial_guess(1,2) = sy * sp * cr - cy * sr;
          initial_guess(2,0) = -sp;
          initial_guess(2,1) = cp * sr;
          initial_guess(2,2) = cp * cr;

          pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
          initial_guess(0, 3) = centroid[0];
          initial_guess(1, 3) = centroid[1];
          initial_guess(2, 3) = centroid[2];
          pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, initial_guess);
          
          pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;  // GICP 泛化的ICP，或者叫Plane to Plane
          icp.setTransformationEpsilon(10.0);
          // icp.setMaxCorrespondenceDistance(150.0);
          icp.setMaximumIterations(200);
          icp.setRANSACIterations(20);
          icp.setInputTarget(depth_pc.second);
          icp.setInputSource(transformed_model_cloud);
          pcl::PointCloud<pcl::PointXYZ> unused_result;
          icp.align(unused_result);

          if (icp.getFitnessScore() < best_fitness)
          {
            best_fitness = icp.getFitnessScore();
            final_transformation[depth_pc.first] = icp.getFinalTransformation() * initial_guess;
          }
          std::cout << icp.getFitnessScore() << '\n';
        }

        // z rotation (0, 90, 180, 270)
        cy = 0, sy = 0; 
        cp = 0, sp = 0; 
        cr = 0, sr = 0;
        for (int y = step; y < 360; y += step)
        {
          std::cout << "Testing initial guess rotation (r,p,y): (0, 0, " << y << "), fitness: ";

          double angle_z = y * M_PI / 180.0;
          sr = sin(0);
          cr = cos(0);
          sp = sin(0);
          cp = cos(0);
          sy = sin(angle_z);
          cy = cos(angle_z);

          Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
          initial_guess(0,0) = cp * cy;
          initial_guess(0,1) = cy * sp * sr - sy * cr;
          initial_guess(0,2) = cy * sp * cr + sy * sr;
          initial_guess(1,0) = cp * sy;
          initial_guess(1,1) = sy * sp * sr + cy * cr;
          initial_guess(1,2) = sy * sp * cr - cy * sr;
          initial_guess(2,0) = -sp;
          initial_guess(2,1) = cp * sr;
          initial_guess(2,2) = cp * cr;

          pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
          initial_guess(0, 3) = centroid[0];
          initial_guess(1, 3) = centroid[1];
          initial_guess(2, 3) = centroid[2];
          pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, initial_guess);
          
          pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;  // GICP 泛化的ICP，或者叫Plane to Plane
          icp.setTransformationEpsilon(10.0);
          // icp.setMaxCorrespondenceDistance(150.0);
          icp.setMaximumIterations(200);
          icp.setRANSACIterations(20);
          icp.setInputTarget(depth_pc.second);
          icp.setInputSource(transformed_model_cloud);
          pcl::PointCloud<pcl::PointXYZ> unused_result;
          icp.align(unused_result);

          if (icp.getFitnessScore() < best_fitness)
          {
            best_fitness = icp.getFitnessScore();
            final_transformation[depth_pc.first] = icp.getFinalTransformation() * initial_guess;
          }
          std::cout << icp.getFitnessScore() << '\n';
        }
#endif
        
        
#else

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4f centroid = computeCentroid(depth_pc.second);
        Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
        translation(0, 3) = centroid[0];
        translation(1, 3) = centroid[1];
        translation(2, 3) = centroid[2];
        pcl::transformPointCloud(*model_clouds[depth_pc.first], *transformed_model_cloud, translation);

        // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;  // GICP 泛化的ICP，或者叫Plane to Plane
        // icp.setTransformationEpsilon(10.0);
        // // icp.setMaxCorrespondenceDistance(150.0);
        // icp.setMaximumIterations(200);
        // icp.setRANSACIterations(20);
        // icp.setInputTarget(depth_pc.second);
        // icp.setInputSource(transformed_model_cloud);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZ>);
        // icp.align(*unused_result);

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setTransformationEpsilon(50.0);
        icp.setMaximumIterations (200);
        icp.setInputSource(depth_pc.second);
        icp.setInputTarget(transformed_model_cloud);
        icp.setMaxCorrespondenceDistance(100); // Example value
        pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZ>);
        icp.align(*unused_result);


        std::cout << "Fitness score: " << icp.getFitnessScore() << '\n';
        final_transformation[depth_pc.first] = icp.getFinalTransformation() * translation;
        // std::cout << "Final transformation[" << depth_pc.first << "]: \n" << final_transformation[depth_pc.first] << "\n\n";

#endif

      }
      
#endif
      int end_time = sec();
      int time_spent = end_time - start_time;
      std::cout << "Time taken for part 2: " << time_spent << " seconds\n";
      time_counts++;
      time_spents += time_spent;
      // ****************************************** end of part 2 ******************************************




      // ****************************************** part 3 evaluate the accuracy ******************************************
      int fileNum = std::stoi(img_filename.substr(0, img_filename.find_last_of('.')));
      std::map<int, Eigen::Matrix4d> gt_poses = all_img_obj_poses[fileNum];

      for (const auto& transform : final_transformation)
      {
        int class_id = transform.first;
        Eigen::Matrix4f pred = transform.second;
        Eigen::Matrix4d pred_d;
        pred_d = pred.cast<double>();

        Eigen::Matrix3d R_est = pred_d.block<3,3>(0,0);
        Eigen::Vector3d t_est = pred_d.block<3,1>(0,3);

        if (gt_poses.find(class_id) != gt_poses.end())
        {
          Eigen::Matrix4d gt = gt_poses[class_id];
          double error = mssd(gt, pred_d, model_clouds[class_id], model_st);
          std::cout << "MSSD error for class " << class_id << ": " << error << " mm\t";

          Eigen::Matrix3d R_gt = gt.block<3,3>(0,0);
          Eigen::Vector3d t_gt = gt.block<3,1>(0,3);

          double trans_error = (t_gt - t_est).norm();
          Eigen::Matrix3d R_diff = R_gt.transpose() * R_est;
          double rot_error = std::acos(std::min(1.0, std::max(-1.0, (R_diff.trace() - 1) / 2))) * (180.0 / M_PI);

          std::cout << "Transition error: " << trans_error << " mm ";
          std::cout << "Rotation error (angle): " << rot_error << " degrees" << std::endl;

          if (mssd_errors.find(class_id) == mssd_errors.end()) mssd_errors[class_id] = error;
          else mssd_errors[class_id] += error;
          if (mssd_error_counts.find(class_id) == mssd_error_counts.end()) mssd_error_counts[class_id] = 1;
          else mssd_error_counts[class_id]++;

          if (trans_errors.find(class_id) == trans_errors.end()) trans_errors[class_id] = trans_error;
          else trans_errors[class_id] += trans_error;
          if (trans_error_counts.find(class_id) == trans_error_counts.end()) trans_error_counts[class_id] = 1;
          else trans_error_counts[class_id]++;

          if (rot_errors.find(class_id) == rot_errors.end()) rot_errors[class_id] = rot_error;
          else rot_errors[class_id] += rot_error;
          if (rot_error_counts.find(class_id) == rot_error_counts.end()) rot_error_counts[class_id] = 1;
          else rot_error_counts[class_id]++;
        }
        else
        {
          std::cout << "No ground truth pose for class " << class_id << std::endl;
        }
      }
      // ****************************************** end of part 3  ******************************************

#if VISUALIZE
      pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
      viewer->setBackgroundColor (0, 0, 0);
      for (const auto& transform : final_transformation)
      {
        int class_id = transform.first;
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_transformed_model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*model_clouds[class_id], *final_transformed_model_cloud, transform.second);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_transformed_model_cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*final_transformed_model_cloud, *final_transformed_model_cloud_rgb);
        for (auto& point : final_transformed_model_cloud_rgb->points)
        {
          point.r = 255;
          point.g = 0;
          point.b = 0;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr depth_cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*depth_point_clouds[class_id], *depth_cloud_rgb);
        for (auto& point : depth_cloud_rgb->points)
        {
          point.r = 0;
          point.g = 255;
          point.b = 0;
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rg1(final_transformed_model_cloud_rgb);
        viewer->addPointCloud<pcl::PointXYZRGB> (final_transformed_model_cloud_rgb, rg1, "model_"+std::to_string(class_id)+"_cloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model_"+std::to_string(class_id)+"_cloud");

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rg2(depth_cloud_rgb);
        viewer->addPointCloud<pcl::PointXYZRGB> (depth_cloud_rgb, rg2, "depth_"+std::to_string(class_id)+"_cloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "depth_"+std::to_string(class_id)+"_cloud");
      }
      viewer->addCoordinateSystem (1.0);
      viewer->initCameraParameters ();

      while (!viewer->wasStopped ())
      {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
      }
#endif

      std::cout << "-------------------------------------------------------------------\n\n";
    }

    for (const auto& err : mssd_errors)
    {
      int class_id = err.first;
      double avg_error = err.second / mssd_error_counts[class_id];
      std::cout << "Average MSSD error for class " << class_id << ": " << avg_error << " mm\n";
    }
    for (const auto& err : trans_errors)
    {
      int class_id = err.first;
      double avg_error = err.second / trans_error_counts[class_id];
      std::cout << "Average Transition error for class " << class_id << ": " << avg_error << " mm\n";
    }
    for (const auto& err : rot_errors)
    {
      int class_id = err.first;
      double avg_error = err.second / rot_error_counts[class_id];
      std::cout << "Average Rotation error for class " << class_id << ": " << avg_error << " degrees\n";
    }
    std::cout << "Average time spent for ICP registration: " << (time_spents / time_counts) << " seconds, " << (time_spents / time_counts)/ 60.0 << " minutes\n";
    std::cout << "file count: " << count << '\n';
  }

  return 0;
}