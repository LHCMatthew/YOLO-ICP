#include <iostream>
#include <string>
#include <json/json.h>
#include <fstream>
#include <vector>
#include <filesystem>
#include <thread>
#include <chrono>

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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace std::chrono_literals;

void visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string cloud_name)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, cloud_name);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cloud_name);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  viewer->addSphere(pcl::PointXYZ(0.0, 0.0, 0.0), 3, 1.0, 0.0, 0.0, "origin_sphere");

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
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
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    return cloud;
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

int main()
{
    for (int i = 0; i < 1; i++)
    {
        std::string folder_name = "../../../train_pbr/0000";
        if (i < 10)
        {
            folder_name += "0";
            folder_name += std::to_string(i);
        }
        else
        {
            folder_name += std::to_string(i);
        }
        std::cout << folder_name << '\n';

        // parse the ground truth pos
        std::map<int, std::map<int, Eigen::Matrix4d>> all_img_obj_poses;
        if (parse_scene_gt_json(folder_name, all_img_obj_poses) != 0)
        {
            continue;
        }

        // create a labels folder
        std::string label_folder = folder_name + "/labels/";
        if (!std::filesystem::exists(label_folder) && !std::filesystem::create_directory(label_folder))
        {
            std::cerr << "Failed to create directory: " << label_folder << '\n';
            continue;
        }

        // start labeling the images
        for (int j = 0; j < 1000; j++)
        {
            std::string file_num = "000";
            if (j < 10)
            {
                file_num += "00";
                file_num += std::to_string(j);
            }
            else if (j < 100)
            {
                file_num += "0";
                file_num += std::to_string(j);
            }
            else
            {
                file_num += std::to_string(j);
            }
            std::string depth_name = folder_name + "/depth/" + file_num + ".png";

            cv::Mat depth_img = cv::imread(depth_name, cv::IMREAD_UNCHANGED);
            if (depth_img.empty())
            {
                std::cerr << "Failed to load depth image: " << depth_name << '\n';
                continue;
            }

            std::string mask_path = folder_name + "/mask_visib/";
            if (!std::filesystem::exists(mask_path) || !std::filesystem::is_directory(mask_path))
            {
                std::cerr << "Mask directory does not exist: " << mask_path << '\n';
                continue;
            }

            std::ofstream label_file;
            label_file.open(label_folder + file_num + ".txt", std::ios::app);
            if (!label_file.is_open())
            {
                std::cerr << "Failed to open label file for writing: " << label_folder + file_num + ".txt" << '\n';
                continue;
            }

            // for each mask label the image
            for (const auto& mask : std::filesystem::directory_iterator(mask_path)) {
                std::string mask_file = mask.path().filename().string();
                std::string mask_depth_id = mask_file.substr(0, mask_file.find('_'));

                if (std::stoi(mask_depth_id) != j)
                    continue;

                std::cout << "Processing mask file: " << mask_file << " folder: " << i << '\n';

                cv::Mat mask_img = cv::imread(mask.path().string(), cv::IMREAD_UNCHANGED);
                if (mask_img.empty())
                {
                    std::cerr << "Failed to load mask image: " << mask.path().string() << '\n';
                    continue;
                }

                cv::Mat masked_depth;
                cv::bitwise_and(depth_img, depth_img, masked_depth, mask_img);

                pcl::PointCloud<pcl::PointXYZ>::Ptr masked_depth_cloud = depth_to_point_cloud(masked_depth);
                Eigen::Vector4f centroid = computeCentroid(masked_depth_cloud);

                if (masked_depth_cloud->points.size() == 0)
                {
                    std::cerr << "No valid points in masked depth image: " << mask_file << '\n';
                    continue;
                }

                // find the obj ID of that img
                double best_dist = std::numeric_limits<double>::max();
                int best_obj_id = -1;
                const auto& obj_pos = all_img_obj_poses[j];
                for (const auto pos : obj_pos)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_depth_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                    Eigen::Matrix4f transform = pos.second.cast<float>().inverse();
                    pcl::transformPointCloud(*masked_depth_cloud, *transformed_depth_cloud, transform);

                    Eigen::Vector4f transformed_centroid = computeCentroid(transformed_depth_cloud);
                    double dist = std::sqrt(std::pow(transformed_centroid[0], 2) +
                                            std::pow(transformed_centroid[1], 2) +
                                            std::pow(transformed_centroid[2], 2));
                        
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_obj_id = pos.first;
                    }
                }

                if (best_obj_id == -1)
                {
                    std::cerr << "No matching object found for mask: " << mask_file << '\n';
                    continue;
                }
                // std::cout << "Best matching object ID: " << best_obj_id << " with distance: " << best_dist << " mm" << '\n';
                    
                // Find contours
                std::vector<std::vector<cv::Point>> contours;
                cv::threshold(mask_img, mask_img, 127, 255, cv::THRESH_BINARY);
                cv::findContours(mask_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                
                // find the largest contour
                int max_contour_size = 0;
                int max_contour_index = -1;
                for (const auto& cnt : contours) 
                {
                    if (cnt.size() > max_contour_size) {
                        max_contour_size = cnt.size();
                        max_contour_index = &cnt - &contours[0];
                    }
                }

                if (max_contour_index == -1)
                {
                    std::cerr << "No contours found in mask: " << mask_file << '\n';
                    continue;
                }

                // Flatten and normalize coordinates
                std::vector<double> seg;
                for (const auto& pt : contours[max_contour_index]) {
                    double norm_x = static_cast<double>(pt.x) / mask_img.cols;
                    double norm_y = static_cast<double>(pt.y) / mask_img.rows;
                    seg.push_back(norm_x);
                    seg.push_back(norm_y);
                }
                
                // Format as YOLO line
                std::ostringstream line_stream;
                line_stream << (best_obj_id-1); // YOLO class index starts from 0
                
                for (double coord : seg) {
                    line_stream << " " << std::fixed << std::setprecision(6) << coord;
                }
                
                label_file << line_stream.str() << "\n";          
            }

            label_file.close();
        }
    }
}