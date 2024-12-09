#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/PositionTarget.h>
#include <math.h>
#include "m3_explorer/hastar.h"
#include "m3_explorer/astar.h"
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

class CircleTrajectory {
public:
    CircleTrajectory() {
        state_sub = nh.subscribe<mavros_msgs::State>("/uav0/mavros/state", 10, &CircleTrajectory::state_cb, this);
        local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>("/uav0/mavros/setpoint_raw/local", 10);
        arming_client = nh.serviceClient<mavros_msgs::CommandBool>("/uav0/mavros/cmd/arming");
        set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/uav0/mavros/set_mode");
    }

    void state_cb(const mavros_msgs::State::ConstPtr& msg) {
        current_state = *msg;
    }

    void run() {
        ros::Publisher sp_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav0/mavros/setpoint", 10);

        // Hastar planning;
        // cout << "start planning" << endl;
        // planning.search_path(nullptr, Eigen::Vector3f(0.0, 0.0, 2.0), Eigen::Vector3f(2.0, 10.0, 2.0), 0.0);
        // cout << "end" << endl;

        ros::Rate rate(20.0);
        while (ros::ok() && !current_state.connected) {
            ros::spinOnce();
            rate.sleep();
        }

        geometry_msgs::PoseStamped sp;
        sp.header.frame_id = "map";
        sp.pose.position.z = 1.5;
        sp.pose.orientation.w = 1.0;

        mavros_msgs::PositionTarget target_pose;
        target_pose.header.frame_id = "map";
        target_pose.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        target_pose.type_mask = 0;
        target_pose.position.z = 1.5;

        ros::Time last_request = ros::Time::now();
        ros::Time start = ros::Time::now();

        //send a few setpoints before starting
        for(int i = 5; ros::ok() && i > 0; --i){
            local_pos_pub.publish(target_pose);
            ros::spinOnce();
            rate.sleep();
        }

        offb_set_mode.request.custom_mode = "OFFBOARD";
        arm_cmd.request.value = true;
        int count = 0;

        while (ros::ok()) {
            if (current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(5.0))) {
                if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent) {
                    ROS_INFO("Offboard mode enabled");
                }
                last_request = ros::Time::now();
            } else {
                if (!current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0))) {
                    if (arming_client.call(arm_cmd) && arm_cmd.response.success) {
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }

            // target_pose.header.stamp = sp.header.stamp = ros::Time::now();
            // target_pose.position.x = sp.pose.position.x = 2.0 * cos( 1.0 /2.0 * ros::Time::now().toSec());
            // target_pose.position.y = sp.pose.position.y = 2.0 * sin( 1.0 /2.0 * ros::Time::now().toSec());

            // target_pose.velocity.x = 2.0 * 1.0 /2.0 * -sin( 1.0 /2.0 * ros::Time::now().toSec());
            // target_pose.velocity.y = 2.0 * 1.0 /2.0 * cos( 1.0 /2.0 * ros::Time::now().toSec());

            // target_pose.acceleration_or_force.x = 2.0 * 1.0 /2.0 * 1.0 /2.0 * -cos( 1.0 /2.0 * ros::Time::now().toSec());
            // target_pose.acceleration_or_force.y = 2.0 * 1.0 /2.0 * 1.0 /2.0 * -sin( 1.0 /2.0 * ros::Time::now().toSec());

            // target_pose.header.stamp = sp.header.stamp = ros::Time::now();
            // target_pose.position.x = sp.pose.position.x = planning.traj[count].pos.x();
            // target_pose.position.y = sp.pose.position.y = planning.traj[count].pos.y();
            // target_pose.position.z = sp.pose.position.z = planning.traj[count].pos.z();

            // target_pose.velocity.x = planning.traj[count].vel.x();
            // target_pose.velocity.y = planning.traj[count].vel.y();
            // target_pose.velocity.z = planning.traj[count].vel.z();

            // target_pose.acceleration_or_force.x = planning.traj[count].acc.x();
            // target_pose.acceleration_or_force.y = planning.traj[count].acc.y();
            // target_pose.acceleration_or_force.z = planning.traj[count].acc.z();

            // target_pose.yaw = planning.traj[count].yaw;
            
            // if(current_state.armed && count < planning.traj.size() -1) count++;

            local_pos_pub.publish(target_pose);
            sp_pub.publish(sp);

            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber state_sub;
    ros::Publisher local_pos_pub;
    ros::ServiceClient arming_client;
    ros::ServiceClient set_mode_client;
    mavros_msgs::State current_state;
    mavros_msgs::SetMode offb_set_mode;
    mavros_msgs::CommandBool arm_cmd;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "circle_trajectory_node");

    CircleTrajectory circle_trajectory;
    circle_trajectory.run();
    return 0;
}