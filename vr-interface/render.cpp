// vr rendering
#include <Eigen/Dense>
#include <string>
#include <iostream>

using Eigen::Matrix3f;
using Eigen::Vector3f;
using namespace std;

const int WIDTH = 1096 * 2;
const int HEIGHT = 1176;

int getVRPose(){
    Matrix3f RIGHTFORWARD_GRIPPER;
    RIGHTFORWARD_GRIPPER << 0.0, 0.0, -1.0, 
                          0.0, 1.0, 0.0, 
                          1.0, 0.0, 0.0;


    Matrix3f RIGHTUP_GRIPPER;
    RIGHTUP_GRIPPER << 0.0, 0.0, 1.0,
               0.0, 1.0, 0.0,
               1.0, 0.0, 0.0;

    const int FLOAT_SIZE = 4;

    float combined_buffer[42]={-0.68, 0.87, -0.36, -0.53, -0.85, -0.03, 0, -0.04, 1, -0.85, 0.53, 0.02, -0.92, 0.84, 0.27, -0.38, 0, 0.93, -0.69, 0.67, -0.28, -0.62, -0.74, -0.26, -0.71, 0.79, -0.04, 0.04,0.58, 0.81, -0.12, 0.81, -0.57, -0.99, -0.07, 0.11, 0.17, -0.2, 0.36, 0.24, -0.35, 0.02};
    Vector3f hmd_pos;
    Matrix3f hmd_mat;
    Vector3f left_pos;
    Matrix3f left_mat;
    Vector3f right_pos;
    Matrix3f right_mat;

    for(int i=0; i<3; i++){
    	hmd_pos(i) = combined_buffer[i];
        left_pos(i) = combined_buffer[i+12];
        right_pos(i) = combined_buffer[i+24];
    }
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            hmd_mat(i,j) = combined_buffer[i*3+j+3];
            left_mat(i,j) = combined_buffer[i*3+j+15];
            right_mat(i,j) = combined_buffer[i*3+j+27];
        }
    }

    Vector3f local_left_pos = left_pos - hmd_pos;
    Vector3f local_right_pos = right_pos - hmd_pos;
    Matrix3f mat_room2hmd = hmd_mat.inverse();

    cout<< "hmt_mat:\n" << hmd_mat <<endl;
    cout<< "mat_room2hmd:\n" << mat_room2hmd <<endl;

    float left_trigger = combined_buffer[36];
    float left_bump = combined_buffer[37];
    float left_button= combined_buffer[38];
    float left_pad= combined_buffer[39];
    
    float right_trigger = combined_buffer[40];
    float right_bump = combined_buffer[41];
    float right_button= combined_buffer[42];
    float right_pad= combined_buffer[43];

    local_left_pos = mat_room2hmd * local_left_pos; 
    local_right_pos = mat_room2hmd * local_right_pos;

    Vector3f transformed_left = -1 * local_left_pos;
    float tempValue = transformed_left(2);
    transformed_left(2) = transformed_left(1);
    transformed_left(1) = transformed_left(0);
    transformed_left(0) = tempValue;

    transformed_left(2) = -transformed_left(2) + .2;
    transformed_left(0) = transformed_left(0) + 0.15;

    Vector3f transformed_right= -1 * local_right_pos;
    tempValue = transformed_right(2);
    transformed_right(2) = transformed_right(1);
    transformed_right(1) = transformed_right(0);
    transformed_right(0) = tempValue;

    transformed_right(2) = -transformed_right(2) + .2;
    transformed_right(0) += .15;

    Matrix3f left_orientation = mat_room2hmd * left_mat;
    Matrix3f right_orientation = mat_room2hmd * right_mat;

    cout<< "transformed_left:\n" << transformed_left <<endl;
    cout<< "transformed_right:\n" << transformed_right<<endl;

    return 1;
}

int main(){
    getVRPose();
    return 0;
}