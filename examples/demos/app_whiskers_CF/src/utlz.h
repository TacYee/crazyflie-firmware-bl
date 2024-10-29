#ifndef UTLZ_H
#define UTLZ_H

// 找到曲率高的点（拐点）
void find_high_curvature_clusters_with_normals(float *points, int num_points, float max_normal_threshold, float min_normal_threshold, float **significant_points, int *num_significant_points);

// 计算惩罚函数
float calculate_rotation_time(float p_x, float p_y, float max_x, float max_y, float current_yaw, float max_turn_rate, float* rotation_direction);

// 对轮廓点应用惩罚
void apply_penalty(float *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c);

#endif // UTLZ_H