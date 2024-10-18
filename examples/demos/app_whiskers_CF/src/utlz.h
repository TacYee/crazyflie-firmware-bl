#ifndef UTLZ_H
#define UTLZ_H

// 计算两点间的欧几里得距离
float euclidean_distance(float *p1, float *p2);

// 计算向量 (p1->p2) 和 (p2->p3) 之间的角度
float compute_angle(float *p1, float *p2, float *p3);

// 找到曲率高的点（拐点）
void find_high_curvature_points(float *contour_points, int num_points, float **significant_points, int *num_significant_points, float angle_threshold);

// 计算惩罚函数
float calculate_penalty(float *point, float *significant_points, int num_significant_points, float c);

// 对轮廓点应用惩罚
void apply_penalty(float *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c);

#endif // UTLZ_H