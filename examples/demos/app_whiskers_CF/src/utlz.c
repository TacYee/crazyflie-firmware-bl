#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "utlz.h"

#define PI 3.14159265358979323846

// 计算两点间的欧几里得距离
float euclidean_distance(float *p1, float *p2) {
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}


void compute_normal(float *p1, float *p2, float *normal) {
    // 法向量为两个点之间的垂直向量，长度未归一化
    float dx = p2[0] - p1[0];
    float dy = p2[1] - p1[1];

    // 垂直向量 (dy, -dx) 或 (-dy, dx)，根据右手法则选取
    normal[0] = dy;
    normal[1] = -dx;

    // 归一化
    float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
    if (length != 0) {
        normal[0] /= length;
        normal[1] /= length;
    }
}

// 计算法向量相似性
float dot_product(float *v1, float *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

// 计算每个簇的质心
void compute_centroid(float *cluster, int cluster_size, float *centroid) 
{
    centroid[0] = 0;
    centroid[1] = 0;
    for (int i = 0; i < cluster_size; ++i) {
        centroid[0] += cluster[i * 2];
        centroid[1] += cluster[i * 2 + 1];
    }
    centroid[0] /= cluster_size;
    centroid[1] /= cluster_size;
}

// 找出曲率大且法向量相似的连续点簇
void find_high_curvature_clusters_with_normals(float *points, int num_points, float max_normal_threshold, float min_normal_threshold, float **significant_points, int *num_significant_points) {
    int max_clusters = num_points / 2;
    *significant_points = (float *)malloc(max_clusters * 2 * sizeof(float));
    *num_significant_points = 0;

    float *current_cluster = (float *)malloc(num_points * 2 * sizeof(float)); // 假设当前簇最多包含所有点
    int cluster_size = 0;

    for (int i = 1; i < num_points - 1; ++i) {
        float *p1 = &points[(i - 1) * 2];
        float *p2 = &points[i * 2];
        float *p3 = &points[(i + 1) % num_points * 2];

        // 计算法向量
        float normal1[2], normal2[2];
        compute_normal(p1, p2, normal1);
        compute_normal(p2, p3, normal2);

        // 计算法向量的相似性
        float cos_theta = dot_product(normal1, normal2);

        // 判断法向量相似性是否在阈值范围内
        if (cos_theta < max_normal_threshold) {
            // 将当前点添加到当前簇
            current_cluster[cluster_size * 2] = p2[0];
            current_cluster[cluster_size * 2 + 1] = p2[1];
            cluster_size++;

            // 如果法向量相似性小于最小阈值，结束当前簇
            if (cos_theta < min_normal_threshold) {
                // 计算当前簇的质心
                float centroid[2];
                compute_centroid(current_cluster, cluster_size, centroid);

                // 将质心添加到显著点数组
                (*significant_points)[(*num_significant_points) * 2] = centroid[0];
                (*significant_points)[(*num_significant_points) * 2 + 1] = centroid[1];
                (*num_significant_points)++;

                // 清空当前簇
                cluster_size = 0;
            }
        } else {
            // 如果当前簇不为空，结束当前簇并计算质心
            if (cluster_size > 0) {
                float centroid[2];
                compute_centroid(current_cluster, cluster_size, centroid);

                (*significant_points)[(*num_significant_points) * 2] = centroid[0];
                (*significant_points)[(*num_significant_points) * 2 + 1] = centroid[1];
                (*num_significant_points)++;

                cluster_size = 0;
            }
        }
    }

    // 如果最后还有未处理的簇
    if (cluster_size > 0) {
        float centroid[2];
        compute_centroid(current_cluster, cluster_size, centroid);

        (*significant_points)[(*num_significant_points) * 2] = centroid[0];
        (*significant_points)[(*num_significant_points) * 2 + 1] = centroid[1];
        (*num_significant_points)++;
    }

    free(current_cluster);
}

// 计算惩罚函数
float calculate_penalty(float *point, float *significant_points, int num_significant_points, float c) {
    float min_distance = FLT_MAX;

    for (int i = 0; i < num_significant_points; ++i) {
        float *sig_point = &significant_points[i * 2];
        float distance = euclidean_distance(point, sig_point);
        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    return -exp(-min_distance * min_distance / (2 * c * c));
}

// 对轮廓点应用惩罚
void apply_penalty(float *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c) {
    for (int i = 0; i < num_contour_points; ++i) {
        float *point = &contour_points[i * 2];
        float penalty = calculate_penalty(point, significant_points, num_significant_points, c);
        y_stds[i] += penalty;
    }
}
