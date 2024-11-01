#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "utlz.h"
#include "physicalConstants.h"
#include "debug.h"

int segmentCount = 0;             // 线段数量
int orderedPointCount = 0;        // 有序轮廓点数量

float calculate_rotation_time(float p_x, float p_y, float max_x, float max_y, float current_yaw, float max_turn_rate, float* rotation_direction) 
{
    // 计算向量 (p_x, p_y) 和 (max_x, max_y) 的单位向量
    float delta_x = max_x - p_x;
    float delta_y = max_y - p_y;

    // 计算目标向量与x轴的角度 (以弧度为单位)
    float target_angle = atan2(delta_y, delta_x);

    // 将当前的 yaw 转换为弧度
    float current_yaw_rad = current_yaw * (M_PI_F / 180.0f);

    // 计算 yaw 与目标角度之间的夹角
    float angle_diff = target_angle - current_yaw_rad;

    // 将角度标准化到 -π 到 π 之间
    while (angle_diff > M_PI_F) angle_diff -= 2 * M_PI_F;
    while (angle_diff < -M_PI_F) angle_diff += 2 * M_PI_F;

    // 确定旋转方向
    float rotation_time = 0.0f;
    if (angle_diff < 0) 
    {
        *rotation_direction = -1.0f; // 顺时针 (速度为负)
    }
    else
    {
        *rotation_direction = 1.0f;
    }
    rotation_time = 50.0f * fabsf(angle_diff) /max_turn_rate;

    // 输出夹角大小和旋转方向
    DEBUG_PRINT("Angle difference: %f radians. Rotation time: %f\n", (double)angle_diff, (double)rotation_time);

    return rotation_time; // 返回旋转速度，用于设置转动方向
}

// 计算两点间的欧几里得距离
float euclidean_distance(float *p1, float *p2) 
{
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}


void compute_normal(Point *p1, Point *p2, float *normal) 
{
    // 法向量为两个点之间的垂直向量，长度未归一化
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;

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

void find_high_curvature_clusters_with_normals(Point *points, int num_points, float max_normal_threshold, float min_normal_threshold, float **significant_points, int *num_significant_points) {
    int max_clusters = num_points;
    *significant_points = (float *)malloc(max_clusters * 2 * sizeof(float)); // 每个点有 x 和 y 坐标
    *num_significant_points = 0;

    float *current_cluster = (float *)malloc(num_points * 2 * sizeof(float)); // 假设当前簇最多包含所有点
    int cluster_size = 0;

    for (int i = 1; i < num_points - 1; ++i) {
        // 计算当前点、前一个点和后一个点的索引
        int prev_index = (i - 1 + num_points) % num_points; // 处理封闭环
        int next_index = (i + 1) % num_points; // 处理封闭环

        Point *p1 = &points[prev_index]; // 前一个点
        Point *p2 = &points[i];          // 当前点
        Point *p3 = &points[next_index]; // 后一个点

        // 计算法向量
        float normal1[2], normal2[2];
        compute_normal(p1, p2, normal1);
        compute_normal(p2, p3, normal2);

        // 计算法向量的相似性
        float cos_theta = dot_product(normal1, normal2);

        // 判断法向量相似性是否在阈值范围内
        if (cos_theta < max_normal_threshold) {
            // 将当前点添加到当前簇
            current_cluster[cluster_size * 2] = p2->x;
            current_cluster[cluster_size * 2 + 1] = p2->y;
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
float calculate_penalty(float *point, float *significant_points, int num_significant_points, float c) 
{
    float min_distance = FLT_MAX;

    for (int i = 0; i < num_significant_points; ++i) 
    {
        float *sig_point = &significant_points[i * 2];
        float distance = euclidean_distance(point, sig_point);
        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    return -exp(-min_distance * min_distance / (2 * c * c));
}

// 对轮廓点应用惩罚
void apply_penalty(Point *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c) 
{
    for (int i = 0; i < num_contour_points; ++i) 
    {
        // 使用 contour_points[i].x 和 contour_points[i].y
        float current_point[2] = { contour_points[i].x, contour_points[i].y }; // 当前点的坐标
        float penalty = calculate_penalty(current_point, significant_points, num_significant_points, c);
        contour_points[i].y_std += penalty; // 更新 y_std
    }
}

// 将线段保存到静态数组中
void saveLineSegment(LineSegment *lineSegments, float x1, float y1, float y_std1, float x2, float y2, float y_std2) 
{
    lineSegments[segmentCount] = (LineSegment){{x1, y1, y_std1}, {x2, y2, y_std2}};
    segmentCount++;
}

// 按顺序保存轮廓点
void saveOrderedContourPoint(Point *orderedContourPoints, float x, float y, float y_std) 
{
    orderedContourPoints[orderedPointCount] = (Point){x, y, y_std};
    orderedPointCount++;
}

void marchingSquares(int grid_size, float *y_preds, float *y_stds, float x_min, float x_step, float y_min, float y_step, LineSegment *lineSegments) 
{
    for (int i = 0; i < grid_size - 1; i++) 
    {
        for (int j = 0; j < grid_size - 1; j++) 
        {
            int idx_00 = i * grid_size + j;     // 左上角
            int idx_01 = i * grid_size + (j + 1); // 右上角
            int idx_10 = (i + 1) * grid_size + j; // 左下角
            int idx_11 = (i + 1) * grid_size + (j + 1); // 右下角

            // 确定每个顶点的状态（正负）
            int topLeft     = y_preds[idx_00] > 0;
            int topRight    = y_preds[idx_01] > 0;
            int bottomLeft  = y_preds[idx_10] > 0;
            int bottomRight = y_preds[idx_11] > 0;

            // 计算当前网格单元的索引
            int cellIndex = (topLeft << 3) | (topRight << 2) | (bottomRight << 1) | bottomLeft;

            // 根据 cellIndex 的不同值，处理不同的轮廓
            switch (cellIndex) 
            {
                case 1: // 0001
                case 14: // 1110
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x1 = x_min + j * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_10]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                case 2: // 0010
                case 13: // 1101
                    {
                        float t1 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x1 = x_min + (j + 1) * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_11] - y_stds[idx_01]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_10]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                case 3: // 0011
                case 12: // 1100
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x1 = x_min + j * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + (j + 1) * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                case 4: // 0100
                case 11: // 1011
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_01]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_01] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + (j + 1) * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                case 5: // 0101
                case 10:
                    break;
                case 6: // 0110
                case 9:
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_01]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_01] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_11]) + fabsf(y_preds[idx_10]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                case 7: // 0111
                case 8: // 1000
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_00]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_00] - y_stds[idx_10]);

                        float t2 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x2 = x_min + j * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_00] + t2 * (y_stds[idx_10] - y_stds[idx_00]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2);
                    }
                    break;
                    
                case 0: //0000
                case 15: // 1111
                    // 在这种情况下，网格内部没有交点，不需要处理。
                    break;
                default:
                    // 未处理的情况可以加以记录或忽略
                    break;
            }
        }
    }
}

// 按顺序连接轮廓线段
void connectContourSegments(LineSegment *lineSegments, Point *orderedContourPoints) 
{
    int visited[25] = {0};
    int foundStart = 0;

    // 从第一条未访问的线段出发
    for (int i = 0; i < segmentCount; i++) {
        if (visited[i]) continue;

        // 第一个线段保存起点和终点
        saveOrderedContourPoint(orderedContourPoints, lineSegments[i].start.x, lineSegments[i].start.y, lineSegments[i].start.y_std);
        saveOrderedContourPoint(orderedContourPoints, lineSegments[i].end.x, lineSegments[i].end.y, lineSegments[i].end.y_std);
        visited[i] = 1;

        Point currentEnd = lineSegments[i].end;
        foundStart = 1;

        // 查找相邻线段
        while (foundStart) {
            foundStart = 0;
            for (int j = 0; j < segmentCount; j++) {
                if (!visited[j]) {
                    if (fabs(lineSegments[j].start.x - currentEnd.x) < 1e-6 &&
                        fabs(lineSegments[j].start.y - currentEnd.y) < 1e-6) {
                        // 如果起点与 currentEnd 相同，则连接，保存终点
                        saveOrderedContourPoint(orderedContourPoints, lineSegments[j].end.x, lineSegments[j].end.y, lineSegments[j].end.y_std);
                        currentEnd = lineSegments[j].end;
                        visited[j] = 1;
                        foundStart = 1;
                        break;
                    }
                    else if (fabs(lineSegments[j].end.x - currentEnd.x) < 1e-6 &&
                             fabs(lineSegments[j].end.y - currentEnd.y) < 1e-6) {
                        // 反向连接，保存起点为新的终点
                        saveOrderedContourPoint(orderedContourPoints, lineSegments[j].start.x, lineSegments[j].start.y, lineSegments[j].start.y_std);
                        currentEnd = lineSegments[j].start;
                        visited[j] = 1;
                        foundStart = 1;
                        break;
                    }
                }
            }
        }
    }
}
