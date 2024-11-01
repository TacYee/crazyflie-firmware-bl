#ifndef UTLZ_H
#define UTLZ_H

typedef struct {
    float x, y;      // 点的坐标
    float y_std;     // 点的 y_std 值
} Point;

typedef struct {
    Point start;     // 线段的起点
    Point end;       // 线段的终点
} LineSegment;

extern int segmentCount;          // 线段数量
extern int orderedPointCount;     // 有序轮廓点数量

void saveLineSegment(LineSegment *lineSegments, float x1, float y1, float y_std1, float x2, float y2, float y_std2);
void saveOrderedContourPoint(Point *orderedContourPoints, float x, float y, float y_std);
void marchingSquares(int grid_size, float *y_preds, float *y_stds, float x_min, float x_step, float y_min, float y_step, LineSegment *lineSegments);
void connectContourSegments(LineSegment *lineSegments, Point *orderedContourPoints);

// 找到曲率高的点（拐点）
void find_high_curvature_clusters_with_normals(Point *points, int num_points, float max_normal_threshold, float min_normal_threshold, float **significant_points, int *num_significant_points);

// 计算惩罚函数
float calculate_rotation_time(float p_x, float p_y, float max_x, float max_y, float current_yaw, float max_turn_rate, float* rotation_direction);

// 对轮廓点应用惩罚
void apply_penalty(Point *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c);

#endif // UTLZ_H