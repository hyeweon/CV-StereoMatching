#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


struct MyPixel
{
    double SSD = 0;     // SSD로 계산한 dissimilarity
    double cost = 0;    // 해당 픽셀까지의 optimal cost
    int index = 0;      // optimal path의 방향
};

double mysqrt(double n);
void fSSD(Mat img1, Mat img2, int patch_size, MyPixel*** mypixel);
void fStereoMatching1(MyPixel*** mypixel, int n_rows, int n_cols, int patch_size, int constraint, double occlusion, vector<vector<vector<int>>>& path);
void fStereoMatching2(MyPixel*** mypixel, int n_rows, int n_cols, int patch_size, int constraint, double occlusion, vector<vector<vector<int>>>& path);
void fPathFind1(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer);
void fPathFind2(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer);
void fPathFind_left(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer);
void fHoleFilling1(int n_rows, int n_cols, int patch_size, uchar* upper_pointer, uchar* pointer);
void fHoleFilling2(int n_rows, int n_cols, int patch_size, uchar* pointer);

int main() {

    // 처리할 이미지 Mat으로 읽어오기
    Mat img1 = imread("im6.png", 0);
    Mat img2 = imread("im2.png", 0);

    // 처리할 이미지의 크기
    int n_rows = img1.rows;
    int n_cols = img1.cols;

    // 각 픽셀의 정보를 저장할 3차원 동적 배열
    MyPixel*** mypixel;
    mypixel = new MyPixel * *[img1.rows];
    for (int i = 0; i < n_rows; i++)
    {
        mypixel[i] = new  MyPixel * [img1.cols];
        for (int j = 0; j < n_cols; j++)
        {
            mypixel[i][j] = new MyPixel[n_cols];
        }
    }
    
    // SSD 패치 크기 (9*9)
    int patch_size = 9;

    // SSD
    fSSD(img1, img2, patch_size, mypixel);

    int constraint = 64;        // path constraint (high)
    double occlusion = 100;      // occlusion이 일어날 때 더해줄 cost

    // optimal path를 저장할 3차원 벡터
    vector<vector<vector<int>>> path(n_rows, vector<vector<int>>(0, vector<int>(2, 0)));

    // optimal path 찾아서 disparity 구하기
    fStereoMatching1(mypixel, n_rows, n_cols, patch_size, constraint, occlusion, path);
    //fStereoMatching2(mypixel, n_rows, n_cols, patch_size, constraint, occlusion, path);

   // 동적 할당 삭제
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            delete[] mypixel[i][j];
        }
    }
    for (int i = 0; i < n_rows; i++)
    {
        delete[] mypixel[i];
    }
    delete[] mypixel;

    return 0;
}

// 루트n 반환 함수
double mysqrt(double n)
{
    double x = 2;
    for (int i = 0; i < 10; i++) {
        x = (x + (n / x)) / 2;
    }

    return x;
}

// SSD
void fSSD(Mat img1, Mat img2, int patch_size, MyPixel*** mypixel) {

    // 처리할 이미지의 크기
    int n_rows = img1.rows;
    int n_cols = img1.cols;

    // 이미지 픽셀 값에 접근할 포인터
    uchar* pointer_img1;
    uchar* pointer_img2;

    // 계산 중 값 저장을 위한 변수
    double tmp = 0;
    double ans = 0;

    /*
    Mat result = Mat::zeros(n_cols, n_cols, CV_8UC1);   // DSI 출력을 위한 Mat
    double max = 0;                                     // 픽셀의 밝기 정규화를 위한 최대값
    double min = 255 * 3;                               // 픽셀의 밝기 정규화를 위한 최대값
    int row_show = 100;                                 // DSI로 출력할 row
    */

    // SSD를 이용한 dissimilarity 계산
    for (int row = patch_size / 2; row < n_rows - patch_size / 2; row++)
    {
        for (int col1 = patch_size / 2; col1 < n_cols - patch_size / 2; col1++)
        {
            for (int col2 = patch_size / 2; col2 < n_cols - patch_size / 2; col2++)
            {
                for (int i = -(patch_size / 2); i <= patch_size / 2; i++) {
                    pointer_img1 = img1.ptr<uchar>(row + i);
                    pointer_img2 = img2.ptr<uchar>(row + i);
                    for (int j = -(patch_size / 2); j <= (patch_size / 2); j++) {
                        tmp = (double)pointer_img1[col1 + j] - (double)pointer_img2[col2 + j];  // tmp = 픽셀 밝기의 차
                        ans += tmp * tmp;                                                       // patch의 tmp 제곱 모두 더하기
                    }
                }
                ans = mysqrt(ans);                      // ans = patch 내 픽셀들의 tmp^2 합의 제곱근
                mypixel[row][col1][col2].SSD = ans;     // 배열의 해당 위치에 저장
                /*
                // row_show번째 row일 경우 최대값과 최소값 찾기
                if (row == row_show) {
                    if (ans < min) {
                        min = ans;
                    }
                    if (ans > max) {
                        max = ans;
                    }
                }
                */
                ans = 0;                                // 다음 patch 계산을 위해 ans를 0으로 초기화
            }
        }
        printf("%d\n", row);
    }

    /*
    // row_show번째 row의 DSI 출력
    for (int col1 = patch_size / 2; col1 < n_cols - patch_size / 2; col1++) {
        uchar* pointer_result = result.ptr<uchar>(col1);
        for (int col2 = patch_size / 2; col2 < n_cols - patch_size / 2; col2++) {
            pointer_result[col2] = (uchar)((mypixel[row_show][col1][col2].SSD - min) / (max - min) * 255);  // 0 ~ 255의 값으로 정규화
        }
    }
    imshow("result", result);
    imwrite("DSI.jpg", result);
    waitKey(0);
    */
}

void fStereoMatching1(MyPixel*** mypixel, int n_rows, int n_cols, int patch_size, int constraint, double occlusion, vector<vector<vector<int>>>& path) {

    double cost1, cost2, cost3;                         // 각 방향의 cost를 저장할 변수

    Mat final = Mat::zeros(n_rows, n_cols, CV_8UC1);    // 최종 이미지 출력을 위한 Mat

    for (int row = patch_size / 2; row < n_rows - patch_size / 2; row++) {
        // 첫 번째 row
        for (int j = 0; j < constraint; j++) {
            cost3 = mypixel[row][patch_size / 2][patch_size / 2 + j - 1].cost + occlusion;  // →

            mypixel[row][patch_size / 2][patch_size / 2 + j].cost = cost3;
            mypixel[row][patch_size / 2][patch_size / 2 + j].index = 3;

        }

        // 나머지 row
        for (int i = patch_size / 2 + 1; i < n_cols - patch_size / 2; i++) {
            for (int j = 0; (j < constraint) && (i + j < n_cols - patch_size / 2); j++) {
                cost1 = mypixel[row][i - 1][i + j - 1].cost + mypixel[row][i][i + j].SSD;   // ↘
                cost2 = mypixel[row][i - 1][i + j].cost + occlusion;                        // ↓
                cost3 = mypixel[row][i][i + j - 1].cost + occlusion;                        // →

                if (cost2 <= cost3 && j != constraint - 1 || j == 0) {                      // j가 0이면 →방향은 고려 X
                    if (cost1 <= cost2) {
                        mypixel[row][i][i + j].cost = cost1;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 1;                                   // 해당 픽셀까지의 optimal path 방향 (1 : ↘)
                    }
                    else {
                        mypixel[row][i][i + j].cost = cost2;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 2;                                   // 해당 픽셀까지의 optimal path 방향 (2 : ↓)
                    }
                }
                else {                                                                      // j가 constraint - 1 이면 ↓방향은 고려 X
                    if (cost1 <= cost3) {
                        mypixel[row][i][i + j].cost = cost1;
                        mypixel[row][i][i + j].index = 1;
                    }
                    else {
                        mypixel[row][i][i + j].cost = cost3;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 3;                                   // 해당 픽셀까지의 optimal path 방향 (3 : →)
                    }
                }
            }
        }

        // optimal patch를 찾기 위한 가장 끝 픽셀 (backtrack 시작 픽셀)
        int col1_end = n_cols - (patch_size / 2) - 1;
        int col2_end = n_cols - (patch_size / 2) - 1;

        uchar* final_pointer = final.ptr<uchar>(row);   // 최종 이미지 pointer

        // backtrack으로 optimal patch 찾기
        fPathFind1(mypixel, n_rows, n_cols, row, col1_end, col2_end, path, final_pointer);
        //fPathFind2(mypixel, n_rows, n_cols, row, col1_end, col2_end, path, final_pointer);
        //fPathFind_left(mypixel, n_rows, n_cols, row, col1_end, col2_end, path, final_pointer);
    }

    /*
    // 위 픽셀과 같은 값으로 hole filling
    for (int row = patch_size / 2 + 1; row < n_rows - patch_size / 2; row++) {
        uchar* final_upper_pointer = final.ptr<uchar>(row - 1);
        uchar* final_pointer = final.ptr<uchar>(row);
        fHoleFilling1(n_rows, n_cols, patch_size, final_upper_pointer, final_pointer);
    }
    */
    
    /*
    // 오른쪽 픽셀과 같은 값으로 hole filling
    for (int row = patch_size / 2 + 1; row < n_rows - patch_size / 2; row++) {
        uchar* final_pointer = final.ptr<uchar>(row);
        fHoleFilling2(n_rows, n_cols, patch_size, final_pointer);
    }
    */

    // 최종 이미지 출력
    imshow("final", final);
    imwrite("final.jpg", final);
    waitKey(0);
}

// 이미지의 양 끝에서 occlusion cost를 1/2로 적용
void fStereoMatching2(MyPixel*** mypixel, int n_rows, int n_cols, int patch_size, int constraint, double occlusion, vector<vector<vector<int>>>& path) {

    double cost1, cost2, cost3;                         // 각 방향의 cost를 저장할 변수

    Mat final = Mat::zeros(n_rows, n_cols, CV_8UC1);    // 최종 이미지 출력을 위한 Mat

    for (int row = patch_size / 2; row < n_rows - patch_size / 2; row++) {
        // 첫 번째 row
        for (int j = 0; j < constraint; j++) {
            cost3 = mypixel[row][patch_size / 2][patch_size / 2 + j - 1].cost + occlusion / 2;  // → (occlusion cost를 1/2로 적용)

            mypixel[row][patch_size / 2][patch_size / 2 + j].cost = cost3;
            mypixel[row][patch_size / 2][patch_size / 2 + j].index = 3;

        }

        // 중간 row
        for (int i = patch_size / 2 + 1; i < n_cols - patch_size / 2; i++) {
            for (int j = 0; (j < constraint) && (i + j < n_cols - patch_size / 2 - 1); j++) {
                cost1 = mypixel[row][i - 1][i + j - 1].cost + mypixel[row][i][i + j].SSD;   // ↘
                cost2 = mypixel[row][i - 1][i + j].cost + occlusion;                        // ↓
                cost3 = mypixel[row][i][i + j - 1].cost + occlusion;                        // →

                if (cost2 <= cost3 && j != constraint - 1 || j == 0) {                      // j가 0이면 →방향은 고려 X
                    if (cost1 <= cost2) {
                        mypixel[row][i][i + j].cost = cost1;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 1;                                   // 해당 픽셀까지의 optimal path 방향 (1 : ↘)
                    }
                    else {
                        mypixel[row][i][i + j].cost = cost2;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 2;                                   // 해당 픽셀까지의 optimal path 방향 (2 : ↓)
                    }
                }
                else {                                                                      // j가 constraint - 1 이면 ↓방향은 고려 X
                    if (cost1 <= cost3) {
                        mypixel[row][i][i + j].cost = cost1;
                        mypixel[row][i][i + j].index = 1;
                    }
                    else {
                        mypixel[row][i][i + j].cost = cost3;                                // 해당 픽셀까지의 optimal cost
                        mypixel[row][i][i + j].index = 3;                                   // 해당 픽셀까지의 optimal path 방향 (3 : →)
                    }
                }
            }
        }

        // 마지막 row
        for (int i = n_cols - patch_size / 2 - constraint; i < n_cols - patch_size / 2; i++) {
            cost1 = mypixel[row][i - 1][n_cols - patch_size / 2 - 2].cost + mypixel[row][i][n_cols - patch_size / 2 - 1].SSD;   // ↘
            cost2 = mypixel[row][i - 1][n_cols - patch_size / 2 - 1].cost + occlusion / 2;                                      // ↓ (occlusion cost를 1/2로 적용)
            cost3 = mypixel[row][i - 1][n_cols - patch_size / 2 - 2].cost + occlusion;                                          // →

            if (cost2 <= cost3 && i != n_cols - patch_size / 2 - constraint || i == n_cols - patch_size / 2 - 1) {
                if (cost1 <= cost2) {
                    mypixel[row][i][n_cols - patch_size / 2 - 1].cost = cost1;
                    mypixel[row][i][n_cols - patch_size / 2 - 1].index = 1;
                }
                else {
                    mypixel[row][i][n_cols - patch_size / 2 - 1].cost = cost2;
                    mypixel[row][i][n_cols - patch_size / 2 - 1].index = 2;
                }
            }
            else {
                if (cost1 <= cost3) {
                    mypixel[row][i][n_cols - patch_size / 2 - 1].cost = cost1;
                    mypixel[row][i][n_cols - patch_size / 2 - 1].index = 1;
                }
                else {
                    mypixel[row][i][n_cols - patch_size / 2 - 1].cost = cost3;
                    mypixel[row][i][n_cols - patch_size / 2 - 1].index = 3;
                }
            }
        }

        // optimal patch를 찾기 위한 가장 끝 픽셀 (backtrack 시작 픽셀)
        int col1_end = n_cols - (patch_size / 2) - 1;
        int col2_end = n_cols - (patch_size / 2) - 1;

        uchar* final_pointer = final.ptr<uchar>(row);   // 최종 이미지 pointer

        // backtrack으로 optimal patch 찾기
        fPathFind1(mypixel, n_rows, n_cols, row, col1_end, col2_end, path, final_pointer);
        //fPathFind2(mypixel, n_rows, n_cols, row, col1_end, col2_end, path, final_pointer);
    }

    /*
    // 위 픽셀과 같은 값으로 hole filling
    for (int row = patch_size / 2 + 1; row < n_rows - patch_size / 2; row++) {
        uchar* final_upper_pointer = final.ptr<uchar>(row - 1);
        uchar* final_pointer = final.ptr<uchar>(row);
        fHoleFilling1(n_rows, n_cols, patch_size, final_upper_pointer, final_pointer);
    }
    */

    /*
    // 오른쪽 픽셀과 같은 값으로 hole filling
    for (int row = patch_size / 2 + 1; row < n_rows - patch_size / 2; row++) {
        uchar* final_pointer = final.ptr<uchar>(row);
        fHoleFilling2(n_rows, n_cols, patch_size, final_pointer);
    }
    */

    // 최종 이미지 출력
    imshow("final", final);
    imwrite("final.jpg", final);
    waitKey(0);
}

// disparity 그대로 이미지 출력
void fPathFind1(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer) {

    vector<int>index_v(2, 0);                               // path의 node가 되는 픽셀 위치를 저장할 벡터

    //Mat result = Mat::zeros(n_cols, n_cols, CV_8UC3);       // optimal path 이미지 출력을 위한 Mat
    //int row_show = 100;                                     // 이미지로 출력할 row

    while (index_col1 >= 4 && index_col2 >= 4) {
        // 픽셀 위치를 path에 저장
        index_v[0] = index_col1;
        index_v[1] = index_col2;
        path[row].insert(path[row].begin(), index_v);

        // row_show번째 row의 optimal path를 이미지에 빨간색으로 표시
        //if (row == row_show) { result.at<Vec3b>(index_col1, index_col2)[2] = 255; }

        if (mypixel[row][index_col1][index_col2].index == 1) {                          // optimal path의 방향이 ↖일 경우
            pointer[index_col1] = (path[row][0][1] - path[row][0][0]) * 255 / 64;       // 최종 이미지에 해당 픽셀을 disparity 값으로 표시
            index_col1--;
            index_col2--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 2) {                     // optimal path의 방향이 ↑일 경우
            index_col1--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 3) {                     // optimal path의 방향이 ←일 경우
            index_col2--;
        }
    }

    // row_show번째 row의 optiaml path를 이미지로 출력
    /*
    if (row == row_show) {
        printf("ㅎㅎ");
        imshow("result", result);
        imwrite("optimalpath.jpg", result);
        waitKey(0);
    }
    */
}

// depth = 1 / disparity 로 이미지 출력
void fPathFind2(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer) {

    vector<int>index_v(2, 0);                               // path의 node가 되는 픽셀 위치를 저장할 벡터

    //Mat result = Mat::zeros(n_cols, n_cols, CV_8UC3);       // optimal path 이미지 출력을 위한 Mat
    //int row_show = 100;                                     // 이미지로 출력할 row

    while (index_col1 >= 4 && index_col2 >= 4) {
        // 픽셀 위치를 path에 저장
        index_v[0] = index_col1;
        index_v[1] = index_col2;
        path[row].insert(path[row].begin(), index_v);

        // row_show번째 row의 optimal path를 이미지에 빨간색으로 표시
        //if (row == row_show) { result.at<Vec3b>(index_col1, index_col2)[2] = 255; }

        if (mypixel[row][index_col1][index_col2].index == 1) {                                      // optimal path의 방향이 ↖일 경우
            if ((path[row][0][1] - path[row][0][0]) == 0) {
                pointer[index_col1] = 0;
                cout << "0" << endl;    
            }
            else {
                pointer[index_col1] = 255 - (255.0 * 12 / (path[row][0][1] - path[row][0][0]));      // 최종 이미지에 해당 픽셀을 depth(1 / disparity) 값으로 표시
            }
            index_col1--;
            index_col2--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 2) {                                 // optimal path의 방향이 ↑일 경우
            index_col1--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 3) {                                 // optimal path의 방향이 ←일 경우
            index_col2--;
        }
    }

    // row_show번째 row의 optiaml path를 이미지로 출력
    /*
    if (row == row_show) {
        printf("ㅎㅎ");
        imshow("result", result);
        imwrite("optimalpath.jpg", result);
        waitKey(0);
    }
    */
}

// 왼쪽 이미지 출력
void fPathFind_left(MyPixel*** mypixel, int n_rows, int n_cols, int row, int index_col1, int index_col2, vector<vector<vector<int>>>& path, uchar* pointer) {

    vector<int>index_v(2, 0);                               // path의 node가 되는 픽셀 위치를 저장할 벡터

    //Mat result = Mat::zeros(n_cols, n_cols, CV_8UC3);       // optimal path 이미지 출력을 위한 Mat
    //int row_show = 100;                                     // 이미지로 출력할 row

    while (index_col1 >= 4 && index_col2 >= 4) {
        // 픽셀 위치를 path에 저장
        index_v[0] = index_col1;
        index_v[1] = index_col2;
        path[row].insert(path[row].begin(), index_v);

        // row_show번째 row의 optimal path를 이미지에 빨간색으로 표시
        //if (row == row_show) { result.at<Vec3b>(index_col1, index_col2)[2] = 255; }

        if (mypixel[row][index_col1][index_col2].index == 1) {                          // optimal path의 방향이 ↖일 경우
            pointer[index_col2] = (path[row][0][1] - path[row][0][0]) * 255 / 64;       // 최종 이미지에 해당 픽셀을 disparity 값으로 표시
            index_col1--;
            index_col2--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 2) {                     // optimal path의 방향이 ↑일 경우
            index_col1--;
        }
        else if (mypixel[row][index_col1][index_col2].index == 3) {                     // optimal path의 방향이 ←일 경우
            index_col2--;
        }
    }

    // row_show번째 row의 optiaml path를 이미지로 출력
    /*
    if (row == row_show) {
        printf("ㅎㅎ");
        imshow("result", result);
        imwrite("optimalpath.jpg", result);
        waitKey(0);
    }
    */
}

// 위 픽셀과 같은 값으로 hole filling
void fHoleFilling1(int n_rows, int n_cols, int patch_size, uchar* upper_pointer, uchar* pointer) {
    for (int i = patch_size / 2; i < n_cols - patch_size / 2; i++) {
        if (pointer[i] == 0) {
            pointer[i] = upper_pointer[i];
        }
    }
}

// 오른쪽 픽셀과 같은 값으로 hole filling
void fHoleFilling2(int n_rows, int n_cols, int patch_size, uchar* pointer) {
    double tmp = 0;
    for (int i = n_cols - patch_size / 2 - 1; i >= patch_size / 2; i--) {
        if (pointer[i] == 0) {
            pointer[i] = tmp;
        }
        else {
            tmp = pointer[i];
        }
    }
}