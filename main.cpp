#include <stdio.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "opencv2/core/core.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

//surf matching
int main()
{

	//필요한 변수 및 객체 선언
	char key = 'a';
	int framecount = 0;


	
	Ptr<ORB> orb = ORB::create(500);
	Ptr<SURF> detector = SURF::create(500);
	Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();
	FlannBasedMatcher matcher;
	
	Mat frame, des_object, image;
	Mat des_image, img_matches, H;

	std::vector<KeyPoint> kp_object;
	std::vector<Point2f> obj_corners(4);
	std::vector<KeyPoint> kp_image;
	std::vector<std::vector<DMatch>> matches;
	std::vector<DMatch > good_matches;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	std::vector<Point2f> scene_corners(4);


	//레퍼런스 이미지를 로딩합니다.
	Mat object = imread("411.jpg", 1);

	if (!object.data)
	{
		std::cout << "Error reading object " << std::endl;
		return -1;
	}


	
	//레퍼런스 이미지에서 특징점 검출될 것을 keypoint에 담는다.
	detector->detect(object, kp_object);
	extractor->compute(object, kp_object, des_object);
	
	
	//비디오 영상을 불러 오거나 캠을 실행시키기 위해 객체를 생성
	VideoCapture cap;
	cap.open("move1.mp4");
	//VideoCapture cap(0);//이런식으로 사용하면 디폴트값으로 실시간 캠을 사용할 수 있다.

	//레퍼런스 이미지로부터 각 코너점들을 얻는다.
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(object.cols, 0);
	obj_corners[2] = cvPoint(object.cols, object.rows);
	obj_corners[3] = cvPoint(0, object.rows);

	//esc키를 입력받기 전에 무한루프를 돌아 실시간 영상 처리를 위한,
	while (key != 27)
	{
		//비디오에서 프레임을 캡처하여 영상을 이미지 이름 'frame'에 저장
		cap >> frame;

		if (framecount < 50)
		{
			framecount++;
			continue;
		}

		//캡쳐된 이미지는 그레이스케일로 변환
		cvtColor(frame, image, CV_RGB2GRAY);

		//캡처된 프레임의 검출기를 통하여 추출.
		detector->detect(image, kp_image);
		extractor->compute(image, kp_image, des_image);

		//참조 및 캡처된 이미지의 일치하는 이미지를 찾습니다.
		matcher.knnMatch(des_object, des_image, matches, 2);

		//기하학적 거리를 곱한 것과 같은 간격을 가진 keypoints 거리를 아래와 같은 거리에서 측정
		for (int i = 0; i < min(des_image.rows - 1, (int)matches.size()); i++)
		{
			if ((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size() > 0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}

		//good match만 그린다.
		drawMatches(object, kp_object, frame, kp_image, good_matches, img_matches,
			Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//3 good matches are enough to describe an object as a right match.
		//3개의 매칭된것은 오른쪽이미지에 표기되기에 충분하기 때문에 3개이상으로 3개이상일시간 표기한다.
		if (good_matches.size() >= 3)
		{

			for (int i = 0; i < good_matches.size(); i++)
			{
				//good match로 부터 키포인트들을 얻는다.
				obj.push_back(kp_object[good_matches[i].queryIdx].pt);
				scene.push_back(kp_image[good_matches[i].trainIdx].pt);
			}
			try
			{
				H = findHomography(obj, scene, CV_RANSAC);
			}
			catch (Exception e) {}

			perspectiveTransform(obj_corners, scene_corners, H);

			//cam위에 객체 검출된 사각형을 그린다.
			line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
		}

		//검출된 매칭 출력
		imshow("Matching", img_matches);

		//검출된 배열 초기화
		good_matches.clear();

		key = waitKey(1);
	}
	return 0;
}





