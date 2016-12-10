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

	//�ʿ��� ���� �� ��ü ����
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


	//���۷��� �̹����� �ε��մϴ�.
	Mat object = imread("411.jpg", 1);

	if (!object.data)
	{
		std::cout << "Error reading object " << std::endl;
		return -1;
	}


	
	//���۷��� �̹������� Ư¡�� ����� ���� keypoint�� ��´�.
	detector->detect(object, kp_object);
	extractor->compute(object, kp_object, des_object);
	
	
	//���� ������ �ҷ� ���ų� ķ�� �����Ű�� ���� ��ü�� ����
	VideoCapture cap;
	cap.open("move1.mp4");
	//VideoCapture cap(0);//�̷������� ����ϸ� ����Ʈ������ �ǽð� ķ�� ����� �� �ִ�.

	//���۷��� �̹����κ��� �� �ڳ������� ��´�.
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(object.cols, 0);
	obj_corners[2] = cvPoint(object.cols, object.rows);
	obj_corners[3] = cvPoint(0, object.rows);

	//escŰ�� �Է¹ޱ� ���� ���ѷ����� ���� �ǽð� ���� ó���� ����,
	while (key != 27)
	{
		//�������� �������� ĸó�Ͽ� ������ �̹��� �̸� 'frame'�� ����
		cap >> frame;

		if (framecount < 50)
		{
			framecount++;
			continue;
		}

		//ĸ�ĵ� �̹����� �׷��̽����Ϸ� ��ȯ
		cvtColor(frame, image, CV_RGB2GRAY);

		//ĸó�� �������� ����⸦ ���Ͽ� ����.
		detector->detect(image, kp_image);
		extractor->compute(image, kp_image, des_image);

		//���� �� ĸó�� �̹����� ��ġ�ϴ� �̹����� ã���ϴ�.
		matcher.knnMatch(des_object, des_image, matches, 2);

		//�������� �Ÿ��� ���� �Ͱ� ���� ������ ���� keypoints �Ÿ��� �Ʒ��� ���� �Ÿ����� ����
		for (int i = 0; i < min(des_image.rows - 1, (int)matches.size()); i++)
		{
			if ((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size() > 0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}

		//good match�� �׸���.
		drawMatches(object, kp_object, frame, kp_image, good_matches, img_matches,
			Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//3 good matches are enough to describe an object as a right match.
		//3���� ��Ī�Ȱ��� �������̹����� ǥ��Ǳ⿡ ����ϱ� ������ 3���̻����� 3���̻��Ͻð� ǥ���Ѵ�.
		if (good_matches.size() >= 3)
		{

			for (int i = 0; i < good_matches.size(); i++)
			{
				//good match�� ���� Ű����Ʈ���� ��´�.
				obj.push_back(kp_object[good_matches[i].queryIdx].pt);
				scene.push_back(kp_image[good_matches[i].trainIdx].pt);
			}
			try
			{
				H = findHomography(obj, scene, CV_RANSAC);
			}
			catch (Exception e) {}

			perspectiveTransform(obj_corners, scene_corners, H);

			//cam���� ��ü ����� �簢���� �׸���.
			line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
		}

		//����� ��Ī ���
		imshow("Matching", img_matches);

		//����� �迭 �ʱ�ȭ
		good_matches.clear();

		key = waitKey(1);
	}
	return 0;
}





