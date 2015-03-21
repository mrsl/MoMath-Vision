#include "OpenCVKinect.h"

using namespace std;
using namespace cv;

int main()
{
	OpenCVKinect cap;
	if (!cap.init())
	{
		std::cout << "Error initializing" << std::endl;
		return 0;
	}

	Mat imgOriginal;
	char ch = ' ';
	while (ch != 27)
	{
		bool success = cap.read(imgOriginal, ImageType::COLOR);
		if (!success) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		imshow("Color", imgOriginal);
		ch = cv::waitKey(10);
	}
	return 1;
}