#include "OpenCVKinect.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

// max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;

//minimum and maximum object area
const double MIN_OBJECT_AREA = 20 * 20;
const double MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;

string intToString(int number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

void drawObject(int x, int y, Mat& frame)
{
	//use some of the openCV drawing functions to draw crosshairs on your tracked image!
	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
	{
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	}
	else
	{
		line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	}

	if (y + 25 < FRAME_HEIGHT)
	{
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	}
	else
	{
		line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	}

	if (x - 25 > 0)
	{
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	}
	else
	{
		line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	}

	if (x + 25 < FRAME_WIDTH)
	{
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	}
	else
	{
		line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);
	}

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

void trackFilteredObject(Mat& threshold, Mat& cameraFeed)
{
	Mat temp;
	threshold.copyTo(temp); // OpenCV performs a shallow copy to save time and space; copyTo function creates a deep copy

	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		size_t numObjects = hierarchy.size();

		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS)
		{
			int x, y = 0;
			for (int index = 0; index >= 0; index = hierarchy[index][0]) 
			{

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if ((area > MIN_OBJECT_AREA) && (area < MAX_OBJECT_AREA) && (area > refArea))
				{
					x = (int) (moment.m10 / area);
					y = (int) (moment.m01 / area);
					objectFound = true;
					refArea = area;
				}
				else
				{
					objectFound = false;
				}
			}

			//let user know you found an object
			if (objectFound == true)
			{
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else
		{
			putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
		}
	}
}

int main()
{
	OpenCVKinect cap;
	if (!cap.init())
	{
		std::cout << "Error initializing" << std::endl;
		return 0;
	}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	while (true)
	{
		// Read Image
		Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal, ImageType::COLOR);
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		//Convert the captured frame from BGR to HSV 
		Mat imgHSV;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

		// Binary Min/Max Threshold
		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

		// Morphological Operations to remove background noise
		// morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// Find Contours and Moments to track object
		trackFilteredObject(imgThresholded, imgOriginal);

		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}