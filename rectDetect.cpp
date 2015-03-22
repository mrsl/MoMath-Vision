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

RNG rng(12345);

OpenCVKinect cap;
int lowThreshold = 50;
int const maxLowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

template<typename Type>
string numToString(Type number)
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

	putText(frame, numToString(x) + "," + numToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

// Color Filter Tracking
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

				for (int i = 0; i< contours.size(); i++)
				{
					Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
					drawContours(cameraFeed, contours, i, color, 2, 8, hierarchy, 0, Point());
				}
			}
		}
		else
		{
			putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
		}
	}
}

// Edge Detection
void CannyThreshold(int, void*)
{
	// Read Image
	Mat imgOriginal;
	bool bSuccess = cap.read(imgOriginal, ImageType::COLOR);
	if (!bSuccess) //if not success, break loop
	{
		cout << "Cannot read a frame from video stream" << endl;
	}

	// Convert the captured frame from BGR to GRAY 
	Mat imgGray;
	cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);

	Mat detected_edges;
	// Reduce noise with a kernel 3x3
	blur(imgGray, detected_edges, Size(3, 3));

	// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	// Using Canny's output as a mask, we display our result
	Mat canny = Mat::zeros(imgOriginal.size(), imgOriginal.type());
	imgOriginal.copyTo(canny, detected_edges);
	imshow("Thresholded Image", canny); //show the thresholded image

	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//find contours of filtered image using openCV findContours function
	findContours(detected_edges.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	vector<Point> approx;
	Mat dst = imgOriginal.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.01, true);

		if (approx.size() == 4)
		{
			// Find Center
			Moments moment = moments((cv::Mat) contours[i]);
			double area = moment.m00;

			//if the area is less than 20 px by 20px then it is probably just noise
			//if the area is the same as the 3/2 of the image size, probably just a bad filter
			//we only want the object with the largest area so we safe a reference area each
			//iteration and compare it to the area in the next iteration.
			if ((area > MIN_OBJECT_AREA) && (area < MAX_OBJECT_AREA))
			{
				int x = (int)(moment.m10 / area);
				int y = (int)(moment.m01 / area);
				drawObject(x, y, dst);

				float wx, wy, wz = 0;
				cap.distanceToPixel(x, y, wx, wy, wz);
				putText(dst, numToString(wx) + "," + numToString(wy) + "," + numToString(wz), Point(0, 50), 1, 2, Scalar(0, 0, 255), 1);
			}

			line(dst, approx.at(0), approx.at(0), cvScalar(0, 0, 255), 4);
			line(dst, approx.at(1), approx.at(1), cvScalar(0, 0, 255), 4);
			line(dst, approx.at(2), approx.at(2), cvScalar(0, 0, 255), 4);
			line(dst, approx.at(3), approx.at(3), cvScalar(0, 0, 255), 4);
		}
	}
	imshow("detected lines", dst);
}


int main()
{
	if (!cap.init())
	{
		std::cout << "Error initializing" << std::endl;
		return 0;
	}
	cap.registerDepthAndImage();

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	//Create trackbars in "Control" window
	cvCreateTrackbar("Threshold", "Control", &lowThreshold, maxLowThreshold);

	while (waitKey(30) != 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
	{
		CannyThreshold(0, 0);
	}
	return 0;
}