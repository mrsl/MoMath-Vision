// *******************************************************************************
//	OpenCVKinect: Provides method to access Kinect Color and Depth Stream        *
//				  in OpenCV Mat format.                                           *
//                                                                                *
//				  Pre-requisites: OpenCV_2.x, OpenNI_2.x, KinectSDK_1.8           *
//                                                                                *
//   Copyright (C) 2013  Muhammad Asad                                            *
//                       Webpage: http://seevisionc.blogspot.co.uk/p/about.html   *
//						 Contact: masad.801@gmail.com                             *
//                                                                                *
//   This program is free software: you can redistribute it and/or modify         *
//   it under the terms of the GNU General Public License as published by         *
//   the Free Software Foundation, either version 3 of the License, or            *
//   (at your option) any later version.                                          *
//                                                                                *
//   This program is distributed in the hope that it will be useful,              *
//   but WITHOUT ANY WARRANTY; without even the implied warranty of               *
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
//   GNU General Public License for more details.                                 *
//                                                                                *
//   You should have received a copy of the GNU General Public License            *
//   along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
//                                                                                *
// *******************************************************************************

#include "OpenCVKinect.h"

OpenCVKinect::OpenCVKinect(void)
{
	m_depthTimeStamp = 0;
	m_colorTimeStamp = 0;
}

bool OpenCVKinect::init()
{
	openni::Status m_status = openni::Status::STATUS_OK;
	m_status = openni::OpenNI::initialize();
	if (m_status != openni::STATUS_OK)
	{
		std::cout << "OpenNI Initialization Error: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
	}

	// open the device
	m_status = m_device.open(openni::ANY_DEVICE);
	if (m_status != openni::STATUS_OK)
	{
		std::cout << "OpenCVKinect: Device open failseed: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	// create a depth object
	m_status = m_depth.create(m_device, openni::SENSOR_DEPTH);
	if (m_status == openni::STATUS_OK)
	{
		m_status = m_depth.start();
		if (m_status != openni::STATUS_OK)
		{
			std::cout << "OpenCVKinect: Couldn't start depth stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			m_depth.destroy();
			return false;
		}
	}
	else
	{
		std::cout << "OpenCVKinect: Couldn't find depth stream: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		return false;
	}

	// create a color object
	m_status = m_color.create(m_device, openni::SENSOR_COLOR);
	if (m_status == openni::STATUS_OK)
	{
		m_status = m_color.start();
		if (m_status != openni::STATUS_OK)
		{

			std::cout << "OpenCVKinect: Couldn't start color stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			m_color.destroy();
			return false;
		}
	}
	else
	{
		std::cout << "OpenCVKinect: Couldn't find color stream: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		return false;
	}

	if (!m_depth.isValid() && !m_color.isValid())
	{
		std::cout << "OpenCVKinect: No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	this->m_streams = new openni::VideoStream*[C_NUM_STREAMS];
	m_streams[C_DEPTH_STREAM] = &m_depth;
	m_streams[C_COLOR_STREAM] = &m_color;

	return true;
}

bool OpenCVKinect::read(cv::Mat& returnImage, ImageType type)
{
	openni::Status m_status = openni::OpenNI::waitForAnyStream(m_streams, C_NUM_STREAMS, &m_currentStream, C_STREAM_TIMEOUT);
	if (m_status != openni::STATUS_OK)
	{
		std::cout << "OpenCVKinect: Unable to wait for streams. Exiting" << std::endl;
		return false;
	}
	cv::Mat bufferImage;
	switch (type)
	{
	case ImageType::COLOR:
		{
			openni::VideoFrameRef m_colorFrame;
			m_color.readFrame(&m_colorFrame);
			returnImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
			bufferImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
			bufferImage.data = (uchar*)m_colorFrame.getData();
			this->m_colorTimeStamp = m_colorFrame.getTimestamp() >> 16;
			std::cout << "Color Timestamp: " << m_colorTimeStamp << std::endl;
			cv::cvtColor(bufferImage, returnImage, CV_BGR2RGB);
			m_colorFrame.release();
			break;
		}
	case ImageType::DEPTH:
		{
			openni::VideoFrameRef m_depthFrame;
			m_depth.readFrame(&m_depthFrame);
			returnImage.create(m_depthFrame.getHeight(), m_depthFrame.getWidth(), CV_16UC1);
			returnImage.data = (uchar*)m_depthFrame.getData();
			this->m_depthTimeStamp = m_depthFrame.getTimestamp() >> 16;
			std::cout << "Depth Timestamp: " << this->m_depthTimeStamp << std::endl;
			m_depthFrame.release();
			break;
		}
	}
	bufferImage.release();
	return true;
}

openni::Status OpenCVKinect::distanceToPixel(int x, int y, float& wx, float& wy, float& wz)
{
	openni::VideoFrameRef m_depthFrame;
	m_depth.readFrame(&m_depthFrame);
	openni::DepthPixel* pDepth = (openni::DepthPixel*) m_depthFrame.getData();
	int pos = y * m_depthFrame.getWidth() + x;
	return openni::CoordinateConverter::convertDepthToWorld(m_depth, x, y, pDepth[pos], &wx, &wy, &wz);
}

openni::Status OpenCVKinect::registerDepthAndImage()
{
	if (m_device.isValid())
	{
		return m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	}
	return openni::Status::STATUS_NO_DEVICE;
}

OpenCVKinect::~OpenCVKinect(void)
{
	this->m_depth.stop();
	this->m_color.stop();
	openni::OpenNI::shutdown();
	this->m_device.close();
}
