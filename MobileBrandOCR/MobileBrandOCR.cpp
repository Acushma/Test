
#include "ml.h"
#include "highgui.h"
#include "cv.h"
#include "cxcore.h"
#include "MobileBrandOCR.h"
#include "BlobResult.h"

#include "stdio.h"

#define BLOCK_THRESHOLD 10

char wndname[] = "Edge";
char tbarname[] = "Threshold";
int edge_thresh = 1;
IplImage *image = 0, *cedge = 0, *gray = 0, *edge = 0, *bina = 0;



int size = 16;
int classes = 8;//7 letters
int train_samples = 30;
int test_samples = 30;
int sample_dimension = size*size;
int K = 10;


char file_path[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/print data/Result/";
char result_file_path[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/print data/ScaledResult/";
char result_file_name[255];


CvMat* trainClasses = cvCreateMat( classes * train_samples, 1, CV_32FC1 );
CvMat* trainData = cvCreateMat( classes * train_samples, sample_dimension, CV_32FC1 );

void findX(IplImage* imgSrc,int* min, int* max)
{
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->height * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < height*255 then we find the min
	//then continue to end to search the max, if sum < width*255 then is new max
	for (i=0; i<imgSrc->width; i++)
	{
		cvGetCol(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0])
		{
			*max= i;
			if(!minFound)
			{
				*min= i;
				minFound= 1;
			}
		}
	}
}
 
void findY(IplImage* imgSrc,int* min, int* max)
{
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum < width*255 then is new max
	for (i=0; i< imgSrc->height; i++)
	{
		cvGetRow(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0])
		{
			*max=i;
			if(!minFound)
			{
				*min= i;
				minFound= 1;
			}
		}
	}
}

CvRect findBB(IplImage* imgSrc)
{
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin=xmax=ymin=ymax=0;
 
	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);
 
	aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);
 
	//printf("BB: %d,%d - %d,%d\n", aux.x, aux.y, aux.width, aux.height);
 
	return aux;
}

IplImage numberSegmentation(IplImage* sourceImage)
{
	int x=0,y=0;
    int lo_diff = 0, up_diff = 0;
    int connectivity = 4;
    int new_mask_val = 255;
    int flags = connectivity + (new_mask_val << 8)+CV_FLOODFILL_FIXED_RANGE+CV_FLOODFILL_MASK_ONLY;
    CvConnectedComp comp;  
	double max_area=0;
	int max_x=0,max_y=0;
	int max_area_seed_x=0,max_area_seed_y=0;
	int b=0, g=0, r=0;
	CvPoint seed;
	CvScalar brightness;
	
	int width = sourceImage->width;
	int height = sourceImage->height;
	
	IplImage *grayImage  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);//can re move
	IplImage *binaryImage  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
	IplImage *maskImage  = cvCreateImage(cvSize(width+2,height+2),IPL_DEPTH_8U,1);
	
	CvMat *binaryMat = cvCreateMat(height, width,  CV_32FC1 );
	CvMat *maskMat	 = cvCreateMat( height+2,width+2, CV_32FC1 );
	
	cvCvtColor(sourceImage, grayImage, CV_BGR2GRAY);
	cvThreshold(grayImage, binaryImage, 50, 255, CV_THRESH_BINARY);

	cvNamedWindow( "Source", 1 );
	cvShowImage( "Source", binaryImage );
	cvWaitKey(0);

/*
	cvConvert(binaryImage, binaryMat);
	//cvSmooth(binaryMat, binaryMat, CV_GAUSSIAN, 9, 0, 0);
	//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
	//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
	//cvConvert(binaryMat, binaryImage);
	
	cvZero(maskImage);
	cvConvert(maskImage,maskMat);
	for(y=0;y<height;y++)
		for(x=0;x<width;x++)
		{	 
			if(*(maskMat->data.fl+(y+1)*(width+2)+(x+1))==0 && *(binaryMat->data.fl+y*width+x)==0)
			{
				cvThreshold(maskImage,maskImage, 1, 128, CV_THRESH_BINARY);
				seed = cvPoint(x,y);
				b=255, g=255, r=255;
				brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
                
				cvFloodFill(binaryImage, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags, maskImage);   
				
				if(comp.area>max_area)
				{
					max_area=comp.area;
					max_area_seed_x=x;
					max_area_seed_y=y;
				}
			}
		}
		
		//printf("max_area = %f\n",max_area);
		
		if(max_area < BLOCK_THRESHOLD)
		{
			cvReleaseImage(&sourceImage);
			cvReleaseImage(&grayImage);
			cvReleaseImage(&binaryImage);
			cvReleaseImage(&maskImage);
			cvReleaseMat(&binaryMat);
			cvReleaseMat(&maskMat);
			binaryImage = NULL;
		}
		else
		{
			cvZero(maskImage);
			seed = cvPoint(max_area_seed_x,max_area_seed_y);
			b=255, g=255, r=255;
			brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
			
			cvFloodFill( binaryImage, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags,maskImage); 
			//printf("comp.area = %f\n",comp.area);
			
			cvConvert(maskImage,maskMat);
			
			for(int i=0;i<height;i++)
				for(int j=0;j<width;j++)
				{
					
					if(*(maskMat->data.fl+(i+1)*(width+2)+(j+1))!=0)
					{
						*(binaryMat->data.fl+i*width+j)=0;
					}
					else
					{
						*(binaryMat->data.fl+i*width+j)=255;
					}
				}
				
				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
				cvConvert(binaryMat, binaryImage);
		}
*/		
		return *binaryImage;
}


IplImage numberSegmentationBasedOnContour(IplImage* sourceImage)
{
	int x=0,y=0;
    int lo_diff = 0, up_diff = 0;
    int connectivity = 4;
    int new_mask_val = 255;
    int flags = connectivity + (new_mask_val << 8)+CV_FLOODFILL_FIXED_RANGE+CV_FLOODFILL_MASK_ONLY;
    CvConnectedComp comp;  
	double max_area=0;
	int max_x=0,max_y=0;
	int max_area_seed_x=0,max_area_seed_y=0;
	int b=0, g=0, r=0;
	CvPoint seed;
	CvScalar brightness;
	
	int width = sourceImage->width;
	int height = sourceImage->height;
	
	IplImage *grayImage  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);//can remove
	IplImage *binaryImage  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
	IplImage *maskImage  = cvCreateImage(cvSize(width+2,height+2),IPL_DEPTH_8U,1);
	
	CvMat *binaryMat = cvCreateMat(height, width,  CV_32FC1 );
	CvMat *maskMat	 = cvCreateMat( height+2,width+2, CV_32FC1 );



	CvMemStorage* storage = cvCreateMemStorage( 0 );	
	CvSeq* contour = NULL;	
	int find_contour_num;
	CvScalar red = CV_RGB( 255, 255, 255 );
	IplImage* dst = cvCreateImage( cvGetSize(sourceImage), 8, 3 );

	CvContourScanner cs;
	double blCoutourLength=0;
	CvSeq *sqMax=NULL;
	CvSeq * sq = NULL;
	int i=1;
	//void *pTmp=NULL;
	//cvRect cvRectTmp=cvRect(0,0,0,0);


	cvCvtColor(sourceImage, grayImage, CV_BGR2GRAY);
	//cvEqualizeHist(grayImage, grayImage);//useless

	//cvThreshold(grayImage, binaryImage, 100, 255, CV_THRESH_BINARY);

/*
	cvNamedWindow( "Source", 1 );
	cvShowImage( "Source", binaryImage );
	
	find_contour_num = cvFindContours( binaryImage, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	cvZero( dst );
	
	for( ; contour != 0; contour = contour->h_next )
	{
		printf("in");
		CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
		// 用1替代 CV_FILLED  所指示的轮廓外形 
		cvDrawContours( dst, contour, color, color, -1, CV_FILLED, 8 );
	}
	
	cvNamedWindow( "Components", 1 );
	cvShowImage( "Components", dst );
	cvWaitKey(0);
*/

	cvCanny(grayImage, binaryImage, 30, 30*3, 3); 	
	cs=cvStartFindContours(binaryImage,storage,sizeof(CvContour),CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

	do
	{
		sq=cvFindNextContour(cs);
		if(sq==NULL)
			break;
		double bl=cvContourPerimeter(sq);
		if(bl>blCoutourLength)
		{
			blCoutourLength=bl;
			sqMax=sq;//得到最大周长的轮廓的点
		}
		
		i++;
	}while(sq!=0);
	sq=cvEndFindContours(&cs);

	cvDrawContours( dst, sqMax, red, red, -1, 5/*CV_FILLED*/, 8 );

	cvNamedWindow( "Source", 1 );
	cvShowImage( "Source", binaryImage );
	cvNamedWindow( "Components", 1 );
	cvShowImage( "Components", dst );


	cvCvtColor(dst, grayImage, CV_BGR2GRAY);
	cvThreshold(grayImage, binaryImage, 250, 255, CV_THRESH_BINARY_INV);
	
	cvNamedWindow( "result", 1 );
	cvShowImage( "result", binaryImage );

	cvWaitKey(5000);
/*	
	cvConvert(binaryImage, binaryMat);
	//cvSmooth(binaryMat, binaryMat, CV_GAUSSIAN, 9, 0, 0);
	//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
	//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
	//cvConvert(binaryMat, binaryImage);
	
	cvZero(maskImage);
	cvConvert(maskImage,maskMat);
	for(y=0;y<height;y++)
		for(x=0;x<width;x++)
		{	 
			if(*(maskMat->data.fl+(y+1)*(width+2)+(x+1))==0 && *(binaryMat->data.fl+y*width+x)==0)
			{
				cvThreshold(maskImage,maskImage, 1, 128, CV_THRESH_BINARY);
				seed = cvPoint(x,y);
				b=255, g=255, r=255;
				brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
                
				cvFloodFill(binaryImage, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags, maskImage);   
				
				if(comp.area>max_area)
				{
					max_area=comp.area;
					max_area_seed_x=x;
					max_area_seed_y=y;
				}
			}
		}
		
		//printf("max_area = %f\n",max_area);
		
		if(max_area < BLOCK_THRESHOLD)
		{
			cvReleaseImage(&sourceImage);
			cvReleaseImage(&grayImage);
			cvReleaseImage(&binaryImage);
			cvReleaseImage(&maskImage);
			cvReleaseMat(&binaryMat);
			cvReleaseMat(&maskMat);
			binaryImage = NULL;
		}
		else
		{
			cvZero(maskImage);
			seed = cvPoint(max_area_seed_x,max_area_seed_y);
			b=255, g=255, r=255;
			brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
			
			cvFloodFill( binaryImage, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags,maskImage); 
			//printf("comp.area = %f\n",comp.area);
			
			cvConvert(maskImage,maskMat);
			
			for(int i=0;i<height;i++)
				for(int j=0;j<width;j++)
				{
					
					if(*(maskMat->data.fl+(i+1)*(width+2)+(j+1))!=0)
					{
						*(binaryMat->data.fl+i*width+j)=0;
					}
					else
					{
						*(binaryMat->data.fl+i*width+j)=255;
					}
				}
				
				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
				cvConvert(binaryMat, binaryImage);
		}
*/		
		return *binaryImage;
}

IplImage preprocessing(IplImage* imgSrc,int new_width, int new_height, int i = 100, int j = 100)
{
	IplImage* result;
	IplImage* scaledResult;
 
	CvMat data;
	CvMat dataA;
	CvRect bb;//bounding box
	CvRect bba;//boundinb box maintain aspect ratio
 
	//Find bounding box
	bb=findBB(imgSrc);
 
	//Get bounding box data and no with aspect ratio, the x and y can be corrupted
	cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
	
	//Create image with this data with width and height with aspect ratio 1
	//then we get highest size between width and height of our bounding box
	int size=(bb.width>bb.height) ? bb.width : bb.height;
	result=cvCreateImage( cvSize( size, size ), 8, 1 );
	cvSet(result,CV_RGB(255,255,255),NULL);

	//Copy the data in center of image
	int x=(int)floor((float)(size-bb.width)/2.0f);
	int y=(int)floor((float)(size-bb.height)/2.0f);
	bba = cvRect(x,y,bb.width, bb.height);
	cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));
	cvCopy(&data, &dataA, NULL);

	//Scale result
	scaledResult=cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
	cvResize(result, scaledResult, CV_INTER_NN);
	

	if( (i!=100) && (j!=100) )
	{
		if(j<10)
		{
			sprintf(result_file_name,"%s%d/%d0%d.jpg",result_file_path, i, i , j);
		}
		else
		{
			sprintf(result_file_name,"%s%d/%d%d.jpg",result_file_path, i, i , j);
		}				
		cvSaveImage(result_file_name, scaledResult);//
	}

				
	//Return processed data
	return *scaledResult;
}

void getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	for(i =0; i<classes; i++)
	{
		for( j = 0; j< train_samples; j++)
		{
 
			//Load file
			if(j<10)
			sprintf(file,"%s%d/%d0%d.jpg",file_path, i, i , j);
			else
			sprintf(file,"%s%d/%d%d.jpg",file_path, i, i , j);

			src_image = cvLoadImage(file,0);
			if(!src_image)
			{
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}

			//process file
			prs_image = preprocessing(src_image, size, size, i, j);
 
			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(i));
			//Set data
			cvGetRow(trainData, &row, i*train_samples + j);
 
			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);
 
			cvGetSubRect(img, &data, cvRect(0,0, size,size));
 
			CvMat row_header, *row1;
			//convert data matrix size * size to vector
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
	
	printf("finish training\n");
}


float classify(IplImage* img, int showResult)
{
	//CvKNearest knn = new CvKNearest( trainData, trainClasses, 0, false, K );
	CvKNearest knn( trainData, trainClasses, 0, false, K );


	IplImage prs_image;
	CvMat data;
	CvMat* nearest=cvCreateMat(1,K,CV_32FC1);
	float result;
	//process file
	prs_image = preprocessing(img, size, size);
	cvSaveImage("F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\Test\\NUMBER\\result.jpg", &prs_image);//

	//Set data
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );
 
	result=knn.find_nearest(row1,K,0,0,nearest,0);
 
	int accuracy=0;
	for(int i=0;i<K;i++)
	{
		if( nearest->data.fl[i] == result)
		accuracy++;
	}

	float pre=100*((float)accuracy/(float)K);
	if(showResult==1)
	{
		printf("|\t%.0f\t| \t%.2f%%  \t| \t%d of %d \t| \n",result,pre,accuracy,K);
		printf(" ---------------------------------------------------------------\n");
	}

	return result;
}

void test()
{
	IplImage* src_image;
	IplImage prs_image;
	//CvMat row,data;
	char file[255];
	int i,j;
	int error=0;
	int testCount=0;
	for(i =0; i<classes; i++)
	{
		for( j = train_samples; j<test_samples+train_samples; j++)
		{ 
			sprintf(file,"%s%d/%d%d.jpg",file_path, i, i , j);
			src_image = cvLoadImage(file,0);
			if(!src_image)
			{
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size,i,j);
			float r=classify(&prs_image,0);
			if((int)r!=i)
			{
				printf("misclassified image numbe:%d%d\n",i,j);
				error++;
			}
 
			testCount++;
		}
	}
	float totalerror=100*(float)error/(float)testCount;
	printf("System Error: %.2f%%\n", totalerror);
 
}


void batchNumberSegmentation()
{
	char fileNameSrc[255];
	char fileNameDst[255];
	//char filePathSrc[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/Data/";
	//char filePathDst[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/Result/";

	char filePathSrc[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/print data/Data/";
	char filePathDst[] = "F:/pre_study/OCR/MYCODE/MobileBrandOCR/print data/Result/";

	int x=0,y=0;
    int lo_diff = 0, up_diff = 0;
    int connectivity = 4;
    int new_mask_val = 255;
    int flags = connectivity + (new_mask_val << 8)+CV_FLOODFILL_FIXED_RANGE+CV_FLOODFILL_MASK_ONLY;
    CvConnectedComp comp;  
	double max_area=0;
	int max_x=0,max_y=0;
	int max_area_seed_x=0,max_area_seed_y=0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                              
	int b=0, g=0, r=0;
	CvPoint seed;
	CvScalar brightness;

	IplImage *mask = 0;

	for(int i =6; i<classes; i++)
	{
		for(int j = 0; j< 50+50; j++)
		{
			if(j<10)
			{
				sprintf(fileNameSrc,"%s%d/%d0%d.jpg",filePathSrc, i, i , j);
				sprintf(fileNameDst,"%s%d/%d0%d.jpg",filePathDst, i, i , j);
			}
			else
			{
				sprintf(fileNameSrc,"%s%d/%d%d.jpg",filePathSrc, i, i , j);
				sprintf(fileNameDst,"%s%d/%d%d.jpg",filePathDst, i, i , j);
			}

			//char* filename = "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\Data\\0\\002.jpg";

			if( (image = cvLoadImage( fileNameSrc, 1)) == 0 )
				continue;

			int width = image->width;
			int height = image->height;

			cedge = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 3);
			gray  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
			edge  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
			bina  = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
			mask  = cvCreateImage( cvSize(width+2,height+2),IPL_DEPTH_8U,1 );
			
			CvMat *binaryMat	= cvCreateMat(height, width,  CV_32FC1 );
			CvMat *maskMat	= cvCreateMat( height+2,width+2, CV_32FC1 );

			cvCvtColor(image, gray, CV_BGR2GRAY);


			cvThreshold( gray, bina, 50, 255, CV_THRESH_BINARY);

			cvConvert(bina, binaryMat);
			//cvSmooth(binaryMat, binaryMat, CV_GAUSSIAN, 9, 0, 0);
			//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
			//cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
			//cvConvert(binaryMat, bina);

			cvZero(mask);
			cvConvert(mask,maskMat);
			for(y=0;y<height;y++)
				for(x=0;x<width;x++)
				{	 
					if(*(maskMat->data.fl+(y+1)*(width+2)+(x+1))==0 && *(binaryMat->data.fl+y*width+x)==0)
					{
						cvThreshold(mask,mask, 1, 128, CV_THRESH_BINARY);
						seed = cvPoint(x,y);
						b=255, g=255, r=255;
						brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
                
						cvFloodFill(bina, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags, mask);   
                
						//printf("comp.area = %f\n",comp.area);

						if(comp.area>max_area)
						{
							max_area=comp.area;
							max_area_seed_x=x;
							max_area_seed_y=y;
						}
					}
				}

			printf("max_area = %f\n",max_area);

			if(max_area < BLOCK_THRESHOLD)
			{
				cvReleaseImage(&image);
				cvReleaseImage(&cedge);
				cvReleaseImage(&gray);
				cvReleaseImage(&edge);
				cvReleaseImage(&bina);
				cvReleaseMat(&binaryMat);
				cvReleaseMat(&maskMat);
				continue;
			}
			else
			{
				max_area = 0;//reset
				cvZero(mask);
				seed = cvPoint(max_area_seed_x,max_area_seed_y);
				b=255, g=255, r=255;
				brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
				
				cvFloodFill( bina, seed, brightness, cvRealScalar(lo_diff),cvRealScalar(up_diff), &comp, flags,mask); 
				printf("comp.area = %f\n",comp.area);

				cvConvert(mask,maskMat);

				for(int i=0;i<height;i++)
					for(int j=0;j<width;j++)
					{
						
						if(*(maskMat->data.fl+(i+1)*(width+2)+(j+1))!=0)
						{
							*(binaryMat->data.fl+i*width+j)=0;
						}
						else
						{
							*(binaryMat->data.fl+i*width+j)=255;
						}
					}

				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_OPEN,1);//Morphology
				cvMorphologyEx(binaryMat,binaryMat,NULL,NULL,CV_MOP_CLOSE,1);
				cvConvert(binaryMat, bina);
			}


			cvNamedWindow(wndname, 1);
			cvShowImage(wndname, bina);

			cvSaveImage(fileNameDst, bina);//
    
			//cvCreateTrackbar(tbarname, wndname, &edge_thresh, 100, on_trackbar);
			//on_trackbar(0);


			cvWaitKey(100);
			cvReleaseImage(&image);
			cvReleaseImage(&gray);
			cvReleaseImage(&edge);
			cvReleaseImage(&bina);
			cvReleaseImage(&mask);
			cvReleaseMat(&binaryMat);
			cvReleaseMat(&maskMat);
			cvDestroyWindow(wndname);
		}
	}
}

void on_trackbar(int h)
{
    cvSmooth( gray, edge, CV_BLUR, 3, 3, 0, 0 );
    cvNot( gray, edge );
    cvCanny(gray, edge, (float)edge_thresh, (float)edge_thresh*3, 3);  
    cvZero( cedge );
    cvCopy( image, cedge, edge );
    cvShowImage(wndname, cedge);
}

void main()
{
	/************************************************************************/
	/* batch segment                                                        */
	/************************************************************************

	batchNumberSegmentation();

	/************************************************************************/
	/*                                                                      */
	/************************************************************************/





	/************************************************************************/
	/* get feature vector and error rate                                    */
	/************************************************************************

	getData();//it is better to save the trained result into a file
	test();
	cvSave( "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\train data\\trainClasses.xml", trainClasses );
	cvSave( "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\train data\\trainData.xml", trainData );  

	/************************************************************************/
	/*                                                                      */
	/************************************************************************/



	/************************************************************************/
	/* recognize single one                                                 */
	/************************************************************************/
	float number;
	IplImage * loadedImage;
	IplImage segmentedImage;
	char* fileName = "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\Test\\NUMBER\\IMG3140A.jpg";
	char* trainDataFileName = "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\train data\\trainData.xml";
	char* trainClassesFileName = "F:\\pre_study\\OCR\\MYCODE\\MobileBrandOCR\\print data\\train data\\trainClasses.xml";
	  
	trainClasses = (CvMat*) cvLoad( trainClassesFileName ); 
	trainData = (CvMat*) cvLoad( trainDataFileName ); 	
	loadedImage = cvLoadImage(fileName,1);
	segmentedImage = numberSegmentation(loadedImage);
	//segmentedImage = numberSegmentationBasedOnContour(loadedImage);
	//number=classify(&segmentedImage,0);
	printf("number:%d\n",(int)number);
	/************************************************************************/
	/*                                                                      */
	/************************************************************************/
}
