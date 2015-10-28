# include <smmintrin.h> // sse 2 
# include <iostream>
# include <fstream>
# include <ctime>
# include <string>

using namespace std;

float * buffer = NULL, *buffer1 = NULL;
_MM_ALIGN16 class vector4 // class using SSE instructions // sets vector and performs subtraction
{
public:
	inline vector4() : mmvalue(_mm_setzero_ps()) {} // sets to zero
	inline vector4(float w, float x, float y, float z) : mmvalue(_mm_set_ps(w, z, y, x)) {}  // constructor
	inline void set(float w, float x, float y, float z)  
	{
		mmvalue = (_mm_set_ps(w, z, y, x)); // setter
	} 
	inline vector4(__m128 m) : mmvalue(m) {}
	void print () { cout << w << " " << x << " " << y << " " << z << endl; } // display values to user
	bool logicalor () { return (w||x||y||z); } // logical or . If even one of the values is  > 0, we realise that the images are not the same

	inline vector4 operator-(const vector4& b) const { return _mm_sub_ps(mmvalue, b.mmvalue); } // operator overloading

	union
	{
		struct { float w, x, y, z; };   
		__m128 mmvalue; // 128 bit data type 
	};
};

int ReadFiles () // function in C, using fstream functions to read bitmap images into arrays
{
	string name = "\0" , name1 = "\0";

	cout << "Please enter the name of the first image:" << endl;
	getline(cin, name); // name of the first image
	cout << "Please enter the name of the second image:" << endl;
	getline(cin, name1); // name of the second image
	// allocating memory 
	char * arr2 = new char [name.length()+1];
	char * arr3 = new char [name1.length()+1];
	int a = 0;
	for (; a < name.size(); a ++)
		arr2[a] = name[a];
	arr2[a] = '\0'; // placing a null character at the end of the string array

	for (a =0; a < name1.size(); a ++)
		arr3[a] = name1[a];
	arr3[a] = '\0';  // placing a null character at the end of the string array

	int length , length1 ; 	length1 = length = 0;
	char * arr, *arr1 =  0;
	ifstream is (arr2, std::ifstream::binary);

	if (is ) {
		is.seekg (0, is.end);
		length = is.tellg();
		is.seekg (0, is.beg); // seek the beginning after getting file size
		arr = new char [length];
		is.read (arr,length); // in built function

		if (is)
		{
		//	cout << "all characters read successfully." << endl ;
			ifstream it (arr3, std::ifstream::binary );
			if (it) {
				it.seekg (0, it.end);
				length1 = it.tellg();
				it.seekg (0, it.beg);
				arr1 = new char [length1];
				it.read (arr1,length1); // read second image

				if (it){
				//	cout << "all characters read successfully." << endl ;
					buffer = new float [length/4];
					buffer1 = new float [length1/4];
					for (int j = 0; j < length/4; j++){
						for (int i = 0; i < 4; ++i) 
							buffer[j] += arr[4*j+i] << (24 - (i) * 8); // using left shiftwise operator to convert char to float
					}

					for (int j = 0; j < length/4; j++){
						for (int i = 0; i < 4; ++i) 
							buffer1[j] += arr1[4*j+i] << (24 - (i) * 8); // using left shiftwise operator to convert char to float

					}
					
				
					if ( length == length1 && length > 0 )
						return length; // basic check. Two images are equal if their sizes are the same
					else 
						return 0;
				}
				else
					cout << "Error in reading file: " << name << endl;
				it.close();
			}
		}
		else
			cout << "Errorin reading file: " << name1 << endl;
		is.close();
	}
	return 0;
}

void LOandSub( int length)
{
	cout << "Using SSE for subtraction and logical OR: "<< endl;
	int a = 0, time = 0, i = 0, k = 0;

	vector4 v1,v2, v3, v4, v5, v6,v7, v8, result1, result2, result3, result4, resultOR;	

	for ( ; i < (length/(4*4*4)); i++ )
	{ 		  
		v1.set (buffer[16*i],buffer[16*i+1],buffer[16*i+2],buffer[16*i+3]); // load 4 floats into vector
		v2.set (buffer1[16*i],buffer1[16*i+1],buffer1[16*i+2],buffer1[16*i+3]);
		v3.set (buffer[16*i+4],buffer[16*i+5],buffer[16*i+6],buffer[16*i+7]);
		v4.set(buffer1[16*i+4],buffer1[16*i+5],buffer1[16*i+6],buffer1[16*i+7]);
		v5.set (buffer[16*8],buffer[16*i+9],buffer[16*i+10],buffer[16*i+11]);
		v6.set (buffer[16*8],buffer[16*i+9],buffer[16*i+10],buffer[16*i+11]);
		v7.set (buffer[16*i+12],buffer[16*i+13],buffer[16*i+14],buffer[16*i+15]);
		v8.set (buffer[16*i+12],buffer[16*i+13],buffer[16*i+14],buffer[16*i+15]);

		a = clock(); // set the clock
		result1 = _mm_sub_ps(v1.mmvalue, v2.mmvalue); // subtract two vectors that were previously set
		result2 = _mm_sub_ps(v3.mmvalue, v4.mmvalue);
		result3 = _mm_sub_ps(v5.mmvalue, v6.mmvalue);
		result4 = _mm_sub_ps(v7.mmvalue, v8.mmvalue);
		result2 = _mm_or_ps(_mm_or_ps(result1.mmvalue, result2.mmvalue),  _mm_or_ps(result3.mmvalue, result4.mmvalue)); // OR the results. If there is even one value > 0, it will be detected

		time += clock() - a; // end clock
		if(  result2.logicalor() ) // finally, we logically or the  w,x,y,z of result2 to obtain 1
			k++;
		
	}
	
	cout << "Number of differences: " << k*16 <<  " Total iterations: " << i << endl;
	cout << "Time: " << time << endl << endl;

}

void Sub( int length )
{
	cout << "Using SSE for subtraction only: "<< endl;
	int a = 0, time = 0 , k = 0, i  = 0;

	vector4 result1, v3, v4, result;
	
	for ( ; i < (length/(16*2)); i++ ) // parallelizing farther
	{ 		  
		v3.set (buffer[8*i],buffer[8*i+1],buffer[8*i+2],buffer[8*i+3]); // set the vector
		v4.set (buffer1[8*i],buffer1[8*i+1],buffer1[8*i+2],buffer1[8*i+3]);
		a = clock();
		result =  _mm_sub_ps(v3.mmvalue, v4.mmvalue); // perform subtraction using intrinsic
		time += clock()- a;
		v3.set (buffer[8*i+4],buffer[8*i+5],buffer[8*i+6],buffer[8*i+7]);
		v4.set (buffer1[8*i+4],buffer1[8*i+5],buffer1[8*i+6],buffer1[8*i+7]);
		a = clock();
		result1 =  _mm_sub_ps(v3.mmvalue, v4.mmvalue);
		time += clock()- a;
		float C = 0.0; // created for comparison
		a = clock();
		__asm // in line assembly
		{
			finit // empty the stack
				fld result.x; // load onto the stack
			fld C
				fcomi ST(0), ST(1)     // compare the top two values
				JE Below1 //if they are equal, search farther
				JNE Out1 // else jump out directly // this may decrease processing time
Below1:
			finit
				fld result.y;
			fld C
				fcomi ST(0), ST(1)
				JE Below2
				JNE Out1
Below2:
			fld result.w;
			fld C
				fcomi ST(0), ST(1)
				JE Below3
				JNE Out1
Below3:
			fld result.z;
			fld C
				fcomi ST(0), ST(1)
				JE Below4
				JNE Out1
Below4:
			finit
				fld result1.x;
			fld C
				fcomi ST(0), ST(1)     
				JE Below5
				JNE Out1
Below5:
			finit
				fld result1.y;
			fld C
				fcomi ST(0), ST(1)
				JE Below6
				JNE Out1
Below6:
			fld result1.w;
			fld C
				fcomi ST(0), ST(1)
				JE Below7
				JNE Out1
Below7:
			fld result1.z;
			fld C
				fcomi ST(0), ST(1)
				JE Below8
				JNE Out1
Out1:
		} // ending assembly
		time += clock()- a;
		k++;

		__asm
		{
Below8:		 // skip incrementing k
		}
		
	}
	
	cout << "Number of differences: " << k*8 <<  " Total iterations: " << i << endl;
	cout << "Time: " << time << endl << endl;
}

void Assembly( int length )
{
	cout << "Using x86 assembly: "<< endl;
	int  a = 0;
	int k  = 0, i = 0;

	a = clock();

	for ( ; i < (length/(4)); i++ )
	{ 	
		// move two values into temp variables
		float A = buffer[i]; 
		float B = buffer1[i];
		float C = 0.0; // for comparison 
		__asm // inline assembly
		{
			finit // empty the stack
				fld A // load first value
				fsub B // subtract from A
				fld C // load 0		     
				fcomi ST(0),ST(1) // perform comparison and set the flags
				JE Up // if they are equal, go to the end of the loop
				JNE Below // else jump below to increment k
Below:
		}

		k++;

		__asm
		{
Up:

		}

	}
	a = clock() - a; // final time value

	cout << "Number of differences: " << k <<  " Total iterations: " << i << endl;
	cout << "Time: " << a << endl << endl;
}



int main()
{
	system("Color 3F");// setting console

	cout << "\t\t\t Image Comparison!" << endl << endl; // header
	int length = ReadFiles (); // get the files
	cout << endl;
	if ( length != 0 ) // if they are not of unequal length
	{
			LOandSub(length); // SSE function
			Sub(length); // SSE + Assembly 
			Assembly(length); // Assembly only
	}
	else
		cout << "Error! Length: " << length << endl;
	system("pause");
	return 0;

	delete [] buffer;
	delete []buffer1;
}











