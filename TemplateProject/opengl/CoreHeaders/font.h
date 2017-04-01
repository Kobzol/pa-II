////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	font.h
//
// summary:	Declaration and implementation of Font class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved for updates opny.
// 
// Bsed on	http://zenpandainteractive.blogspot.cz/2013/02/rendering-text-in-opengl-using-freetype.html
// and		http://musingsofninjarat.wordpress.com/opengl-freetype-texture-fonts/
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __FONT_H_
#define __FONT_H_
 
#include <ft2build.h>
#include <freetype.h>
#include <ftglyph.h>
#include <map>
#include <glm.hpp>
#include <string.h>

#include "shaderProgram.h"

/*
 * Glyph metrics:
 * --------------
 *
 *                       xmin                     xmax
 *                        |                         |
 *                        |<-------- width -------->|
 *                        |                         |    
 *              |         +-------------------------+----------------- ymax
 *              |         |    ggggggggg   ggggg    |     ^        ^
 *              |         |   g:::::::::ggg::::g    |     |        | 
 *              |         |  g:::::::::::::::::g    |     |        | 
 *              |         | g::::::ggggg::::::gg    |     |        | 
 *              |         | g:::::g     g:::::g     |     |        | 
 *   bearing_x -|-------->| g:::::g     g:::::g     |  bearing_y   | 
 *              |         | g:::::g     g:::::g     |     |        | 
 *              |         | g::::::g    g:::::g     |     |        | 
 *              |         | g:::::::ggggg:::::g     |     |        |  
 *              |         |  g::::::::::::::::g     |     |      height
 *              |         |   gg::::::::::::::g     |     |        | 
 *  baseline ---*---------|---- gggggggg::::::g-----*--------      |
 *            / |         |             g:::::g     |              | 
 *     origin   |         | gggggg      g:::::g     |              | 
 *              |         | g:::::gg   gg:::::g     |              | 
 *              |         |  g::::::ggg:::::::g     |              | 
 *              |         |   gg:::::::::::::g      |              | 
 *              |         |     ggg::::::ggg        |              | 
 *              |         |         gggggg          |              v
 *              |         +-------------------------+----------------- ymin
 *              |                                   |
 *              |------------- advance_x ---------->|
 */


#define NO_GLYPHS 95				// = ascii of '~' 126  minus ascii of ' ' 32 + one   = 95
#define MAX_TEXT_LENGTH 1024		// maximum size of the text that can be stored/rendered in VAO => sincle call of renderText method
#define BYTES_PER_CHARACTER_QUAD  (48*sizeof(float))  //4 * (x|y|z|w|r|g|b|a|s|t|.|.) = 4 * 12 * sizeof(float)

struct Glyph
{
	unsigned int dim[2];			//image dimensions = width, height
	int bearing[2];					//x, y bearing
	unsigned int advance[2];		//x, y advances
	float texCoord[2];				//texture coordinates
	unsigned int tex;				//glyph texture
};

class Font
{
private:
	FT_Face fontFace;
	glm::uint16_t fontSize;
	char fontFileName[1024];
	
    static FT_Library ftLibrary;			//General FT library info
    static bool ftLoaded;
	
	inline unsigned int nextPow2( unsigned int x ) 
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}

public:
	
	Glyph glyphs[NO_GLYPHS];

public:
	Font();
	~Font();

	bool loadFromFile(const char* fileName, const glm::uint16_t fontSize = 11);
	bool resize(const glm::uint32_t newFontSize);
	void setColor(const float r, const float g, const float b, const float a);

	static bool initLibrary();
};

inline bool Font::initLibrary()
{
    if(ftLoaded) return true;
    if(FT_Init_FreeType(&ftLibrary) != 0) return false;
    return (ftLoaded = true);
}

inline Font::Font()
{
	 if(!Font::initLibrary()) return;
}

inline Font::~Font()
{
	FT_Done_FreeType(ftLibrary ); 
}

inline bool Font::loadFromFile(const char *fileName, const glm::uint16_t fontSize)
{
	FT_Error error;
	FT_Glyph glyph;
	unsigned int glyphIndex;

	FT_GlyphSlot slot;
	FT_Bitmap *bitmap;

	unsigned int texId;
	unsigned int width, height;

	if(error = FT_New_Face(ftLibrary, fileName, 0, &fontFace) != 0)							// Create a new font face.
	{
		printf("FT_Error at line %d\n", __LINE__);
        FT_Done_FreeType(ftLibrary ); 
		return false;						
	}
 
	if(error = FT_Select_Charmap(fontFace, FT_ENCODING_UNICODE ) != 0)
	{
		printf("FT_Error at line %d\n", __LINE__);
        FT_Done_FreeType(ftLibrary ); 
		return false;						
	}

	if(error = FT_Set_Char_Size(fontFace, fontSize << 6, fontSize << 6, 72, 72) != 0)		// Set the size of the font face. Since the function expects a size in 1/64 pixels, we multiply  by 64 (same as left-shifting 6 bits) before passing. The 96 represents a 72-dpi font bitmap.
	{
		printf("FT_Error at line %d\n", __LINE__);
        FT_Done_FreeType(ftLibrary ); 
		return false;
	}

	//if(error = FT_Set_Pixel_Sizes(fontFace, 0, 120) != 0)										// Set the size of the font face IN PIXELS.
	//{
	//	printf("FT_Error at line %d\n", __LINE__);
	//	FT_Done_FreeType(ftLibrary ); 
	//	return false;
	//}

	for(size_t i = ' '; i <= '~'; ++i)														// Load all printable characters. If you visit an ASCII table (like www.asciitable.com) you will see  that the only valid values for printing are the space character all the way up to the tilde (~).
	{
		glyphIndex = FT_Get_Char_Index(fontFace, (FT_ULong)i);								// Locate the index of the character in the font face.
		if(glyphIndex == 0) continue;

		#pragma region Variant A
		//if (error = FT_Load_Char( fontFace, (FT_ULong)i, FT_LOAD_RENDER ) != 0)
		//{
		//	printf("FT_Error at line %d\n", __LINE__);
		//	FT_Done_Face( fontFace ); 
		//	FT_Done_FreeType(ftLibrary ); 
		//	return false;
		//}
		#pragma endregion
		
		#pragma region Variant B
		if (error = FT_Load_Glyph(fontFace, glyphIndex, FT_LOAD_RENDER) != 0)						// Load the glyph into the font face.
		{
			printf("FT_Error at line %d\n", __LINE__);
			FT_Done_Face( fontFace ); 
			FT_Done_FreeType(ftLibrary ); 
			return false;
		}
		FT_Get_Glyph(fontFace->glyph, &glyph);												// Put the glyph in the glyph slot into an actual glpyh struct.
		error = FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, 0, true);
		//error = FT_Render_Glyph(fontFace->glyph, FT_RENDER_MODE_NORMAL);					// Render the glyph in a format of 256 grayscale levels.
		#pragma endregion
 
		slot = fontFace->glyph;
		bitmap = &(slot->bitmap);
 
		width = nextPow2((unsigned int)bitmap->width);
		height = nextPow2((unsigned int)bitmap->rows);
 
		glGenTextures(1, &texId);
		glBindTexture(GL_TEXTURE_2D, texId);
 
		#pragma region ALPHA16 texture
		//GLubyte* expanded_data=(GLubyte *)malloc(sizeof(GLubyte)*2*width*height);
		//for (unsigned int k=0; k<height;k++)
		//{
		//	for (unsigned int l=0; l<width; l++)
		//	{
		//		//Flip bitmap
		//		expanded_data[2*(l+k*width)] = expanded_data[2*(l+k*width)+1]= (l>=bitmap->width||k>=bitmap->rows)? 0 : bitmap->buffer[((bitmap->rows - 1 - k) * bitmap->width) + l];
		//	}
		//}
		//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
		//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
		//glTexImage2D(GL_TEXTURE_2D,0,GL_ALPHA16,width,height, 0,GL_LUMINANCE_ALPHA,GL_UNSIGNED_BYTE, expanded_data);									// Restore default alignment value.
		//free((void *)expanded_data);
		#pragma endregion

		GLubyte* expanded_data=(GLubyte *)malloc(sizeof(GLubyte)*width*height);
		for (unsigned int k=0; k<height;k++)
		{
			for (unsigned int l=0; l<width; l++)
			{
				//Flip bitmap
				expanded_data[l + k*width] = (l >= (unsigned int)bitmap->width || k >= (unsigned int)bitmap->rows) ? 0 : bitmap->buffer[((bitmap->rows - 1 - k) * bitmap->width) + l];
			}
		}
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D,0,GL_R8,width,height, 0,GL_RED,GL_UNSIGNED_BYTE, expanded_data);									// Restore default alignment value.
		free((void *)expanded_data);


		FT_Done_Glyph(glyph);
		
		Glyph render_glyph;															// Our custom glyph structure.
		render_glyph.tex = texId;
		render_glyph.dim[0] = bitmap->width;												// total bitmap width
		render_glyph.dim[1] = bitmap->rows;													// total bitmap height
		render_glyph.bearing[0] = slot->bitmap_left;
		render_glyph.bearing[1] = slot->bitmap_top;
		render_glyph.advance[0] = slot->advance.x >> 6;										// advancement to the next character is represented in 1/64ths of a pixel so we need to convert them to accurate on-screen pixels.
		render_glyph.advance[1] = fontFace->glyph->metrics.horiBearingY >> 6;;				// height advance from the top of the character to the line if represented in 1/64ths of a pixel so we need to convert them to accurate on-screen pixels.
		//render_glyph.advance[1] = ((fontFace->glyph->metrics.horiBearingY - fontFace->glyph->metrics.height)>>6);
		render_glyph.texCoord[0] = (float)render_glyph.dim[0] / width;
		render_glyph.texCoord[1] = (float)render_glyph.dim[1] / height;
		
		glyphs[i-32] = render_glyph;														// Assign the glyph to its slot in the dictionary.// Assign the glyph to its slot in the dictionary.
	}

	FT_Done_Face(fontFace);																	// Clean up the font face.
	strcpy_s(fontFileName, 1024, fileName);													// Store the filename internally in-case we need to Resize() later.
	return true;
}

#endif