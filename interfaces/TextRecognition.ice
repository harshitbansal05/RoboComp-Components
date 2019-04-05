
#ifndef TEXTRECOGNITION_ICE
#define TEXTRECOGNITION_ICE

module RoboCompTextRecognition
{
    struct SText
    {
        int startX;
        int startY;
        int endX;
        int endY;
        string label;
    };

    sequence<SText> TextList;

    interface TextRecognition
    {
        idempotent void getTextList(out TextList textL);
    };
};

#endif
