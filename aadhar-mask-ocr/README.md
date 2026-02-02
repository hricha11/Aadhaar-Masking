Start the app

`` python -m uvicorn app:app --reload
``

Test for files 

`` python client.py``


OpenCV → preprocess image
        ↓
Tesseract → read digits
        ↓
OpenCV → draw white boxes (mask)
