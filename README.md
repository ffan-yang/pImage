# pImage
 
Usage : 

**For video compression / modification :**

``` python
import pImage

input_video_path = r"foo/myfoovideo.seq"
output_video_path = r"bar/myconvertedfoovideo.avi"

converter = pImage.converters.Standard_Converter(input_video_path, output_video_path, optionnal_key_value_arguments)
converter.start()
```

Once finished, the script will display `Done` in console
